"""云侧相位调度器:工作纯相位批次。prefill 步只算 prompt 工作(prefill 首块 +
尾巴),decode 步只算已完结请求的 1-token 采样;prefill 批最多一个请求。"""

from __future__ import annotations

from collections import deque

from vllm.logger import init_logger
from vllm.v1.core.sched.output import (
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    SchedulerOutput,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import LwdRangeNotify
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
    LwdReqPhase,
)

logger = init_logger(__name__)

_LWD_PHASE_PREFILL_FIRST = "prefill_first"
_LWD_PHASE_DECODE_FIRST = "decode_first"


class LwdCloudScheduler(LwdBaseScheduler):
    """工作纯相位批次策略;相位(prefill_first/decode_first)构造期自解析。
    前置约束:不兼容 spec decode(eagle 会 shift num_computed_tokens,纯度判据失真)。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lwd_prefill_first = self._lwd_resolve_phase()
        # One-shot 翻转:某相位空步而另一相位有活时,强制下一步走后者;
        # 双标志显式定向,按偏好取反会错翻,造成空步死循环。
        self._force_prefill_once: bool = False
        self._force_decode_once: bool = False
        # 上一个非空步是否为 prefill,驱动 schedule() 的禁连续 prefill 不变量
        self._last_step_was_prefill: bool = False
        # prefill 通知队列:边侧范围预告(RangeNotify)逐条入队,每步取
        # 队首点名其 request_id;预告自带 seqno 即本步 UP 链配对号
        self.prefill_notify_queue: deque[LwdRangeNotify] = deque()
        logger.info(
            "[Lwd] cloud scheduler: single-request prefill batches "
            "enforced (edge/cloud chunk stream stays per-request contiguous)"
        )

    def _lwd_resolve_phase(self) -> bool:
        """返回 True=prefill_first;缺省/未知相位告警回退 prefill_first。"""
        phase = self.vllm_config.lwd_config.scheduler_name
        if phase == _LWD_PHASE_DECODE_FIRST:
            return False
        if phase != _LWD_PHASE_PREFILL_FIRST:
            logger.warning(
                "[Lwd] unknown cloud scheduler phase %r "
                "(known: %s); falling back to %s",
                phase,
                [_LWD_PHASE_DECODE_FIRST, _LWD_PHASE_PREFILL_FIRST],
                _LWD_PHASE_PREFILL_FIRST,
            )
        return True

    def schedule_decode(self) -> SchedulerOutput:
        """纯 decode 步:收集三队列全部 decode 态请求(含被抢占回
        waiting 的),可见集单独调度。"""
        req_ids = [
            req.request_id
            for queue in (self.running, self.waiting, self.skipped_waiting)
            for req in queue
            if self._lwd_the_phase_of_req(req) is LwdReqPhase.DECODE
        ]
        if req_ids:
            logger.info("[Lwd][cloud-sched] decode reqs=%s", req_ids)
        return self._lwd_schedule_for_visible_reqs(req_ids)

    def schedule_prefill(self) -> SchedulerOutput:
        """纯 prefill 步:prefill_notify_queue 有预告则取队首 msg,单独
        调度其请求(按原队列归位,waiting/skipped 来源走原生准入);没有则
        空集进窗口,等价空步,三队列原样保留。"""
        notify = q.popleft() if (q := self.prefill_notify_queue) else None
        if notify is not None and notify.request_id not in self.requests:
            # 请求已被 abort 释放:丢弃陈旧预告,本步按空集走
            notify = None
        if notify is not None:
            logger.info(
                "[Lwd][cloud-sched] prefill notify req=%s seqno=%s num=%s",
                notify.request_id, notify.seqno, notify.num_tokens,
            )
        req_ids = [notify.request_id] if notify is not None else []
        out = self._lwd_schedule_for_visible_reqs(req_ids)
        if notify is None:
            return out
        if not out.num_scheduled_tokens:
            # 未实际准入(典型 KV 压力空步):预告塞回队首原位,decode
            # 泄压后重新点名;本步不挂 lwd_batch,不向 worker 预告配对号
            self.prefill_notify_queue.appendleft(notify)
            return out
        # UP 链 seqno 随批下发云 worker(§9.12 数据面接缝):批配对号直接
        # 取点名预告自带的 seqno(与边侧 EMBED 批派发号同源同值),worker
        # 的 UP recv 以此配对边侧发来的 embeds 张量。batch_meta 承载
        # worker 的 recv 尺寸与注入切行信息:req_ids 取预告请求(单请求
        # 批),token_ids 为占位列表——长度必须等于边侧实际发送的 chunk
        # token 数(= RangeNotify.num_tokens),recv numel 才能与边侧
        # isend 严格相等(HCCL P2P 要求两端 numel 匹配)。
        out.lwd_batch = LwdBatch(
            batch_type=LwdBatchType.LWD_EMBED,
            seqno=notify.seqno,
            batch_meta=LwdEmbedBatch(
                req_ids=[notify.request_id],
                token_ids=[[0] * notify.num_tokens],
            ),
        )
        return out

    def _lwd_select_phase(self) -> LwdReqPhase:
        """相位选择:prefill_first 有 prefill 活即 prefill;decode_first
        只要存在 decode 活就优先 decode。One-shot 强制标志与禁连续
        prefill 不变量在此消费。

        相位工作量直判:prefill 活 = waiting/running 存在 PREFILL 相位
        请求(被抢占回 waiting 的 DECODE 请求不算 prefill 活,由纯
        decode 步的三队列收集服务);decode 活 = running 存在 DECODE。"""
        has_prefill_work = any(
            self._lwd_the_phase_of_req(req) is LwdReqPhase.PREFILL
            for queue in (self.waiting, self.running)
            for req in queue
        )
        has_decode_work = any(
            self._lwd_the_phase_of_req(req) is LwdReqPhase.DECODE
            for req in self.running
        )
        prefer_prefill = (
            has_prefill_work if self._lwd_prefill_first else not has_decode_work
        )
        if self._force_prefill_once:
            self._force_prefill_once = False
            prefer_prefill = True
        elif self._force_decode_once:
            self._force_decode_once = False
            prefer_prefill = False
        # 不变量:prefill 不连续执行两步;仅当中间的 decode 步为空时才允许
        # 连续,空 decode 步经 _force_prefill_once 翻回。
        if prefer_prefill and self._last_step_was_prefill and self.running:
            prefer_prefill = False
        return LwdReqPhase.PREFILL if prefer_prefill else LwdReqPhase.DECODE

    def _lwd_after_phase(self, phase: LwdReqPhase, out: SchedulerOutput) -> None:
        """空步翻转与禁连续 prefill 簿记。

        decode 步不会改变 PREFILL 相位的成员(可见集只含 decode 态,
        被抢占者回 waiting 后相位不变),翻转条件就地重扫与选择时直判等价。"""
        if phase is LwdReqPhase.PREFILL:
            if not out.total_num_scheduled_tokens and self.running:
                # prefill 受 KV 压力阻塞:放行空步,下一步转 decode 泄压
                self._force_decode_once = True
            else:
                self._last_step_was_prefill = True
            return
        self._last_step_was_prefill = False
        if not out.total_num_scheduled_tokens and any(
            self._lwd_the_phase_of_req(req) is LwdReqPhase.PREFILL
            for queue in (self.waiting, self.running)
            for req in queue
        ):
            # decode 无活但有 prefill 活:翻回 prefill(不变量的空步出口)
            self._force_prefill_once = True
