"""云侧相位调度器:工作纯相位批次。prefill 步只算 prompt 工作(prefill 首块 +
尾巴),decode 步只算已完结请求的 1-token 采样;prefill 批最多一个请求。"""

from __future__ import annotations

from collections import deque

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import LwdBatch, LwdBatchType, SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.lwd_control.control_communication.lwd_notify import LwdRangeNotify
from vllm.v1.request import Request

logger = init_logger(__name__)

_LWD_PHASE_PREFILL_FIRST = "prefill_first"
_LWD_PHASE_DECODE_FIRST = "decode_first"


class LwdCloudPhaseScheduler(AsyncScheduler):
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
            "[Lwd] cloud phase scheduler: single-request prefill batches "
            "enforced (edge/cloud chunk stream stays per-request contiguous)"
        )

    def _lwd_resolve_phase(self) -> bool:
        """返回 True=prefill_first;缺省/未知相位告警回退 prefill_first。"""
        from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
            LwdConfig,
        )

        phase = LwdConfig.from_env_and_config(self.vllm_config).scheduler_name
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

    @staticmethod
    def _lwd_is_decode(request) -> bool:
        """prompt 已算完 = decode 态(可采样);未算完的是 prefill 尾巴。"""
        return request.num_computed_tokens >= request.num_prompt_tokens

    def _lwd_has_prefill_tails(self) -> bool:
        return any(not self._lwd_is_decode(r) for r in self.running)

    def _lwd_has_decode_ready(self) -> bool:
        return any(self._lwd_is_decode(r) for r in self.running)

    def _lwd_collect_decode_requests(self) -> list[str]:
        """收集所有 decode 态(prompt 已算完)请求的 req_id。

        按 running/waiting/skipped 顺序遍历三队列,输出可直接作为
        _lwd_schedule_for_visible_reqs 的入参。"""
        return [
            req.request_id
            for queue in (self.running, self.waiting, self.skipped_waiting)
            for req in queue
            if self._lwd_is_decode(req)
        ]

    # ------------------------------------------------------------------ #
    # Phase primitives(容器交换;原生 schedule() 零改动)                  #
    # ------------------------------------------------------------------ #
    def _lwd_new_queue(self, reqs: list[Request]) -> RequestQueue:
        """新建调度策略队列并装入 reqs。"""
        queue = create_request_queue(self.policy)
        for req in reqs:
            queue.add_request(req)
        return queue

    def _lwd_schedule_for_visible_reqs(self, req_ids: list[str]) -> SchedulerOutput:
        """把 req_ids 指定的请求从三队列剔除、单独调度,步后按原队列拼回。

        waiting/skipped 来源的请求走原生准入窗口(allocate/状态迁移/
        记账一样不少),running 来源的走续跑。拼回:仍被调度的接在隐藏
        running 之后,被抢占的排 waiting 尾部,被跳过的排 skipped 队首。"""
        picked = {self.requests[req_id] for req_id in req_ids}
        from_running = [req for req in self.running if req in picked]
        from_waiting = [req for req in self.waiting if req in picked]
        from_skipped = [req for req in self.skipped_waiting if req in picked]
        # 剔除后保存队列状态(隐藏集)
        self.running = [req for req in self.running if req not in picked]
        self.waiting.remove_requests(from_waiting)
        self.skipped_waiting.remove_requests(from_skipped)
        saved = (self.running, self.waiting, self.skipped_waiting)
        # 被剔除的请求按原队列归位成可见集,单独调度
        self.running = from_running
        self.waiting = self._lwd_new_queue(from_waiting)
        self.skipped_waiting = self._lwd_new_queue(from_skipped)
        try:
            out = super().schedule()
        finally:
            # 按原队列拼回:running 存活者接尾,waiting 被抢占者排队尾,
            # skipped 被跳过者排队首
            post = (self.running, self.waiting, self.skipped_waiting)
            self.running, self.waiting, self.skipped_waiting = saved
            self.running += post[0]
            self.waiting.extend(post[1])
            self.skipped_waiting.prepend_requests(post[2])
        return out

    def _schedule_pure_prefill(self) -> SchedulerOutput:
        """纯 prefill 步:prefill_notify_queue 有预告则取队首 msg,单独
        调度其请求(按原队列归位,waiting/skipped 来源走原生准入);没有则
        空集进窗口,等价空步,三队列原样保留。"""
        notify = q.popleft() if (q := self.prefill_notify_queue) else None
        if notify is not None and notify.request_id not in self.requests:
            # 请求已被 abort 释放:丢弃陈旧预告,本步按空集走
            notify = None
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
        # 的 UP recv 以此配对边侧发来的 embeds 张量。
        out.lwd_batch = LwdBatch(
            batch_type=LwdBatchType.LWD_EMBED,
            seqno=notify.seqno,
            batch_meta=None,
        )
        return out

    def _schedule_pure_decode(self) -> SchedulerOutput:
        """纯 decode 步:收集全部 decode 态请求,剔除单独调度后按落点拼回。"""
        return self._lwd_schedule_for_visible_reqs(
            self._lwd_collect_decode_requests()
        )

    @staticmethod
    def _is_empty(out: SchedulerOutput) -> bool:
        return out.total_num_scheduled_tokens == 0

    def _prefer_prefill(self) -> bool:
        """相位选择:prefill_first 有等待/尾巴即 prefill;
        decode_first 只要存在纯 decode 活就优先 decode。"""
        if self._lwd_prefill_first:
            return bool(self.waiting) or self._lwd_has_prefill_tails()
        return not self._lwd_has_decode_ready()

    def schedule(self) -> SchedulerOutput:
        prefer_prefill = self._prefer_prefill()
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
        self._last_step_was_prefill = False

        if prefer_prefill:
            out = self._schedule_pure_prefill()
            if self._is_empty(out) and self.running:
                # prefill 受 KV 压力阻塞:放行空步,下一步转 decode 泄压
                self._force_decode_once = True
            else:
                self._last_step_was_prefill = True
            return out
        out = self._schedule_pure_decode()
        if self._is_empty(out) and (self.waiting or self._lwd_has_prefill_tails()):
            # decode 无活但有 waiting/尾巴:翻回 prefill(不变量的空步出口)
            self._force_prefill_once = True
        return out
