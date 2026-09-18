"""云侧相位调度器:纯相位批次(prefill 批最多一个请求);步元数据
(c2e)经 update_from_output 覆写在原生入账后发边。"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.sched.output import (
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    SchedulerOutput,
)
from vllm.v1.engine import FinishReason
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LWD_NOT_FINISHED,
    LwdC2eNotify,
    LwdRangeNotify,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
    LwdReqPhase,
)
from vllm.v1.lwd_debug import LwdDebug

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreOutputs, ModelRunnerOutput
    from vllm.v1.outputs import LwdC2eMeta

logger = init_logger(__name__)

_LWD_PHASE_PREFILL_FIRST = "prefill_first"
_LWD_PHASE_DECODE_FIRST = "decode_first"

class LwdCloudScheduler(LwdBaseScheduler):
    """工作纯相位批次策略;相位(prefill_first/decode_first)构造期自解析。
    前置约束:不兼容 spec decode(eagle 会 shift num_computed_tokens,纯度判据失真)。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lwd_prefill_first = self._lwd_resolve_phase()
        # One-shot 翻转:某相位空步而另一相位有活时,强制下一步走后者
        self._force_prefill_once: bool = False
        self._force_decode_once: bool = False
        # 驱动禁连续 prefill 不变量
        self._last_step_was_prefill: bool = False
        # prefill 通知队列:边侧范围预告逐条入队,一步弹一条点名
        self.prefill_notify_queue: deque[LwdRangeNotify] = deque()
        # 步末待发 c2e(update_from_output 入列,引擎 post_step 冲刷)
        self.pending_c2e: list[LwdC2eNotify] = []
        logger.info(
            "[Lwd] cloud scheduler: single-request prefill batches "
            "enforced (edge/cloud chunk stream stays per-request contiguous)"
        )

    def _lwd_resolve_phase(self) -> bool:
        """返回 True=prefill_first;未知相位告警回退 prefill_first。"""
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
        """收集三队列全部 decode 态请求(含被抢占回 waiting 的),
        可见集单独调度。"""
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
        """弹一条范围预告点名其请求,可见集调度;空步(KV 压力未准入)
        回塞队首重试,准入后挂 EMBED 批。"""
        notify = q.popleft() if (q := self.prefill_notify_queue) else None
        if notify is not None and notify.request_id not in self.requests:
            # 请求已被 abort 释放:丢弃陈旧预告
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
            self.prefill_notify_queue.appendleft(notify)
            return out
        # 占位 token 行数必须等于边侧实际发送数(HCCL P2P 要求两端
        # numel 匹配);seqno 为 UP 链配对号,与边侧 EMBED 批同源同值
        out.lwd_batch = LwdBatch(
            batch_type=LwdBatchType.LWD_EMBED,
            seqno=notify.seqno,
            batch_meta=LwdEmbedBatch(
                req_ids=[notify.request_id],
                token_ids=[[0] * notify.num_tokens],
            ),
        )
        return out

    def _lwd_has_prefill_work(self) -> bool:
        """waiting/running 存在 PREFILL 相位请求(被抢占回 waiting 的
        decode 请求不算,由纯 decode 步的三队列收集服务)。"""
        return any(
            self._lwd_the_phase_of_req(req) is LwdReqPhase.PREFILL
            for queue in (self.waiting, self.running)
            for req in queue
        )

    def _lwd_select_phase(self) -> LwdReqPhase:
        """相位选择:prefill_first 有 prefill 活即 prefill,decode_first
        有 decode 活即 decode;消费 One-shot 强制标志与禁连续 prefill
        不变量(被抢占回 waiting 的 decode 请求不算 prefill 活)。"""
        has_prefill_work = self._lwd_has_prefill_work()
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
        # 不变量:prefill 不连续两步,空 decode 步经 _force_prefill_once 翻回
        if prefer_prefill and self._last_step_was_prefill and self.running:
            prefer_prefill = False
        return LwdReqPhase.PREFILL if prefer_prefill else LwdReqPhase.DECODE

    def _lwd_after_phase(self, phase: LwdReqPhase, out: SchedulerOutput) -> None:
        """空步翻转与 last_prefill 簿记(decode 步不改变 PREFILL 相位
        成员,翻转条件就地重扫与选择时等价)。"""
        if phase is LwdReqPhase.PREFILL:
            if not out.total_num_scheduled_tokens and self.running:
                # prefill 受 KV 压力阻塞:下一步转 decode 泄压
                self._force_decode_once = True
            else:
                self._last_step_was_prefill = True
            return
        self._last_step_was_prefill = False
        if not out.total_num_scheduled_tokens and self._lwd_has_prefill_work():
            # decode 无活但有 prefill 活:翻回 prefill
            self._force_prefill_once = True

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_output: "ModelRunnerOutput",
    ) -> "dict[int, EngineCoreOutputs]":
        """原生入账后解 pinned 载荷组 c2e 入 pending_c2e(纯提取,
        不做 I/O;引擎 post_step 统一冲刷发边——边侧据此预挂 DOWN
        recv);carrier 缺席仅透传。"""
        engine_core_outputs = super().update_from_output(scheduler_output, model_output)
        carrier = getattr(model_output, "lwd_down_carrier", None)
        if carrier is None:
            return engine_core_outputs
        meta = self._lwd_carrier_to_meta(carrier)
        LwdDebug.cloud_step(self, meta, engine_core_outputs)  # [lwd-debug]
        self.pending_c2e.append(LwdC2eNotify(
            hidden_num_elements=meta.hidden_num_elements,
            top_id_ths=meta.top_id_ths,
            num_accepted_tokens=meta.num_accepted_tokens,
            req_ids=meta.req_ids,
            finish_reasons=self._lwd_c2e_finish_reasons(meta, engine_core_outputs),
            down_seqno=meta.down_seqno,
        ))
        return engine_core_outputs

    @staticmethod
    def _lwd_carrier_to_meta(carrier) -> "LwdC2eMeta":
        """pinned 载荷解包为步元数据;布局 [ranks(各段行)...,
        counts(accepted/请求)..., seg_lens(段长/请求)...]。"""
        from vllm.v1.outputs import LwdC2eMeta

        pinned, req_ids, hidden_numel, seqno = carrier
        n_req = len(req_ids)
        vals = pinned.tolist()
        counts = vals[-2 * n_req : -n_req]
        seg_lens = vals[-n_req:]
        ranks_flat = vals[: -2 * n_req]
        top_id_ths: list[list[int]] = []
        off = 0
        for seg_len in seg_lens:
            top_id_ths.append(ranks_flat[off : off + seg_len])
            off += seg_len
        return LwdC2eMeta(
            hidden_num_elements=hidden_numel,
            top_id_ths=top_id_ths,
            num_accepted_tokens=list(counts),
            req_ids=list(req_ids),
            down_seqno=seqno,
        )

    @staticmethod
    def _lwd_c2e_finish_reasons(
        meta: "LwdC2eMeta",
        engine_core_outputs: "dict[int, EngineCoreOutputs]",
    ) -> list[int]:
        """req_ids 对齐的完成码:带 finish_reason 的输出取其码,仅进
        finished_requests 的缺口按 ABORT 兜底,其余 NOT_FINISHED。"""
        reasons: dict[str, int] = {}
        for outputs in engine_core_outputs.values():
            for out in outputs.outputs:
                if out.finish_reason is not None:
                    reasons.setdefault(out.request_id, int(out.finish_reason))
            for request_id in outputs.finished_requests or ():
                reasons.setdefault(request_id, int(FinishReason.ABORT))
        return [
            reasons.get(request_id, LWD_NOT_FINISHED)
            for request_id in meta.req_ids
        ]

