"""云侧相位调度器:工作纯相位批次。prefill 步只算 prompt 工作(prefill 首块 +
尾巴),decode 步只算已完结请求的 1-token 采样;prefill 批最多一个请求。"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue

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
    def _lwd_is_prefill_tail(request) -> bool:
        """RUNNING 且 prompt 未算完 = prefill 尾巴(尾巴只允许进 prefill 步)。"""
        return request.num_computed_tokens < request.num_prompt_tokens

    def _lwd_split_running(self) -> tuple[list, list]:
        """running 二分为 (decode_ready, tails),各自保持 FIFO 序。"""
        decode_ready: list = []
        tails: list = []
        for request in self.running:
            (tails if self._lwd_is_prefill_tail(request) else decode_ready).append(
                request
            )
        return decode_ready, tails

    def _lwd_has_prefill_tails(self) -> bool:
        return any(self._lwd_is_prefill_tail(r) for r in self.running)

    def _lwd_has_decode_ready(self) -> bool:
        return any(not self._lwd_is_prefill_tail(r) for r in self.running)

    def _schedule_pure_prefill(self) -> SchedulerOutput:
        """纯 prefill 步,批内最多一个请求:尾巴优先,否则放行 waiting 队首;
        其余全部藏起,跑原生 schedule() 后 finally 复原,均保 FIFO。"""
        decode_ready, tails = self._lwd_split_running()
        hidden_waiting = self.waiting
        self.waiting = create_request_queue(self.policy)
        hidden_tails: list = []
        if tails:
            hidden_tails = tails[1:]
            self.running = tails[:1]
        else:
            # 无尾巴时 running 清空,decode-ready 不混入 prefill 批
            self.running = []
            if hidden_waiting:
                self.waiting.add_request(hidden_waiting.peek_request())
        try:
            out = super().schedule()
        finally:
            # schedule 期间该列表 = 幸存尾巴 + 本步 waiting->running 的新请求
            leftover = self.waiting
            self.waiting = hidden_waiting
            while leftover:
                self.waiting.prepend_request(leftover.pop_request())
            self.running = decode_ready + self.running + hidden_tails
        # UP 链 seqno 随批下发云 worker(§9.12 数据面接缝):取登记表
        # 快照挂 SO 动态属性(无 slots 存活至 worker),worker 的 UP recv
        # 以此配对边侧发来的 embeds 张量(与边侧 SO.lwd_batch.seqno 同源)。
        # registry 由云引擎 IO 线程交付(缺省 = 未启用,挂空不扰原生)。
        if hasattr(self, "lwd_seqno_registry"):
            registry = self.lwd_seqno_registry
            out.lwd_up_seqnos = {
                request_id: list(registry.get(request_id, []))
                for request_id in out.num_scheduled_tokens
            }
        return out

    def _schedule_pure_decode(self) -> SchedulerOutput:
        """纯 decode 步:waiting 与 prefill 尾巴都藏起,批内只剩已完结请求的采样。"""
        hidden_waiting = self.waiting
        self.waiting = create_request_queue(self.policy)
        decode_ready, tails = self._lwd_split_running()
        self.running = decode_ready
        try:
            out = super().schedule()
        finally:
            newly_waiting = self.waiting
            self.waiting = hidden_waiting
            # 临时队列防御性清空并回插队首(保 FIFO)
            while newly_waiting:
                self.waiting.append(newly_waiting.popleft())
            # 被抢占请求已随上两行回插队首;复原被藏的尾巴(保 FIFO)
            self.running = tails + self.running
        return out

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
