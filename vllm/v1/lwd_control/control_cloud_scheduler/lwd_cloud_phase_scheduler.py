"""源 pure_phase_scheduler.py 照搬(§10.8),准入收敛 immediate(§10.14)。

= 纯相位批次策略:prefill 与 decode 不混批,相位优先级是部署决策
(prefill_first / decode_first)。请求经引擎门池转 Request 后走原生
add_request 随到随调度(immediate 直进),本调度器只负责批相位组成
—— 不做暂存/释放闸(§10.10 separate_phases 准入已按裁定删除,
对照基线随删)。

照搬差异清单:
  1. 类名映射:PurePhaseSchedulerBase / PrefillFirstPurePhaseScheduler /
     DecodeFirstPurePhaseScheduler → LwdCloudPhaseScheduler /
     LwdCloudPrefillFirstScheduler / LwdCloudDecodeFirstScheduler;
  2. 工厂解析收敛为按相位名一维选择 —— 准入固定 immediate,
     separate_phases 准入族与 LWD_CLOUD_IMMEDIATE_ADMISSION 开关已
     按裁定删除;
  3. curated 版的 is_prefill_chunk 队列手术(工作纯相位)整体删除,
     回到源"按人口分伙"形态:prefill 步只看 WAITING(已开动请求的
     prefill 尾巴停在 decode 步续算),decode 步只看 RUNNING。
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue

logger = init_logger(__name__)


class LwdCloudPhaseScheduler(AsyncScheduler):
    """Shared swap/fallback machinery; subclasses pick the phase order."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # One-shot flag: the last chosen phase produced an empty step and
        # the other population has work — try that phase next step.
        self._force_other_phase: bool = False

    # ------------------------------------------------------------------ #
    # Phase primitives                                                    #
    # ------------------------------------------------------------------ #
    def _schedule_pure_prefill(self) -> SchedulerOutput:
        """One step over WAITING only (RUNNING hidden)."""
        hidden_running = self.running
        self.running = []
        try:
            out = super().schedule()
        finally:
            newly_running = self.running
            self.running = hidden_running
            # Prefills scheduled this step moved waiting->running inside
            # the temp list; append AFTER the pre-existing running entries
            # (they are older, keep their decode priority order).
            self.running.extend(newly_running)
        return out

    def _schedule_pure_decode(self) -> SchedulerOutput:
        """One step over RUNNING only (WAITING hidden)."""
        hidden_waiting = self.waiting
        self.waiting = create_request_queue(self.policy)
        try:
            out = super().schedule()
        finally:
            newly_waiting = self.waiting
            self.waiting = hidden_waiting
            # The temp queue should be empty or hold nothing scheduled;
            # drain defensively back in front (FIFO order preserved).
            while newly_waiting:
                self.waiting.append(newly_waiting.popleft())
        return out

    @staticmethod
    def _is_empty(out: SchedulerOutput) -> bool:
        return out.total_num_scheduled_tokens == 0

    # ------------------------------------------------------------------ #
    # Subclass contract                                                   #
    # ------------------------------------------------------------------ #
    def _prefer_prefill(self) -> bool:
        """Phase choice for this step (before the one-shot fallback)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Entry point                                                         #
    # ------------------------------------------------------------------ #
    def schedule(self) -> SchedulerOutput:
        prefer_prefill = self._prefer_prefill()
        if self._force_other_phase:
            prefer_prefill = not prefer_prefill
            self._force_other_phase = False

        if prefer_prefill:
            out = self._schedule_pure_prefill()
            if self._is_empty(out) and self.running:
                # Prefill blocked (KV pressure: preemption found no RUNNING
                # requests in view).  Yield the empty cleanup step and let
                # the next step run decode so KV pressure can drain.
                self._force_other_phase = True
            return out
        out = self._schedule_pure_decode()
        if self._is_empty(out) and self.waiting:
            # Every running request is gated (async placeholders etc.) and
            # prefills are waiting — try a prefill step next.
            self._force_other_phase = True
        return out


class LwdCloudPrefillFirstScheduler(LwdCloudPhaseScheduler):
    """strict prefill-priority pure phases (the mode default).

    When WAITING is non-empty every step is a pure prefill batch (chunked
    prefill of long prompts and/or several waiting requests within the
    token budget).  When WAITING is empty, steps are pure decode: decode
    only runs once no prefill work remains (TTFT-first, no decode yield
    before a prefill step).
    """

    def _prefer_prefill(self) -> bool:
        return bool(self.waiting)


class LwdCloudDecodeFirstScheduler(LwdCloudPhaseScheduler):
    """decode-priority pure phases (throughput-first alternative).

    While any request is running, every step is a pure decode batch;
    prefills are scheduled only when no decode work is pending.  Kept as
    the second registry entry to make strategy evolution cheap.
    """

    def _prefer_prefill(self) -> bool:
        return not self.running


# Registry + resolution: 相位名 -> 类;准入固定 immediate 直进(§10.14),
# separate_phases 准入族已按裁定删除。
_LWD_CLOUD_DEFAULT_PHASE = "prefill_first"
_LWD_CLOUD_SCHEDULER_REGISTRY = {
    "prefill_first": LwdCloudPrefillFirstScheduler,
    "decode_first": LwdCloudDecodeFirstScheduler,
}


def get_pure_phase_scheduler_cls(name: str | None = None):
    """Resolve phase order by name (warn-and-fallback)."""
    phase = name or _LWD_CLOUD_DEFAULT_PHASE
    scheduler_cls = _LWD_CLOUD_SCHEDULER_REGISTRY.get(phase)
    if scheduler_cls is not None:
        return scheduler_cls
    logger.warning(
        "[Lwd] unknown cloud scheduler phase %r (known: %s); falling back to %s",
        phase,
        sorted(_LWD_CLOUD_SCHEDULER_REGISTRY),
        _LWD_CLOUD_DEFAULT_PHASE,
    )
    return _LWD_CLOUD_SCHEDULER_REGISTRY[_LWD_CLOUD_DEFAULT_PHASE]
