"""源 pure_phase_scheduler.py 全文件照搬(§10.8)+ 相位准入收编(§10.10)。

= 纯相位批次策略:prefill 与 decode 不混批,相位优先级是部署决策
(prefill_first / decode_first)。相位准入(separate_phases / immediate)
自源 ActiveEdgeCloudEngineCore._apply_scheduling_policy + 准入策略族收编:
add_request 进暂存池,schedule() 顶部过释放闸(调度器完全排空才整批
放行)。§10.14 主线程化后,首预告门与 PRE_OUT 处理属线上语义,住云引擎
子类(lwd_cloud_engine);本调度器只保留调度纪律 —— 暂存池/释放闸与
纯相位排批,不含任何控制面接口。

照搬差异清单:
  1. 类名映射:PurePhaseSchedulerBase / PrefillFirstPurePhaseScheduler /
     DecodeFirstPurePhaseScheduler → LwdCloudPhaseScheduler /
     LwdCloudPrefillFirstScheduler / LwdCloudDecodeFirstScheduler;
  2. 工厂扩展为 (scheduler_name, admission_name) 二维解析,注册表折叠
     源 BATCH_POLICY_REGISTRY 与准入策略族两张表;lwd_cloud_admission.py
     随收编删除;
  3. curated 版的 is_prefill_chunk 队列手术(工作纯相位)整体删除,
     回到源"按人口分伙"形态:prefill 步只看 WAITING(已开动请求的
     prefill 尾巴停在 decode 步续算),decode 步只看 RUNNING;
  4. 准入语义保真(源 SeparatePhasesPolicy/ImmediateAdmissionPolicy):
     释放条件 unfinished == 0、max_num_seqs <=0 不截断、溢出留待下一轮
     排空相;immediate 即 staging 直通(原生等价),两纪律折叠为类属性
     LWD_CLOUD_IMMEDIATE_ADMISSION(部署决策 = 类身份,经 scheduler_cls
     注入,跨进程按模块引用序列化安全);
  5. §10.11 控制面功能接口收编已被 §10.14 取代:首预告门/请求工厂/
     出口绑定迁云引擎子类(主线程收发,免 marshal),本文件零控制面
     接口、零控制面状态。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vllm.v1.request import Request

logger = init_logger(__name__)


class LwdCloudPhaseScheduler(AsyncScheduler):
    """Shared swap/fallback machinery; subclasses pick the phase order."""

    # immediate 变体置 True:staging 直通恢复原生混合(源 Immediate 策略,
    # 分离的关闭开关/对照基线);默认 False = separate_phases 暂存纪律。
    LWD_CLOUD_IMMEDIATE_ADMISSION: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # One-shot flag: the last chosen phase produced an empty step and
        # the other population has work — try that phase next step.
        self._force_other_phase: bool = False
        # separate_phases 暂存池(rid -> Request,FIFO):已过首预告门的
        # 请求在此等排空窗口,未进原生簿记(self.requests 不可见,源
        # pending 池同义),故不计入 unfinished —— 否则释放条件永假。
        self._staged: OrderedDict[str, Request] = OrderedDict()

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
    # 相位准入(收编自源准入策略族)                                      #
    # ------------------------------------------------------------------ #
    def _lwd_release_staged(self) -> None:
        """separate_phases 释放闸:调度器完全排空才整批放行。

        源 SeparatePhasesPolicy 语义:unfinished == 0 才放行(每批生命
        周期内无新请求搅入),整批截断到 max_num_seqs,溢出留待下一轮
        排空相;<=0 不截断。放行走原生 add_request 全路径(native
        簿记/connector/统计事件一个不少)。
        """
        if not self._staged or self.get_num_unfinished_requests() > 0:
            return
        limit = self.scheduler_config.max_num_seqs
        release_ids = list(self._staged)[:limit] if limit > 0 else list(self._staged)
        for rid in release_ids:
            super().add_request(self._staged.pop(rid))
            logger.info("[Lwd] cloud released staged request %s", rid)

    def add_request(self, request: Request) -> None:
        if self.LWD_CLOUD_IMMEDIATE_ADMISSION:
            super().add_request(request)
            return
        rid = request.request_id
        if rid in self._staged:
            logger.warning("[Lwd] duplicate staged request %s ignored", rid)
            return
        self._staged[rid] = request

    def has_requests(self) -> bool:
        """暂存池非空即有工作:排空窗口到达时 schedule()(含释放闸)可被驱动。"""
        return bool(self._staged) or super().has_requests()

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        """暂存中的请求就地摘除(未入原生簿记,无 KV/队列需释放)。"""
        if request_ids is None:
            staged_hits = set(self._staged)
            native_ids = None
        else:
            ids = {request_ids} if isinstance(request_ids, str) else set(request_ids)
            staged_hits = ids & self._staged.keys()
            native_ids = ids - staged_hits
        # 按池内 FIFO 序摘除;先取快照再 pop(遍历中变异 OrderedDict 会炸)。
        staged_ids = [rid for rid in self._staged if rid in staged_hits]
        finished = [(rid, self._staged.pop(rid).client_index) for rid in staged_ids]
        return finished + super().finish_requests(native_ids, finished_status)

    # ------------------------------------------------------------------ #
    # Entry point                                                         #
    # ------------------------------------------------------------------ #
    def schedule(self) -> SchedulerOutput:
        # 释放闸先于选相:放行的请求当步即参与 prefill 相排批。
        self._lwd_release_staged()
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


class LwdCloudPrefillFirstImmediateScheduler(LwdCloudPhaseScheduler):
    """prefill_first 相位 + immediate 准入:staging 直通(原生等价)。"""

    LWD_CLOUD_IMMEDIATE_ADMISSION = True

    def _prefer_prefill(self) -> bool:
        return bool(self.waiting)


class LwdCloudDecodeFirstImmediateScheduler(LwdCloudPhaseScheduler):
    """decode_first 相位 + immediate 准入:staging 直通(原生等价)。"""

    LWD_CLOUD_IMMEDIATE_ADMISSION = True

    def _prefer_prefill(self) -> bool:
        return not self.running


# Registry + resolution: (相位序, 准入纪律) -> 类;折叠源两张策略表(§10.10)。
_LWD_CLOUD_DEFAULT_PHASE = "prefill_first"
_LWD_CLOUD_DEFAULT_ADMISSION = "separate_phases"

_LWD_CLOUD_SCHEDULER_REGISTRY = {
    (_LWD_CLOUD_DEFAULT_PHASE, _LWD_CLOUD_DEFAULT_ADMISSION): (
        LwdCloudPrefillFirstScheduler
    ),
    ("prefill_first", "immediate"): LwdCloudPrefillFirstImmediateScheduler,
    ("decode_first", "separate_phases"): LwdCloudDecodeFirstScheduler,
    ("decode_first", "immediate"): LwdCloudDecodeFirstImmediateScheduler,
}


def get_pure_phase_scheduler_cls(
    name: str | None = None, admission_name: str | None = None
):
    """Resolve (phase order, admission discipline) by names (warn-and-fallback)."""
    phase = name or _LWD_CLOUD_DEFAULT_PHASE
    admission = admission_name or _LWD_CLOUD_DEFAULT_ADMISSION
    scheduler_cls = _LWD_CLOUD_SCHEDULER_REGISTRY.get((phase, admission))
    if scheduler_cls is not None:
        return scheduler_cls
    logger.warning(
        "[Lwd] unknown cloud scheduler policy (%r, %r) (known: %s); falling "
        "back to (%s, %s)",
        phase,
        admission,
        sorted(_LWD_CLOUD_SCHEDULER_REGISTRY),
        _LWD_CLOUD_DEFAULT_PHASE,
        _LWD_CLOUD_DEFAULT_ADMISSION,
    )
    return _LWD_CLOUD_SCHEDULER_REGISTRY[
        (_LWD_CLOUD_DEFAULT_PHASE, _LWD_CLOUD_DEFAULT_ADMISSION)
    ]
