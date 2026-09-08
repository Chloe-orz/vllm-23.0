"""源 pure_phase_scheduler.py 全文件照搬(§10.8)+ 相位准入收编(§10.10)
+ 控制面功能接口二次收编(§10.11)。

= 纯相位批次策略:prefill 与 decode 不混批,相位优先级是部署决策
(prefill_first / decode_first)。相位准入(separate_phases / immediate)
自源 ActiveEdgeCloudEngineCore._apply_scheduling_policy + 准入策略族收编:
add_request 进暂存池,schedule() 顶部过释放闸(调度器完全排空才整批
放行)。首预告门(chunk-0 就绪)与 PRE_OUT 三类通知的调度侧处理经
lwd_cloud_on_*_notify 接口收编(§10.11 用户裁定,偏离方案 §2.3"门留
桥线程"备注):LwdCloudCore 泵做线上语义(offset==0 判首)后转发,
过门请求经装配层 request factory 建请求进暂存池 —— 调度器不触达
传输对象与 EngineCore。

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
  5. 控制面功能接口自 LwdCloudCore 移入(§10.11):首预告门状态、
     请求/预告/abort 三处理接口、消费水位推导、统计;
  6. 桥线程侧注入(§10.12):提升出口/abort 出口经 lwd_cloud_bind_bridge
     绑定(缺省直进,调度状态变更 marshal 到循环线程,泵线程只碰门);
     shutdown() 覆写转发停桥(原生 scheduler.shutdown 钩子,core.py 零
     改动);LwdCloudCore 整文件删除,泵与数据面接缝迁装配层桥线程。
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
    from collections.abc import Callable, Iterable

    from vllm.v1.lwd_control.control_communication.lwd_notify import (
        LwdRequestNotify,
    )
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
        # 首预告门状态(§10.11 自 LwdCloudCore 移入):_lwd_gate_pending
        # 收未过门的线上请求元数据,_lwd_gate_ready 记已收首预告的 rid。
        self._lwd_gate_pending: OrderedDict[str, LwdRequestNotify] = OrderedDict()
        self._lwd_gate_ready: set[str] = set()
        # 请求工厂(装配层绑定):线上元数据 -> Request(L3 唯一建请求点)
        self._lwd_request_factory: Callable | None = None
        # 桥线程侧注入(§10.12,缺省 = 单线程直进/自终结,供直连形态与单测):
        # 提升出口与 abort 出口把调度状态变更 marshal 到循环线程,桥线程只碰门。
        self._lwd_admit_sink: Callable | None = None
        self._lwd_abort_sink: Callable | None = None
        self._lwd_bridge_stop: Callable | None = None
        # 消费水位簿记(§9.1 裁:传输点未接,推导保留待回接)
        self._lwd_consumed_sent: dict[str, int] = {}

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
    # Phase admission(收编自源准入策略族)                                #
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
    # 控制面功能接口(§10.11 自 LwdCloudCore 移入;泵侧做线上语义)        #
    # ------------------------------------------------------------------ #
    def lwd_cloud_bind_request_factory(
        self, factory: Callable[[LwdRequestNotify], Request]
    ) -> None:
        """绑定请求工厂(装配层唯一建请求点,Request/SamplingParams 留 L3)。"""
        self._lwd_request_factory = factory
        logger.info("[Lwd] cloud request factory bound")

    def lwd_cloud_on_request_notify(self, wire: LwdRequestNotify) -> None:
        """请求预告:未过门先住门池;首预告已先行(乱序防御)则即过门。"""
        rid = wire.request_id
        if rid in self._lwd_gate_pending or rid in self._staged:
            logger.warning("[Lwd] duplicate request metadata %s ignored", rid)
            return
        self._lwd_gate_pending[rid] = wire
        if rid in self._lwd_gate_ready:
            self._lwd_promote(rid)

    def lwd_cloud_on_range_notify(self, request_id: str) -> None:
        """首预告就绪(offset==0 由泵侧判定):过门即建请求进暂存/准入。"""
        self._lwd_gate_ready.add(request_id)
        if request_id in self._lwd_gate_pending:
            self._lwd_promote(request_id)

    def lwd_cloud_on_abort_notify(self, request_id: str) -> None:
        """abort 预告:门池/门标记就地清理(桥线程独占);暂存与已准入的
        终结经 abort 出口 marshal 到循环线程走 finish_requests(原生分发)。"""
        if self._lwd_gate_pending.pop(request_id, None) is not None:
            logger.info("[Lwd] cloud aborted pending request %s", request_id)
        self._lwd_gate_ready.discard(request_id)
        if self._lwd_abort_sink is not None:
            self._lwd_abort_sink(request_id)
        else:
            self.finish_requests([request_id], RequestStatus.FINISHED_ABORTED)

    def lwd_cloud_bind_bridge(
        self, admit_sink: Callable, abort_sink: Callable, stop: Callable
    ) -> None:
        """绑定桥线程侧注入:提升出口/abort 出口/停桥句柄(§10.12)。"""
        self._lwd_admit_sink = admit_sink
        self._lwd_abort_sink = abort_sink
        self._lwd_bridge_stop = stop
        logger.info("[Lwd] cloud bridge bound to scheduler")

    def lwd_cloud_control_plane_bound(self) -> bool:
        """幂等判据:控制面是否已绑定到本调度器(装配期防重复装配)。"""
        return self._lwd_request_factory is not None

    def shutdown(self) -> None:
        """关停转发:云形态下桥线程经原生 scheduler.shutdown 钩子收到停机。"""
        if self._lwd_bridge_stop is not None:
            self._lwd_bridge_stop()
            self._lwd_bridge_stop = None
        super().shutdown()

    def _lwd_promote(self, request_id: str) -> None:
        """过门:工厂建请求经提升出口交暂存/准入(缺省直进 add_request)。"""
        if self._lwd_request_factory is None:
            logger.warning("[Lwd] request factory unbound, %s stays gated", request_id)
            return
        wire = self._lwd_gate_pending.pop(request_id)
        if self._lwd_admit_sink is not None:
            self._lwd_admit_sink(self._lwd_request_factory(wire))
        else:
            self.add_request(self._lwd_request_factory(wire))
        logger.info("[Lwd] cloud gated request %s admitted", request_id)

    def lwd_cloud_stats(self) -> dict[str, int]:
        """只读观测:首预告门与暂存池规模。"""
        return {
            "lwd_gate_pending": len(self._lwd_gate_pending),
            "lwd_gate_ready": len(self._lwd_gate_ready),
            "lwd_staged": len(self._staged),
        }

    def lwd_cloud_publish_consumed_watermarks(self) -> None:
        """消费水位推导(§9.1 裁:传输点未接,upto 推导保留待回接)。"""
        chunk_size = self.scheduler_config.max_num_batched_tokens
        sent = self._lwd_consumed_sent
        for rid, request in self.requests.items():
            n_prompt = request.num_prompt_tokens
            if request.num_computed_tokens >= n_prompt:
                upto = (n_prompt + chunk_size - 1) // chunk_size - 1
            else:
                upto = request.num_computed_tokens // chunk_size - 1
            if upto < 0:
                continue
            last = sent.get(rid, -1)
            if upto > last:
                sent[rid] = upto
                logger.debug(
                    "[Lwd] watermark rid=%s upto=%s (transport cut, §9.1)",
                    rid,
                    upto,
                )

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
