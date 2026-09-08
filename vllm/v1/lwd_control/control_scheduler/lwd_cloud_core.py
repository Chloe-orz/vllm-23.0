"""云侧执行类:唯一接口 step_with_batch_queue,由 core.py 守卫委托(§9.8/§9.12)。

不再继承 EngineCore:真实 EngineCore 实例照常装配(worker/scheduler/batch_queue),
本类只替换其 step 语义;对 EngineCore 的触达一律经 LwdEnginePort(§7.3-C1)。
数据面(接收/落位/消费释放)不在本层,落位后经登记回调对接(§9.12)。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.lwd_control.control_communication.lwd_message import (
    LwdAbortNotify,
    LwdRangeNotify,
    LwdRequestNotify,
)
from vllm.v1.lwd_control.control_scheduler.lwd_step_core import (
    LwdLog,
    LwdStepCore,
    LwdStepSettings,
)

if TYPE_CHECKING:
    from vllm.v1.lwd_control.control_communication.lwd_message import (
        EngineCoreOutputs,
    )
    from vllm.v1.lwd_control.control_communication.lwd_control_subscriber import (
        LwdControlSubscriber,
    )
    from vllm.v1.lwd_control.control_scheduler.lwd_cloud_admission import (
        LwdCloudAdmissionPolicy,
        LwdCloudSchedulingState,
    )
    from vllm.v1.lwd_control.control_scheduler.lwd_step_core import LwdEnginePort

logger = init_logger(__name__)


class LwdCloudCore(LwdStepCore):
    """云侧 prefill_only core:控制面 drain -> 相位调度执行 -> 步后记账。

    调度器是装配期注入的相位调度器实例(scheduler_cls 经
    lwd_cloud_scheduler_cls() 选取):分块决策走原生 AsyncScheduler 语义
    (§9.9),Lwd 增量只有相位偏好。请求只经 PRE_OUT 到达(add/abort 全在
    drain 内处理),公共接口仅 step_with_batch_queue(§9.8)。
    """

    def __init__(
        self,
        subscriber: LwdControlSubscriber,
        admission_policy: LwdCloudAdmissionPolicy,
        engine_port: LwdEnginePort,
        admit_request: Callable[[LwdRequestNotify], None],
        settings: LwdStepSettings,
    ) -> None:
        self._lwd_subscriber = subscriber
        self._admission_policy = admission_policy
        self._engine_port = engine_port
        self._admit_request = admit_request
        self._log = LwdLog(settings.debug)
        self._zombie_interval_s = settings.zombie_log_interval_s
        self._pending: dict[str, LwdRequestNotify] = {}
        # 已收首个 range notify 的请求(首预告门,§10.5-3)
        self._notified: set[str] = set()
        # seqno -> (request_id, offset, num_tokens, 登记时刻)
        self._embed_registry: dict[int, tuple[str, int, int, float]] = {}
        self._request_seqnos: dict[str, set[int]] = {}

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """编排:drain 控制面 -> 相位调度 + 执行(经 engine_port)-> 步后记账。"""
        self._lwd_drain_control_plane()
        self._lwd_apply_scheduling_policy()
        outputs, model_executed = self._engine_port.lwd_step_with_batch_queue()
        self._lwd_collect_finished(outputs)
        return outputs, model_executed

    # ---- 控制面处理(单向:仅订阅 drain,无快路径) ----

    def _lwd_drain_control_plane(self) -> None:
        """按到达序分发 notify/add_request/abort(§9.3 通信物自带 seqno)。"""
        for message in self._lwd_subscriber.drain():
            if isinstance(message, LwdRangeNotify):
                self._lwd_handle_range_notify(message)
            elif isinstance(message, LwdRequestNotify):
                self._lwd_handle_request_notify(message)
            elif isinstance(message, LwdAbortNotify):
                self._lwd_handle_abort_notify(message.request_id)

    def _lwd_handle_range_notify(self, notify: LwdRangeNotify) -> None:
        """登记 seqno->request;数据面落位后在此转投接收侧(§9.12)。"""
        self._embed_registry[notify.seqno] = (
            notify.request_id,
            notify.offset,
            notify.num_tokens,
            time.monotonic(),
        )
        self._request_seqnos.setdefault(notify.request_id, set()).add(notify.seqno)
        self._notified.add(notify.request_id)

    def _lwd_handle_request_notify(self, message: LwdRequestNotify) -> None:
        """暂存待准入请求;重复预告幂等(边侧重试会产生重复)。"""
        if message.request_id not in self._pending:
            self._pending[message.request_id] = message

    def _lwd_handle_abort_notify(self, request_id: str) -> None:
        """abort:清请求登记(数据面清理由其落位侧对接)。"""
        self._engine_port.lwd_abort_requests([request_id])
        self._lwd_release_request(request_id)

    # ---- 步进内部 ----

    def _lwd_apply_scheduling_policy(self) -> None:
        """准入策略(经 scheduler view 快照)决定本步可准入的 waiting 集合。

        首预告门(§10.5-3):策略放行后仍需已收首个 notify 才准入,
        防止 fill 在未预告请求上空等(无超时门,§8.3-2)。
        """
        state = self._lwd_scheduling_state()
        for request_id in self._admission_policy.lwd_plan_admission(state):
            if request_id not in self._notified:
                continue
            message = self._pending.pop(request_id, None)
            if message is not None:
                self._admit_request(message)
                self._log.phase("cloud admit: %s", request_id)

    def _lwd_scheduling_state(self) -> LwdCloudSchedulingState:
        """由 view 快照派生准入输入;decode 活跃 = 存在 prefill 已完的请求。"""
        view = self._engine_port.lwd_scheduler_view()
        progress = list(view.lwd_request_progress())
        return LwdCloudSchedulingState(
            waiting_request_ids=list(self._pending),
            running_request_ids=[request_id for request_id, _, _ in progress],
            request_progress=progress,
            decode_phase_active=any(
                computed >= prompt for _, computed, prompt in progress
            ),
        )

    def _lwd_collect_finished(self, outputs) -> None:
        """finished 请求本地记账(结果不外发,§9.1)+ registry 僵尸观测。"""
        for engine_outputs in (outputs or {}).values():
            for output in engine_outputs.outputs:
                if output.finished:
                    self._lwd_release_request(output.request_id)
        self._lwd_warn_zombie_registry()

    def _lwd_warn_zombie_registry(self) -> None:
        """登记长期无人认领的预告(无 30s 等待门下的可观测兜底,§8.3-2)。"""
        now = time.monotonic()
        live_ids = self._lwd_live_request_ids()
        for seqno, (request_id, _, _, registered_at) in self._embed_registry.items():
            is_stale = now - registered_at > self._zombie_interval_s
            if request_id not in live_ids and is_stale:
                logger.warning(
                    "[Lwd] zombie range notify: seqno=%s request=%s", seqno, request_id
                )

    def _lwd_live_request_ids(self) -> set[str]:
        """仍存活 = 已入调度器未完 or 待准入。"""
        view = self._engine_port.lwd_scheduler_view()
        live = {request_id for request_id, _, _ in view.lwd_request_progress()}
        return live | set(self._pending)

    def _lwd_release_request(self, request_id: str) -> None:
        """请求登记表清理唯一实现(W11/§10.5-4):三路径共调。

        abort / finished / finish_reason 统一走本方法。"""
        self._pending.pop(request_id, None)
        self._notified.discard(request_id)
        for seqno in self._request_seqnos.pop(request_id, set()):
            self._embed_registry.pop(seqno, None)

    def lwd_stats(self) -> dict[str, int]:
        """只读观测(§9.1):控制面登记规模,诊断/外部统计用。"""
        return {
            "lwd_pending_requests": len(self._pending),
            "lwd_registered_embeds": len(self._embed_registry),
        }
