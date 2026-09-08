"""云侧准入策略族(策略模式):决定哪些 waiting 请求可进入调度。

策略源:SeparatePhases(相位分离)/ Immediate(立即准入)。

waiting 指 LwdCloudCore 的 pending(尚未注入调度器);准入即把请求
交给调度器,此后进度由原生调度器自持(§9.9)。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class LwdCloudSchedulingState:
    """准入决策输入快照(由 LwdCloudSchedulerView 派生,不触调度器内部)。"""

    waiting_request_ids: list[str] = field(default_factory=list)
    running_request_ids: list[str] = field(default_factory=list)
    # (request_id, num_computed_tokens, num_prompt_tokens)
    request_progress: list[tuple[str, int, int]] = field(default_factory=list)
    decode_phase_active: bool = False


class LwdCloudAdmissionPolicy(ABC):
    """准入策略抽象:输入快照,输出本轮可准入的 request_id 列表。"""

    @abstractmethod
    def lwd_plan_admission(self, state: LwdCloudSchedulingState) -> list[str]: ...


class LwdCloudSeparatePhasesPolicy(LwdCloudAdmissionPolicy):
    """相位分离:decode 相位不掺新请求,避免相位抖动。"""

    def lwd_plan_admission(self, state: LwdCloudSchedulingState) -> list[str]:
        if state.decode_phase_active:
            return []
        return list(state.waiting_request_ids)


class LwdCloudImmediatePolicy(LwdCloudAdmissionPolicy):
    """立即准入:waiting 即进,最大化吞吐。"""

    def lwd_plan_admission(self, state: LwdCloudSchedulingState) -> list[str]:
        return list(state.waiting_request_ids)


_LWD_ADMISSION_POLICIES = {
    "separate_phases": LwdCloudSeparatePhasesPolicy,
    "immediate": LwdCloudImmediatePolicy,
}


def lwd_cloud_admission_policy(name: str | None = None) -> LwdCloudAdmissionPolicy:
    """工厂:按配置名取准入策略;未知名回退默认并告警(不抛)。"""
    policy_cls = _LWD_ADMISSION_POLICIES.get(name or "separate_phases")
    if policy_cls is None:
        logger.warning(
            "[Lwd] unknown admission policy %r, fallback to separate_phases", name
        )
        policy_cls = LwdCloudSeparatePhasesPolicy
    return policy_cls()
