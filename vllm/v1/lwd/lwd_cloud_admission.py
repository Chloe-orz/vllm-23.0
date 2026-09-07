"""云侧准入策略族(策略模式):决定哪些 waiting 请求可进入调度(源 SeparatePhases/Immediate)。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LwdCloudSchedulingState:
    """准入决策输入快照(由 LwdCloudSchedulerView 派生,不触调度器内部)。"""

    waiting_request_ids: "list[str]" = field(default_factory=list)
    running_request_ids: "list[str]" = field(default_factory=list)
    # (request_id, num_computed_tokens, num_prompt_tokens)
    request_progress: "list[tuple[str, int, int]]" = field(default_factory=list)
    decode_phase_active: bool = False


class LwdCloudAdmissionPolicy(ABC):
    """准入策略抽象:输入快照,输出本轮可准入的 request_id 列表。"""

    @abstractmethod
    def lwd_plan_admission(self, state: LwdCloudSchedulingState) -> "list[str]":
        ...


class LwdCloudSeparatePhasesPolicy(LwdCloudAdmissionPolicy):
    """相位分离:prefill 相位不掺新请求,避免相位抖动。"""


class LwdCloudImmediatePolicy(LwdCloudAdmissionPolicy):
    """立即准入:waiting 即进,最大化吞吐。"""


def lwd_cloud_admission_policy(name: str | None = None) -> LwdCloudAdmissionPolicy:
    """工厂:按配置名取准入策略;未知名回退默认并告警(不抛)。"""
    ...
