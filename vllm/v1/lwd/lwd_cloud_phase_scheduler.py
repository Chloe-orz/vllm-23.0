"""云侧相位调度器:纯 prefill/纯 decode 交替(内核唯一上游继承点之一,§7.5-1)。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class LwdCloudPhaseScheduler(AsyncScheduler):
    """模板方法:schedule 编排固定,子类只翻转 _lwd_prefer_prefill 原语钩子。"""

    def __init__(self, *args, **kwargs) -> None:
        ...

    def _lwd_schedule_pure_prefill(self) -> SchedulerOutput:
        ...

    def _lwd_schedule_pure_decode(self) -> SchedulerOutput:
        ...

    @staticmethod
    def _lwd_is_empty(out: SchedulerOutput) -> bool:
        ...

    def _lwd_prefer_prefill(self) -> bool:
        """相位偏好原语钩子(基类默认 prefill 优先)。"""
        ...

    def schedule(self) -> SchedulerOutput:
        """上游签名(不改名);两相皆空时返回空输出保持步进节奏。"""
        ...


class LwdCloudPrefillFirstScheduler(LwdCloudPhaseScheduler):
    """prefill 相位优先(默认策略)。"""


class LwdCloudDecodeFirstScheduler(LwdCloudPhaseScheduler):
    """decode 相位优先(延迟敏感策略)。"""


def lwd_cloud_scheduler_cls(name: str | None = None) -> "type[LwdCloudPhaseScheduler]":
    """工厂:按配置名取相位策略类;未知名回退默认并告警(不抛)。"""
    ...
