"""Lwd 执行类抽象与端口协议:边云对 EngineCore 的唯一扩展点是 step 接口(§9.8)。

core.py 的 step_with_batch_queue 顶部守卫:
    if self.step_wrapper is not None:
        return self.step_wrapper.step_with_batch_queue()
装配期(L3)把 prefill_only 专用 core 赋给 EngineCore.step_wrapper;
非 PO 路径 step_wrapper 恒为 None,上游行为不变。

本文件同时承载内核共享的纯支撑(§10.3 折入):LwdEnginePort /
LwdCloudSchedulerView 端口协议、LwdStepSettings plain 值载体、LwdLog。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from vllm.logger import init_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vllm.v1.lwd_control.control_communication.lwd_control_communicator import (
        EngineCoreOutputs,
    )

logger = init_logger(__name__)


@dataclass(frozen=True)
class LwdStepSettings:
    """步进编排 plain 值;装配期由 LwdConfig 派生,内核不解析配置(§10.3)。"""

    debug: bool = False
    zombie_log_interval_s: float = 30.0


class LwdLog:
    """按 debug 开关分级的极简诊断面;error 留给真正的业务中断点。"""

    def __init__(self, debug: bool = False) -> None:
        self._debug = debug

    def phase(self, message: str, *args) -> None:
        """步进/相位/准入轨迹;默认关,生产路径零输出。"""
        if self._debug:
            logger.info("[Lwd] %s", message % args if args else message)

    def degrade(self, message: str, *args) -> None:
        """可恢复降级(warning):丢预告、降级原生等。"""
        logger.warning("[Lwd] %s", message % args if args else message)


class LwdStepCore(ABC):
    """prefill_only 专用 core 的执行类抽象:唯一接口 step_with_batch_queue。

    与 EngineCore.step_with_batch_queue(core.py:484)同签名同返回;
    边/云实现类除本接口与构造注入外,不暴露其它公共接口(§9.8)。
    """

    @abstractmethod
    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """单步推进;返回 (输出表 | None, 是否有工作)。"""
        ...


class LwdEnginePort(Protocol):
    """边/云 step 编排对 EngineCore 的最小触达面(§7.3-C1/§9.8,S2 定稿)。

    step_wrapper 模式下专用 core 不继承 EngineCore,scheduler/executor
    一律经本端口方法触达;边/云适配器各落在两侧装配文件(§10.3),
    是台账登记的 engine_core 属性容忍点(§7.3-C5)。
    """

    def lwd_scheduler(self):
        """装配期注入的 Lwd 调度器(边:LwdEdgeScheduler / 云:相位调度器)。"""
        ...

    def lwd_execute_model(self, scheduler_output):
        """边侧原生执行提交(同步返回 ModelRunnerOutput,输出内容不消费)。"""
        ...

    def lwd_engine_core(self):
        """云侧增强步体的 EngineCore 触达(lwd_cloud_core 步外接口用)。"""
        ...

    def lwd_abort_requests(self, request_ids: list[str]) -> None:
        """云侧 abort:走 EngineCore.abort_requests 公共路径(含输出簿记)。"""
        ...

    def lwd_scheduler_view(self) -> LwdCloudSchedulerView:
        """调度器只读快照(准入策略输入,云侧使用)。"""
        ...


class LwdCloudSchedulerView(Protocol):
    """调度器只读快照(3 方法,§7.3-C1,准入策略输入)。"""

    def lwd_unfinished_count(self) -> int: ...

    def lwd_waiting_count(self) -> int: ...

    def lwd_request_progress(self) -> Iterable[tuple[str, int, int]]: ...
