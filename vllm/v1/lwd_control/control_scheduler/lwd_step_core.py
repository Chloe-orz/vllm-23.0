"""Lwd 执行类抽象:边云对 EngineCore 的唯一扩展点就是 step 接口(§9.8)。

core.py 的 step_with_batch_queue 顶部守卫:
    if self.step_wrapper is not None:
        return self.step_wrapper.step_with_batch_queue()
装配期(L3)把 prefill_only 专用 core 赋给 EngineCore.step_wrapper;
非 PO 路径 step_wrapper 恒为 None,上游行为不变。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.lwd.lwd_message import EngineCoreOutputs


class LwdStepCore(ABC):
    """prefill_only 专用 core 的执行类抽象:唯一接口 step_with_batch_queue。

    与 EngineCore.step_with_batch_queue(core.py:484)同签名同返回;
    边/云实现类除本接口与构造注入外,不暴露其它公共接口(§9.8)。
    """

    @abstractmethod
    def step_with_batch_queue(
        self,
    ) -> "tuple[dict[int, EngineCoreOutputs] | None, bool]":
        """单步推进;返回 (输出表 | None, 是否有工作)。"""
        ...
