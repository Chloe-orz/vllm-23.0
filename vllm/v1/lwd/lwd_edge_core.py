"""边侧执行类:唯一接口 step_with_batch_queue,由 core.py 守卫委托(§9.8/§9.12)。

step 语义 = 原生步进的控制面替代:调度器出分块决策并发预告 ->
原生路径提交执行 -> 步末推进进度。数据面动作不在本层(§9.12)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.lwd.lwd_step_core import LwdStepCore

if TYPE_CHECKING:
    from vllm.v1.lwd.lwd_config import LwdConfig
    from vllm.v1.lwd.lwd_message import EngineCoreOutputs
    from vllm.v1.lwd.lwd_ports import LwdEnginePort


class LwdEdgeCore(LwdStepCore):
    """边侧 prefill_only core:只发不收(§9.1),输出恒为 (None, 是否有工作)。"""

    def __init__(self, engine_port: LwdEnginePort, config: LwdConfig) -> None:
        ...

    def step_with_batch_queue(
        self,
    ) -> "tuple[dict[int, EngineCoreOutputs] | None, bool]":
        """编排:调度器 schedule() -> 发预告 -> 提交执行 -> 步末推进进度。"""
        ...

    def _lwd_edge_step_schedule(self) -> "object | None":
        """取分块决策;调度器为装配期注入的 LwdEdgeScheduler(纯 prefill,§9.10)。"""
        ...

    def _lwd_edge_step_dispatch(self) -> None:
        """调度器 lwd_edge_notify 发预告 + 经 engine_port.lwd_execute_model 提交。"""
        ...

    def _lwd_edge_step_update_progress(self) -> None:
        """步末推进调度器进度并终结已完请求(执行量来源由数据面对接,§9.12)。"""
        ...
