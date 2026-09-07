"""边侧执行类:唯一接口 step_with_batch_queue,由 core.py 守卫委托(§9.8/§9.9)。

step 语义 = 原生步进的嵌入替代:原生 schedule() 出分块决策 ->
通知云侧 -> 经执行器送 worker 嵌入 -> 按回执更新调度器进度。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.lwd.lwd_step_core import LwdStepCore

if TYPE_CHECKING:
    from vllm.v1.lwd.lwd_config import LwdConfig
    from vllm.v1.lwd.lwd_edge_dispatcher import LwdEdgeDispatcher
    from vllm.v1.lwd.lwd_message import EngineCoreOutputs
    from vllm.v1.lwd.lwd_ports import LwdEnginePort


class LwdEdgeCore(LwdStepCore):
    """边侧 prefill_only core:只发不收(§9.1),输出恒为 (None, 是否有工作)。"""

    def __init__(
        self,
        dispatcher: LwdEdgeDispatcher,
        engine_port: LwdEnginePort,
        config: LwdConfig,
    ) -> None:
        ...

    def step_with_batch_queue(
        self,
    ) -> "tuple[dict[int, EngineCoreOutputs] | None, bool]":
        """编排:原生 schedule() -> 发预告 -> 提交嵌入 -> 更新进度。"""
        ...

    def _lwd_edge_step_schedule(self) -> "object | None":
        """取分块决策;调度器为装配期注入的 LwdEdgeScheduler(纯 prefill,§9.10)。"""
        ...

    def _lwd_edge_step_dispatch(self) -> None:
        """发 LwdEmbedNotify + 经 engine_port.lwd_execute_model 提交原生 SO(队满则本轮放弃)。"""
        ...

    def _lwd_edge_step_poll_acks(self) -> None:
        """engine_port.lwd_drain_embed_acks 取回执,经 lwd_edge_update_progress 推进并终结。"""
        ...
