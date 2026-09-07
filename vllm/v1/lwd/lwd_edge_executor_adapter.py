"""边侧执行器适配器:原生 SchedulerOutput 经本地环 MQ 直达 worker(§9.3/§9.9)。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.lwd.lwd_message import LwdEdgeEmbedAck
    from vllm.v1.lwd.lwd_ports import LwdFuture


class LwdEdgeExecutorAdapter:
    """包住 vllm executor 实现 LwdEdgeExecutorPort。

    分块决策已在原生 SO 内(schedule() 产出),本类零转换:
    原样入 MQ,worker_base 按角色守卫路由到嵌入处理器。
    """

    def __init__(self, executor) -> None:
        ...

    def lwd_submit_embeds(self, scheduler_output) -> LwdFuture:
        """原生 SO 入本地环 MQ 提交边 worker(non_block 语义)。"""
        ...

    def lwd_drain_acks(self) -> "list[LwdEdgeEmbedAck]":
        """从 executor 输出侧取回嵌入完成回执。"""
        ...

    def shutdown(self) -> None:
        ...
