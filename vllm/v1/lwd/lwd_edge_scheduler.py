"""边侧调度器:纯 prefill 调度,prompt 调度完成即本地终结(不进 decode,§9.10)。

原生 AsyncScheduler 的两个前提在边侧不成立:prompt 算完会转 decode、
请求只能由模型输出终结;边侧无模型执行且只发不收,故需专用调度器。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm.v1.lwd.lwd_message import LwdEdgeEmbedAck


class LwdEdgeScheduler(AsyncScheduler):
    """纯 prefill 语义:chunked 决策复用原生,prefill 完成即 finished。

    纯 prefill 调度原语与云侧 LwdCloudPhaseScheduler 同源
    (_lwd_schedule_pure_prefill),S2 实现时评估下沉公共基类避免两处复制。
    """

    def schedule(self) -> SchedulerOutput:
        """只调度 prefill 范围(分块复用原生);已完 prefill 的请求不进 decode。"""
        ...

    def lwd_edge_update_progress(self, acks: "list[LwdEdgeEmbedAck]") -> None:
        """嵌入回执 -> 推进 num_computed;prompt 全部嵌入完成即本地终结请求。

        终结不依赖任何模型输出,这是与原生 update_from_output 的唯一语义差。
        """
        ...
