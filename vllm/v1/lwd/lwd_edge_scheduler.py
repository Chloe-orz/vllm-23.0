"""边侧调度器:纯 prefill 调度 + 控制面发布(notify/abort/seqno,§9.10/§9.12)。

原生 AsyncScheduler 的两个前提在边侧不成立:prompt 算完会转 decode、
请求只能由模型输出终结;边侧无本地解码,故需专用调度器。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm.v1.lwd.lwd_edge_channel import LwdEdgeControlPublisher


class LwdEdgeScheduler(AsyncScheduler):
    """纯 prefill 语义 + notify/abort/seqno(唯一控制面出口)。

    纯 prefill 调度原语与云侧 LwdCloudPhaseScheduler 同源
    (_lwd_schedule_pure_prefill),S2 实现时评估下沉公共基类避免两处复制。
    """

    def __init__(self, *args, publisher: LwdEdgeControlPublisher | None = None,
                 **kwargs) -> None:
        """publisher 经装配期注入(scheduler_cls 以 partial 携带通道)。"""
        ...

    def schedule(self) -> SchedulerOutput:
        """只调度 prefill 范围(分块复用原生);已完 prefill 的请求不进 decode。"""
        ...

    def lwd_edge_notify(self, scheduler_output: SchedulerOutput) -> bool:
        """对新调度的 prefill 发 LwdEmbedNotify(seqno 先行)。

        publish 队满返回 False,调用方本步视为未派发、下一步重试
        (原生 SO 由调度器自然复现,无需回滚)。
        """
        ...

    def lwd_edge_abort(self, request_ids: "list[str]") -> None:
        """发 LwdAbortSignal。"""
        ...

    def lwd_edge_update_progress(self, executed: "dict[str, int]") -> None:
        """步末按已执行量推进 num_computed;prompt 全部完成即本地终结请求。

        终结不依赖任何模型输出,这是与原生 update_from_output 的唯一语义差;
        executed 的实际来源由数据面落位时对接(§9.12)。
        """
        ...

    def _lwd_edge_next_seqno(self) -> int:
        """seqno 单调分配;控制面登记与数据面 tag 都由它派生。"""
        ...
