"""边侧 embed 派发器:分块决策复用原生 schedule(),本类只负责通知与 abort(§9.9)。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm.v1.lwd.lwd_config import LwdConfig
    from vllm.v1.lwd.lwd_message import LwdEmbedNotify


class LwdEdgeDispatcher:
    """消费原生 SchedulerOutput:为新调度的 prefill 范围发预告,不维护自建状态机。

    请求进度(num_computed 等)由原生调度器自持,本类仅额外维护
    seqno 分配与在途请求登记(abort 判定用)。
    """

    def __init__(self, publisher, config: LwdConfig) -> None:
        ...

    def lwd_edge_dispatch(self, scheduler_output: SchedulerOutput) -> bool:
        """对原生 SO 中每个新调度的 prefill 发 LwdEmbedNotify(seqno 先行)。

        publish 队满返回 False,调用方本步视为未派发、下一步重试
        (原生 SO 由调度器自然复现,无需回滚)。
        """
        ...

    def abort_requests(self, request_ids: "list[str]") -> None:
        """发 LwdAbortSignal;在途数据不承诺取消(云侧 tag 无人认领作废)。"""
        ...

    def on_embed_acked(self, request_id: str, num_tokens: int) -> None:
        """worker 完成回执推进在途登记(全部落定后可清请求)。"""
        ...

    def has_requests(self) -> bool:
        ...

    def _lwd_edge_build_notify(self, request_id: str, offset: int,
                               num_tokens: int) -> LwdEmbedNotify:
        ...

    def _lwd_edge_next_seqno(self) -> int:
        """seqno 单调分配;tag 与云侧登记都由它派生。"""
        ...

    def shutdown(self) -> None:
        ...
