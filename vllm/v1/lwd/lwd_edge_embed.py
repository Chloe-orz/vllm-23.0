"""边侧数据面:嵌入前向 + torch.distributed tag 直传(内核仅此文件使用 dist,§8.3/§9.9)。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm.v1.lwd.lwd_config import LwdConfig
    from vllm.v1.lwd.lwd_message import LwdEdgeEmbedAck


class LwdEdgeEmbedHandler:
    """worker 侧嵌入处理器:按原生 SO 的调度范围做嵌入并 isend(tag)。"""

    def __init__(self, model, config: LwdConfig) -> None:
        ...

    def lwd_edge_execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> "list[LwdEdgeEmbedAck]":
        """worker_base 按角色守卫路由的入口;逐请求处理其调度 token 范围。

        seqno 对齐:notify 先于数据发布,worker 按 (request, 已发次数)
        本地推演出与 dispatcher/云侧一致的 seqno(§9.9)。
        """
        ...

    def _lwd_edge_embed_range(self, token_ids: "list[int]") -> torch.Tensor:
        """单个 token 范围的嵌入前向(get_input_embeddings)。"""
        ...

    def _lwd_edge_send_embeds(self, hidden: torch.Tensor, seqno: int) -> None:
        """dist.isend(dst=cloud_first_rank, tag=LWD_WIRE_TAG_BASE + seqno);TP=1 由配置校验。"""
        ...

    def shutdown(self) -> None:
        ...
