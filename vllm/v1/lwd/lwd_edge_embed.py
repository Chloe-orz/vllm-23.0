"""边侧数据面:嵌入前向 + torch.distributed tag 直传(内核仅此文件使用 dist,§8.3/§9.9)。

worker_base 按角色守卫直接调用模块函数(§9.11):无 handler 类,
原生 SchedulerOutput 进,嵌入回执出。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm.v1.lwd.lwd_message import LwdEdgeEmbedAck


def lwd_edge_execute_embeds(
    model, scheduler_output: SchedulerOutput
) -> "list[LwdEdgeEmbedAck]":
    """worker 入口:逐请求处理原生 SO 的调度 token 范围,嵌入并 isend。

    seqno 对齐:notify 先于数据发布,按 (request, 已发次数)
    本地推演出与 dispatcher/云侧一致的 seqno(§9.9)。
    """
    ...


def _lwd_edge_embed_range(model, token_ids: "list[int]") -> torch.Tensor:
    """单个 token 范围的嵌入前向(get_input_embeddings)。"""
    ...


def _lwd_edge_send_embeds(hidden: torch.Tensor, seqno: int) -> None:
    """dist.isend(dst=cloud_first_rank, tag=LWD_WIRE_TAG_BASE + seqno);TP=1 由配置校验。"""
    ...
