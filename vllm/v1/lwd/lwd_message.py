"""Lwd 线上消息:仅依赖 msgspec,不携带 tensor(§2.7/§9.9)。

分块决策复用上游 Scheduler.schedule() 的原生 chunked prefill;
本模块只承载通知与回执,seqno 由边侧分配、随 notify 先于数据发出。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import msgspec

if TYPE_CHECKING:
    # 上游输出类型经本模块 re-export,其余内核文件不得直接 import vllm.v1.engine
    # (白名单,§7.2)
    from vllm.v1.engine import EngineCoreOutputs  # noqa: F401

# torch.distributed 直传 tag 基数:tag = LWD_WIRE_TAG_BASE + seqno(§8.3,
# tag 匹配天然乱序安全;seqno 由 dispatcher 分配,worker/云侧按 notify 对齐)
LWD_WIRE_TAG_BASE = 10_000


class LwdEmbedNotify(msgspec.Struct, gc=False):
    """边->云嵌入预告(PRE_OUT);发布先于数据面 isend。

    offset/num_tokens 直接取自原生 SchedulerOutput 的调度决策
    (chunked prefill 的 num_computed/num_scheduled);云侧据此预登记
    recv 范围并推导张量形状。
    """

    request_id: str
    offset: int
    num_tokens: int
    seqno: int


class LwdAbortSignal(msgspec.Struct, gc=False):
    """边->云 abort(PRE_OUT);云侧丢弃已收/在途 embeds 并清理请求。"""

    request_id: str


@dataclass
class LwdEdgeEmbedAck:
    """worker->dispatcher 完成回执(边进程内,调度器进度更新的依据)。"""

    request_id: str
    num_tokens: int
