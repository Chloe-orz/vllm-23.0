"""Lwd 线上消息:仅依赖 msgspec,不携带 tensor(§2.7/§9,控制面专用,§9.12)。

数据面不在本目录范围(§9.12):notify 只承载调度决策预告,
实际张量传输与落位由数据面另行对接。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

if TYPE_CHECKING:
    # 上游输出类型经本模块 re-export,其余内核文件不得直接 import vllm.v1.engine
    # (白名单,§7.2)
    from vllm.v1.engine import EngineCoreOutputs  # noqa: F401


class LwdEmbedNotify(msgspec.Struct, gc=False):
    """边->云嵌入预告(PRE_OUT);发布先于任何数据面动作。

    offset/num_tokens 直接取自原生 SchedulerOutput 的调度决策
    (chunked prefill 的 num_computed/num_scheduled);云侧登记
    seqno->request 映射,数据面落位后由此对接接收。
    """

    request_id: str
    offset: int
    num_tokens: int
    seqno: int


class LwdAbortSignal(msgspec.Struct, gc=False):
    """边->云 abort(PRE_OUT);云侧清理请求登记。"""

    request_id: str
