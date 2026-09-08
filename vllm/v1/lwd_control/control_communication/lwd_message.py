"""Lwd 线上消息:仅依赖 msgspec,不携带 tensor(§2.7/§9,控制面专用,§9.12)。

数据面不在本目录范围:notify 只承载调度决策预告,张量传输与落位
由数据面经既有接缝对接。上游输出类型经本模块 re-export,
其余内核文件不得直接 import vllm.v1.engine(白名单,§7.2)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import msgspec

if TYPE_CHECKING:
    # 上游输出类型 re-export:step 载体的返回注解共用,避免各文件直连上游
    from vllm.v1.engine import EngineCoreOutputs  # noqa: F401


class LwdRangeNotify(msgspec.Struct, gc=False, tag=True):
    """边->云调度范围预告(PRE_OUT);发布先于任何数据面动作。

    offset/num_tokens 直接取自原生 SchedulerOutput 的调度决策
    (chunked prefill 的 num_computed/num_scheduled);云侧登记
    seqno->request 映射,数据面落位后由此对接接收。重复预告按
    (request_id, offset) 幂等登记(边侧队满重试天然产生重复)。
    """

    request_id: str
    offset: int
    num_tokens: int
    seqno: int


class LwdRequestNotify(msgspec.Struct, gc=False, tag=True):
    """边->云请求预告(PRE_OUT);云侧据此建调度请求并驱动本地生命周期。

    结果不回边(§9.1),线上只带调度决策所需字段;max_tokens 供云侧
    调度器判定终结,采样参数等扩展由数据面/后续迭代按需增补(additive)。
    """

    request_id: str
    num_prompt_tokens: int
    max_tokens: int = 16


class LwdAbortNotify(msgspec.Struct, gc=False, tag=True):
    """边->云 abort 预告(PRE_OUT);云侧清理请求登记。"""

    request_id: str


# typing.Union 而非 PEP 604 `|`:msgspec 解码器的全版本支持路径
LwdWireMessage = Union[LwdRangeNotify, LwdRequestNotify, LwdAbortNotify]  # noqa: UP007

_WIRE_DECODER = msgspec.msgpack.Decoder(LwdWireMessage)


def lwd_encode_wire(message: LwdWireMessage) -> bytes:
    """线上编码(发布端唯一入口)。"""
    return msgspec.msgpack.encode(message)


def lwd_decode_wire(data: bytes) -> LwdWireMessage:
    """线上解码(订阅端唯一入口);坏帧抛异常,由订阅线程捕获丢弃。"""
    return _WIRE_DECODER.decode(data)
