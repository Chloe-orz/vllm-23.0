"""Lwd 控制面通知(notify):仅依赖 msgspec,不携带 tensor(§2.7/§9,控制面专用,§9.12)。

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
    block_hashes:边侧本地算好的 prompt 全量满块链(自位置 0 起,
    len == num_prompt_tokens // hash_block_size)—— 云 prompt token 是
    占位零值,本地哈希算不出真实链,链随预告下发让云侧前缀缓存按真实
    内容命中;缺省空 = 边侧未提供,云侧回退本地(占位链,不影响正确性)。
    """

    request_id: str
    num_prompt_tokens: int
    max_tokens: int = 16
    block_hashes: list[bytes] = []


class LwdAbortNotify(msgspec.Struct, gc=False, tag=True):
    """边->云 abort 预告(PRE_OUT);云侧清理请求登记。"""

    request_id: str


class LwdHelloNotify(msgspec.Struct, gc=False, tag=True):
    """云->边发现通告(POST_OUT 面,首拍一次,无周期重发)。

    云侧启动即通告一次:边侧装配期阻塞等待的唯一发现窗口(裁定:
    不考虑云换址/边重启自愈,任一侧重启即整组重拉)。
    pre_out_* 是边侧连接云端点
    的唯一事实源(边侧不读配置里的 host——决策 B:单一事实源)。
    pre_out_host 必须是边可路由的真实 IP(或同机 127.0.0.1),
    0.0.0.0 不可作为通告值(serve 守卫拦截)。
    """

    pre_out_host: str
    pre_out_port: int


class LwdC2eNotify(msgspec.Struct, gc=False, tag=True):
    """云->边步元数据通告(POST_OUT,云->边唯一载荷类型):先于隐藏张量
    到达,边侧据 hidden_num_elements 预挂精确尺寸 recv;req_ids/
    top_id_ths 按隐藏行序(ModelRunnerOutput.lwd_c2e_meta 的线上形态)。

    finished 与 req_ids 对齐(additive,缺省空 = 兼容"全部完结"旧语义):
    云侧逐 decode 步回传 token 时逐条 False,终结步置 True;纯终结
    通告 = hidden_num_elements 为 0、仅列完结请求——边侧本地终结,
    不派发 unembed 批。"""

    hidden_num_elements: int
    top_id_ths: list[list[int]]
    num_accepted_tokens: list[int]
    req_ids: list[str]
    finished: list[bool] = []


# typing.Union 而非 PEP 604 `|`:msgspec 解码器的全版本支持路径
LwdNotify = Union[LwdRangeNotify, LwdRequestNotify, LwdAbortNotify]  # noqa: UP007
# 云->边方向(POST_OUT):HELLO 发现 + 步元数据(唯一载荷,兼结果回传
# 驱动);重同步消息在此 union 上 additive 扩展
LwdCloudNotify = Union[LwdHelloNotify, LwdC2eNotify]  # noqa: UP007
# 数据面批型定义归 vllm/v1/core/sched/output.py(LwdBatch/LwdBatchType/
# LwdEmbedBatch/LwdUnembedBatch,f8182fd5 定稿),协议层不重复声明

_NOTIFY_DECODER = msgspec.msgpack.Decoder(LwdNotify)
_CLOUD_NOTIFY_DECODER = msgspec.msgpack.Decoder(LwdCloudNotify)


def lwd_encode_notify(message: LwdNotify) -> bytes:
    """通知编码为传输字节(发布端唯一入口)。"""
    return msgspec.msgpack.encode(message)


def lwd_decode_notify(data: bytes) -> LwdNotify:
    """传输字节解码为通知(订阅端唯一入口);坏包抛异常,由接收线程捕获丢弃。"""
    return _NOTIFY_DECODER.decode(data)


def lwd_encode_cloud_notify(message: LwdCloudNotify) -> bytes:
    """云->边通知编码(POST_OUT 发布端唯一入口)。"""
    return msgspec.msgpack.encode(message)


def lwd_decode_cloud_notify(data: bytes) -> LwdCloudNotify:
    """云->边通知解码(POST_OUT 订阅端唯一入口);坏包抛异常由接收方丢弃。"""
    return _CLOUD_NOTIFY_DECODER.decode(data)
