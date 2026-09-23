"""Lwd 控制面通知:仅依赖 msgspec,不携带 tensor,只承载调度决策预告。
上游输出类型经本模块 re-export,其余内核文件不得直接 import vllm.v1.engine。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import msgspec

if TYPE_CHECKING:
    # 上游输出类型 re-export:step 载体的返回注解共用,避免各文件直连上游
    from vllm.v1.engine import EngineCoreOutputs  # noqa: F401

# 线协议版本:2 = register/ack 发现协议(ROUTER/DEALER,边连云);
# 1 = HELLO 首拍发现(PUSH/PULL)+ 计数互校;0 = 无互校字段的旧 HELLO
LWD_WIRE_VERSION = 2

# 完成码哨兵:finish_reasons 中的"本步未终结"值
LWD_NOT_FINISHED = -1


class LwdRangeNotify(msgspec.Struct, gc=False, tag=True):
    """边->云调度范围预告;offset/num_tokens 取自原生调度决策,
    重复预告按 (request_id, offset) 幂等登记(边侧队满重试天然产生重复)。"""

    request_id: str
    offset: int
    num_tokens: int
    seqno: int
    has_mrope: bool = False
    edge_id: int = -1
    """来源边实例 id(fan-in 定向/按边分队列的归属键);-1 = 旧版未携带。"""
    dp_idx: int = -1
    """来源 dp 序号(与 edge_id 联合定位 dp 级连接);-1 = 旧版未携带。"""


class LwdRequestNotify(msgspec.Struct, gc=False, tag=True):
    """边->云请求预告,线上只带调度决策所需字段;block_hashes 为
    边侧算好的 prompt 满块链,供云侧前缀缓存命中,缺省空回退本地哈希。

    采样参数透传(additive,缺省 = SamplingParams 原生缺省):只带影响
    云侧 token 选择的字段(采样核/惩罚/EOS 策略/min_tokens);stop 字符串
    等 detokenizer 层参数留在边侧前端原生处理,不上 wire。"""

    request_id: str
    num_prompt_tokens: int
    max_tokens: int = 16
    block_hashes: list[bytes] = []
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: Optional[int] = None
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    ignore_eos: bool = False
    stop_token_ids: list[int] = []
    min_tokens: int = 0
    eos_token_id: Optional[int] = None
    """结束符 id:边侧前端自 tokenizer 解析,云侧占位 prompt 无从
    得知,必须随预告透传;缺省 None = 云侧 EOS 判定落空,仅
    max_tokens 兜底(旧版边侧,additive 兼容)。"""
    edge_id: int = -1
    dp_idx: int = -1


class LwdAbortNotify(msgspec.Struct, gc=False, tag=True):
    """边->云 abort 预告;云侧清理请求登记。"""

    request_id: str
    edge_id: int = -1
    dp_idx: int = -1


class LwdRegisterNotify(msgspec.Struct, gc=False, tag=True):
    """边->云注册帧(DEALER 连上即发,未收 ack 周期重发,幂等)。
    取代旧 HELLO:端点通告职能取消(拓扑文件静态可算),保留并加强互校——
    版本/两侧卡数/拓扑指纹(digest)不符由云侧拒绝(不回 ack,边侧重发
    耗尽即 fail-fast)。"""

    edge_id: int
    cloud_id: int
    """目标云实例 id:云侧校验"连对了端口"(每个云 dp 一个端口)。"""
    dp_idx: int
    wire_version: int = 0
    edge_npu_count: int = 0
    cloud_npu_count: int = 0
    topology_digest: str = ""
    """sha256(YAML 原文) 前 16 hex;挡住"卡数相同但 ranks 排布不同"的
    错配(该错配挂死在建组期且不报错)。空串 = 旧版未携带,仅跳过比对。"""


class LwdRegisterAckNotify(msgspec.Struct, gc=False, tag=True):
    """云->边注册应答:云引擎全量初始化(权重/KV/图编译)完成后经同一条
    连接回发——边侧收到即"该连接对端就绪"。digest/计数校验已在云侧
    register 入口完成,ack 只携带版本供边侧核对。"""

    cloud_id: int
    dp_idx: int
    wire_version: int = 0


class LwdC2eNotify(msgspec.Struct, gc=False, tag=True):
    """云->边步元数据通告,先于隐藏张量到达,边侧据此预挂精确
    尺寸 recv;hidden_num_elements=0 为纯终结通告,边侧本地终结不派发 unembed。"""

    hidden_num_elements: int
    top_id_ths: list[list[int]]
    num_accepted_tokens: list[int]
    req_ids: list[str]
    finish_reasons: list[int] = []
    """逐请求完成码,与 req_ids 按位对齐:LWD_NOT_FINISHED(-1) = 本步未
    终结;否则为 FinishReason IntEnum 值(STOP=0/LENGTH=1/ABORT=2/ERROR=3/
    REPETITION=4),边侧原样透传 finish_reason(LENGTH 不再伪装成 STOP)。
    空列表 = 云侧未携带,错配即 IndexError fail-fast。"""
    down_seqno: int = -1
    """本步 DOWN 隐藏张量的通道序号,自 LwdC2eMeta.down_seqno 原样透传:
    云 worker 发送时分配(通道级单调),边侧按此值预挂配对 irecv;
    缺省 -1 = 旧版云侧未携带(msgspec 带默认字段,线上 additive 兼容)。

    多连接下的 down_seqno 仍为通道级单域(per-link 拆号与数据面
    D4/D6 的 per-link 组一起落地,本字段届时按连接独立计数)。"""
    edge_id: int = -1
    """目标边归属(fan-in 拆发时按行切组);-1 = 旧版未携带。"""
    dp_idx: int = -1


# typing.Union 而非 PEP 604 `|`:msgspec 解码器的全版本支持路径
LwdNotify = Union[  # noqa: UP007
    LwdRangeNotify,
    LwdRequestNotify,
    LwdAbortNotify,
    LwdRegisterNotify,
]
# 云->边方向:register 应答 + 步元数据(唯一载荷,兼结果回传驱动)
LwdCloudNotify = Union[LwdRegisterAckNotify, LwdC2eNotify]  # noqa: UP007
# 数据面批型定义归 vllm/v1/core/sched/output.py(LwdBatch/LwdBatchType/
# LwdEmbedBatch/LwdUnembedBatch,f8182fd5 定稿),协议层不重复声明

_NOTIFY_DECODER = msgspec.msgpack.Decoder(LwdNotify)
_CLOUD_NOTIFY_DECODER = msgspec.msgpack.Decoder(LwdCloudNotify)


def lwd_encode_notify(message: LwdNotify) -> bytes:
    """通知编码为传输字节(边侧发布端唯一入口)。"""
    return msgspec.msgpack.encode(message)


def lwd_decode_notify(data: bytes) -> LwdNotify:
    """传输字节解码为通知(云侧接收端唯一入口);坏包抛异常,由接收方丢弃。"""
    return _NOTIFY_DECODER.decode(data)


def lwd_encode_cloud_notify(message: LwdCloudNotify) -> bytes:
    """云->边通知编码(云侧发布端唯一入口)。"""
    return msgspec.msgpack.encode(message)


def lwd_decode_cloud_notify(data: bytes) -> LwdCloudNotify:
    """云->边通知解码(边侧接收端唯一入口);坏包抛异常由接收方丢弃。"""
    return _CLOUD_NOTIFY_DECODER.decode(data)


def lwd_check_peer_topology(
    *,
    exp_wire_version: int,
    exp_edge_npu_count: int,
    exp_cloud_npu_count: int,
    exp_topology_digest: str,
    got_wire_version: int,
    got_edge_npu_count: int,
    got_cloud_npu_count: int,
    got_topology_digest: str,
) -> str | None:
    """对端拓扑互校的公共实现(云侧校 register;边侧校 ack 的版本段)。
    None = 通过;否则返回可直接 raise 的错误描述(报出两侧取值,便于定位
    哪边拿错了文件)。互校项:线协议版本、两侧卡数、拓扑指纹。"""
    if got_wire_version != exp_wire_version:
        return (
            f"[Lwd] peer wire version mismatch "
            f"(peer={got_wire_version}, local={exp_wire_version}); "
            "both sides must run the same prefill_only_newsetting build"
        )
    if got_wire_version >= 1 and (
        got_edge_npu_count != exp_edge_npu_count
        or got_cloud_npu_count != exp_cloud_npu_count
    ):
        return (
            f"[Lwd] peer NPU count mismatch (peer announces "
            f"edge={got_edge_npu_count}/cloud={got_cloud_npu_count}, "
            f"local topology edge={exp_edge_npu_count}/"
            f"cloud={exp_cloud_npu_count}); use identical topology "
            "contents on both sides"
        )
    if exp_topology_digest and got_topology_digest and (
        got_topology_digest != exp_topology_digest
    ):
        return (
            f"[Lwd] peer topology digest mismatch "
            f"(peer={got_topology_digest}, local={exp_topology_digest}); "
            "the topology files differ between the two sides"
        )
    return None
