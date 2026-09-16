"""多边多云下的 req_id 命名空间适配(复用旧架构 id_adapter 语义)。

云侧要能区分请求来自哪条边,沿用旧架构的 req_id 包装方案:
  * 入口包装 ``wrap_req_id(edge_id, req_id) -> "e{edge_id}-{req_id}"``;
  * 出口剥离 ``unwrap_req_id`` 还原边侧原始 id,边完全不感知云内命名空间。

仅三个时刻解析来源:接收时标记来源、回传时解析、失联边残留清理。
边侧 ``req_id`` 在多边间可能碰撞,包装后云内全局唯一。
"""

from __future__ import annotations

_LWD_ID_PREFIX = "e"


def wrap_req_id(edge_id: int, req_id: str) -> str:
    """入口包装:边 id + 原始 req_id -> 云内全局唯一 id。"""
    return f"{_LWD_ID_PREFIX}{edge_id}-{req_id}"


def unwrap_req_id(wrapped: str) -> str:
    """出口剥离:还原边侧原始 req_id;非包装 id 原样返回(单边兼容)。"""
    return parse_edge_id(wrapped)[1]


def parse_edge_id(wrapped: str) -> tuple[int, str]:
    """解析来源边与原始 req_id。

    包装格式 ``e{edge_id}-{req_id}``;非包装 id(无 ``e<num>-`` 前缀)按
    edge_id=0 原样返回,保证单边(未启用命名空间隔离)路径零改动。
    """
    prefix, sep, rest = wrapped.partition("-")
    if (
        sep == "-"
        and len(prefix) > 1
        and prefix[0] == _LWD_ID_PREFIX
        and prefix[1:].isdigit()
    ):
        return int(prefix[1:]), rest
    return 0, wrapped
