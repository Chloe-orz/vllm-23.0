"""边侧 ZMQ 通道:仅控制面发布(PRE_OUT,边->云单向);无结果面接收(§9)。"""

from __future__ import annotations


class LwdEdgeControlPublisher:
    """PRE_OUT 发布端;publish 队满返回 False,调用方视为未派发、下一步重试(§2.4 背压)。"""

    def __init__(self, endpoint: str) -> None:
        ...

    def publish(self, msg: object) -> bool:
        """元数据入队;False = 队满未发(可预期失败,不抛异常不丢已发消息)。"""
        ...

    def _publisher_thread(self) -> None:
        ...

    def shutdown(self) -> None:
        ...
