"""云侧 ZMQ 通道:仅控制面订阅(PRE_OUT);无快路径线程内直投(§9)。"""

from __future__ import annotations


class LwdCloudControlSubscriber:
    """PRE_OUT 订阅端;元数据阻塞等待绝不丢,消息统一由主循环 drain 处理。"""

    def __init__(self, endpoint: str) -> None:
        ...

    def _subscriber_thread(self) -> None:
        ...

    def drain(self) -> "list[object]":
        """非阻塞取走积压消息(notify/add_request/abort 统一路径)。"""
        ...

    def shutdown(self) -> None:
        ...
