"""传输层方向原语:INBOUND 订阅端(side-agnostic)。

边/云身份与 bind/connect 是装配期 wiring(control_scheduler 侧
assemble 文件决定);本类只实现"后台线程接收解码 -> 桥接队列"的
方向语义,不关心消息来自谁。
"""

from __future__ import annotations

import threading
from collections import deque

import msgspec
import zmq

from vllm.logger import init_logger
from vllm.v1.lwd_control.control_communication.lwd_control_communicator import (
    LwdControlCommunicator,
)
from vllm.v1.lwd_control.control_communication.lwd_message import (
    LwdWireMessage,
    lwd_decode_wire,
)

logger = init_logger(__name__)


class LwdControlSubscriber(LwdControlCommunicator):
    """控制面订阅端;元数据阻塞等待绝不丢,消息统一由主循环 drain 处理。"""

    def __init__(self, endpoint: str, *, bind: bool) -> None:
        self._messages: deque[LwdWireMessage] = deque()
        self._lock = threading.Lock()
        super().__init__(endpoint, zmq.PULL, bind=bind, thread_name="lwd-subscriber")

    def _communicator_thread(self) -> None:
        while True:
            try:
                data = self._socket.recv()
                message = lwd_decode_wire(data)
            except zmq.ZMQError:
                # socket 已被 shutdown 关闭:线程唯一退出路径
                break
            except (msgspec.DecodeError, msgspec.ValidationError):
                # 坏帧/垃圾帧丢弃(两者是 msgspec 的平行异常类)
                logger.warning("[Lwd] drop malformed PRE_OUT frame")
                continue
            with self._lock:
                self._messages.append(message)

    def drain(self) -> list[LwdWireMessage]:
        """非阻塞取走积压消息(保持到达序),由调用方按消息 tag 分派。"""
        with self._lock:
            taken = list(self._messages)
            self._messages.clear()
        return taken

    def _request_stop(self) -> None:
        """close(0) 使阻塞 recv 以 ZMQError 退出(pyzmq 重复 close 安全)。"""
        self._socket.close(0)
