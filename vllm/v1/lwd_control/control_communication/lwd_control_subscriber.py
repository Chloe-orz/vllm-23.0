"""传输层方向原语:INBOUND 订阅端(side-agnostic)。

边/云身份与 bind/connect 是装配期 wiring(control_scheduler 侧
assemble 文件决定);本类无线程,收发由调用方线程驱动 —— recv_available
阻塞至多有消息或超时,返回整批已解码通知,调用方按消息 tag 分派。
坏包/垃圾数据丢弃(两者是 msgspec 的平行异常类),其余异常上抛调用方。
"""

from __future__ import annotations

import msgspec
import zmq

from vllm.logger import init_logger
from vllm.v1.lwd_control.control_communication.lwd_control_communicator import (
    LwdControlCommunicator,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LwdNotify,
    lwd_decode_notify,
)

logger = init_logger(__name__)


class LwdControlSubscriber:
    """控制面订阅端;元数据阻塞等待绝不丢,由调用方决定阻塞位置。"""

    def __init__(self, endpoint: str, *, bind: bool) -> None:
        self._closed = False
        self._socket = LwdControlCommunicator(endpoint, zmq.PULL, bind=bind)

    def recv_available(self, timeout: float) -> list[LwdNotify]:
        """阻塞至多有消息或超时(s,0 = 非阻塞),返回整批通知(到达序)。"""
        if timeout > 0 and not self._socket.poll(int(timeout * 1000)):
            return []
        messages: list[LwdNotify] = []
        while True:
            try:
                data = self._socket.recv(block=False)
            except zmq.Again:
                break
            try:
                messages.append(lwd_decode_notify(data))
            except (msgspec.DecodeError, msgspec.ValidationError):
                logger.warning("[Lwd] drop malformed PRE_OUT notify")
        return messages

    def shutdown(self) -> None:
        """关停(幂等):close 释放 fd,terminate 回收 ctx。"""
        if self._closed:
            return
        self._closed = True
        self._socket.close()
        self._socket.terminate()
