"""传输层方向原语:INBOUND 订阅端(side-agnostic)。

边/云身份与 bind/connect 是装配期 wiring(control_scheduler 侧
assemble 文件决定);本类无线程,阻塞 recv 由调用方线程驱动,收到即
解码返回(坏包丢弃取下一条);关停(ETERM)返回 None。
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
    """控制面订阅端;阻塞 recv 一条解码一条,等待节奏由调用方定。"""

    def __init__(self, endpoint: str, *, bind: bool) -> None:
        self._closed = False
        self._socket = LwdControlCommunicator(endpoint, zmq.PULL, bind=bind)

    def recv(self) -> LwdNotify | None:
        """阻塞收一条并解码;关停(ETERM/已 close)返回 None;坏包丢弃取下一条。"""
        while not self._closed:
            try:
                data = self._socket.recv()
            except zmq.ZMQError:
                return None
            try:
                return lwd_decode_notify(data)
            except (msgspec.DecodeError, msgspec.ValidationError):
                logger.warning("[Lwd] drop malformed PRE_OUT notify")
        return None

    def shutdown(self) -> None:
        """关停(幂等):close 释放 fd,term 使阻塞 recv 以 ETERM 返回。"""
        if self._closed:
            return
        self._closed = True
        self._socket.close()
        self._socket.terminate()
