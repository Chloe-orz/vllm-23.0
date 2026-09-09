"""传输层方向原语:INBOUND 订阅端(side-agnostic)。

边/云身份与 bind/connect 是装配期 wiring(control_scheduler 侧
assemble 文件决定);本类无线程,阻塞 recv 由调用方线程驱动,收到即
解码返回(坏包丢弃取下一条);关停(ETERM)返回 None。decoder 经
构造注入:PRE_OUT 面传边->云通知解码器,POST_OUT 面传云->边解码器
(§10.7 协议分面)。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import msgspec
import zmq

from vllm.logger import init_logger
from vllm.v1.lwd_control.control_communication.lwd_control_communicator import (
    LwdControlCommunicator,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    lwd_decode_notify,
)

logger = init_logger(__name__)


class LwdControlSubscriber:
    """控制面订阅端;阻塞 recv 一条解码一条,等待节奏由调用方定。

    recv(timeout_ms) 超时与关停都返回 None,以 closed 属性区分——
    周期超时给调用方做心跳位(云侧 HELLO 重发即挂在此返回值上)。
    """

    def __init__(
        self,
        endpoint: str,
        *,
        bind: bool,
        decoder: Callable[[bytes], Any] = lwd_decode_notify,
    ) -> None:
        self._closed = False
        self._decoder = decoder
        self._socket = LwdControlCommunicator(endpoint, zmq.PULL, bind=bind)

    @property
    def closed(self) -> bool:
        """已关停为 True;用于区分 recv 超时返回与关停返回。"""
        return self._closed

    def recv(self, timeout_ms: int | None = None) -> Any | None:
        """收一条并解码;超时/关停返回 None(以 closed 区分);坏包丢弃取下一条。"""
        while not self._closed:
            try:
                if timeout_ms is None:
                    data = self._socket.recv()
                else:
                    if not self._socket.poll(timeout_ms):
                        return None
                    data = self._socket.recv(block=False)
            except zmq.ZMQError:
                # ETERM(关停打断)与 NOBLOCK 竞态 Again 都走此出口
                return None
            try:
                return self._decoder(data)
            except (msgspec.DecodeError, msgspec.ValidationError):
                logger.warning("[Lwd] drop malformed notify frame")
        return None

    def shutdown(self) -> None:
        """关停(幂等):close 释放 fd,term 使阻塞 recv 以 ETERM 返回。"""
        if self._closed:
            return
        self._closed = True
        self._socket.close()
        self._socket.terminate()
