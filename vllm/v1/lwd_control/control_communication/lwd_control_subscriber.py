"""传输层方向原语:INBOUND 订阅端(side-agnostic)。

边/云身份与 bind/connect 是装配期 wiring(control_scheduler 侧
assemble 文件决定);本类自有线程接收解码入桥接队列,不关心消息
来自谁。
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
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LwdNotify,
    lwd_decode_notify,
)

logger = init_logger(__name__)


class LwdControlSubscriber:
    """控制面订阅端;元数据阻塞等待绝不丢,消息统一由主循环 drain 处理。"""

    def __init__(self, endpoint: str, *, bind: bool) -> None:
        self._closed = False
        self._messages: deque[LwdNotify] = deque()
        self._lock = threading.Lock()
        self._communicator = LwdControlCommunicator(endpoint, zmq.PULL, bind=bind)
        self._thread = threading.Thread(
            target=self._receive_thread, name="lwd-subscriber", daemon=True
        )
        # 线程最后启动:全部自有状态就绪后才开始接收
        self._thread.start()

    def drain(self) -> list[LwdNotify]:
        """非阻塞取走积压消息(保持到达序),由调用方按消息 tag 分派。"""
        with self._lock:
            taken = list(self._messages)
            self._messages.clear()
        return taken

    def shutdown(self) -> None:
        """关停(幂等):close 释放 fd -> term 打断阻塞 recv -> join(2s) 收尸。"""
        if self._closed:
            return
        self._closed = True
        self._communicator.close()
        # close(0) 不保证唤醒跨线程阻塞 recv(实测 macOS 不唤醒):
        # term 使 recv 以 ETERM 返回,是唯一可靠的打断手段
        self._communicator.terminate()
        self._thread.join(timeout=2.0)

    def _receive_thread(self) -> None:
        while True:
            try:
                data = self._communicator.recv()
                message = lwd_decode_notify(data)
            except zmq.ZMQError:
                # close 后再 recv / term 打断(ETERM)均走此退出
                break
            except (msgspec.DecodeError, msgspec.ValidationError):
                # 坏包/垃圾数据丢弃(两者是 msgspec 的平行异常类)
                logger.warning("[Lwd] drop malformed PRE_OUT notify")
                continue
            with self._lock:
                self._messages.append(message)
