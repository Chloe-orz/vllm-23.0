"""传输层方向原语:OUTBOUND 发布端(side-agnostic)。

边/云身份与 bind/connect 是装配期 wiring(control_scheduler 侧
assemble 文件决定);本类只实现"有界队列 -> 后台线程编码发送"的
方向语义,对谁发布一无所知。
"""

from __future__ import annotations

import contextlib
import queue

import zmq

from vllm.v1.lwd_control.control_communication.lwd_control_communicator import (
    LwdControlCommunicator,
)
from vllm.v1.lwd_control.control_communication.lwd_message import (
    LwdWireMessage,
    lwd_encode_wire,
)

# 关停令:发布线程收到即退出;None 不会与线上消息混淆(它们是 msgspec 结构)
_LWD_PUBLISH_SHUTDOWN = None

LWD_PUBLISH_QUEUE_MAX = 1000


class LwdControlPublisher(LwdControlCommunicator):
    """控制面发布端;publish 队满返回 False,调用方视为未派发、下一步重试(§2.4 背压)。

    PUSH 的阻塞语义天然形成对端背压:对端消费慢 -> send 阻塞 -> 内部
    有界队列涨满 -> publish 返回 False,本步不执行(§9.1 静态封顶)。
    """

    def __init__(
        self,
        endpoint: str,
        *,
        bind: bool,
        queue_max: int = LWD_PUBLISH_QUEUE_MAX,
    ) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=queue_max)
        super().__init__(endpoint, zmq.PUSH, bind=bind, thread_name="lwd-publisher")

    def publish(self, msg: LwdWireMessage) -> bool:
        """元数据入队;False = 队满未发(可预期失败,不抛异常不丢已发消息)。"""
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            return False
        return True

    def _communicator_thread(self) -> None:
        while True:
            msg = self._queue.get()
            if msg is _LWD_PUBLISH_SHUTDOWN:
                break
            # 无对端时 send 阻塞,背压由有界队列传导给 publish 返回值
            self._socket.send(lwd_encode_wire(msg))
        self._socket.close(0)

    def _request_stop(self) -> None:
        """关停令入队;送不出去(线程卡在阻塞 send)时随进程退出。"""
        with contextlib.suppress(queue.Full):
            self._queue.put(_LWD_PUBLISH_SHUTDOWN, timeout=1.0)
