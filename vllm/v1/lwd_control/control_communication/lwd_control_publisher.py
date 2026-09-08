"""传输层方向原语:OUTBOUND 发布端(side-agnostic)。

边/云身份与 bind/connect 是装配期 wiring(control_scheduler 侧
assemble 文件决定);本类自有线程把有界队列的元数据编码发送,
对谁发布一无所知。
"""

from __future__ import annotations

import contextlib
import queue
import threading

import zmq

from vllm.v1.lwd_control.control_communication.lwd_control_communicator import (
    LwdControlCommunicator,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LwdNotify,
    lwd_encode_notify,
)

# 关停令:发布线程收到即退出;None 不会与通知混淆(它们是 msgspec 结构)
_LWD_PUBLISH_SHUTDOWN = None

LWD_PUBLISH_QUEUE_MAX = 1000


class LwdControlPublisher:
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
        self._closed = False
        self._queue: queue.Queue = queue.Queue(maxsize=queue_max)
        self._communicator = LwdControlCommunicator(endpoint, zmq.PUSH, bind=bind)
        self._thread = threading.Thread(
            target=self._send_thread, name="lwd-publisher", daemon=True
        )
        # 线程最后启动:全部自有状态就绪后才开始消费
        self._thread.start()

    def publish(self, msg: LwdNotify) -> bool:
        """元数据入队;False = 队满未发(可预期失败,不抛异常不丢已发消息)。"""
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            return False
        return True

    def shutdown(self) -> None:
        """关停(幂等):关停令入队 -> join(2s) -> 残留线程以 term 收尾。"""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(queue.Full):
            self._queue.put(_LWD_PUBLISH_SHUTDOWN, timeout=1.0)
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            # 线程卡在阻塞 send(无对端):term 使 send 以 ETERM 退出,
            # 线程随后走 close(0) 收尾;term 本身阻塞至 send 真正返回
            self._communicator.terminate()
            self._thread.join(timeout=1.0)

    def _send_thread(self) -> None:
        while True:
            msg = self._queue.get()
            if msg is _LWD_PUBLISH_SHUTDOWN:
                break
            # 无对端时 send 阻塞,背压由有界队列传导给 publish 返回值
            self._communicator.send(lwd_encode_notify(msg))
        self._communicator.close()
