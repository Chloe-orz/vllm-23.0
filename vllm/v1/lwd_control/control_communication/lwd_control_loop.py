"""控制面 IO 循环:单线程驱动全部控制面 socket(zmq 线程亲和)。

一个类同时承担原 publisher(出站有界队列+发送线程)与原 subscriber/
接收线程(入站 poll+分发)两种角色——DEALER 一条连接一条 socket 且双向
使用,收发必须同线程,两角色模型在 ROUTER/DEALER 下不成立。

边云共用:边侧 sockets = {link: DEALER}(key=link,send 按 link 选路);
云侧 sockets = {None: ROUTER}(routing=True,send 的 key 即对端 identity,
recv 的 routing_key 自信封剥出)。"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Callable
from typing import Any

import zmq

from vllm.logger import init_logger
from vllm.v1.lwd_control.control_communication.lwd_control_communicator import (
    LwdControlCommunicator,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    lwd_encode_notify,
)

logger = init_logger(__name__)

# 出站队列容量:队满时 send 返回 False,背压传导给调用方
LWD_PUBLISH_QUEUE_MAX = 1000

# 关停令:(None, None) 元组入队,msg is None 即关停;不与通知混淆(msgspec 结构)
_LWD_IO_SHUTDOWN = (None, None)

# IO 线程 poll 超时拍:仅作关停响应上限(唤醒靠 inproc PAIR)
_LWD_IO_POLL_TIMEOUT_MS = 5000


class LwdControlLoop:
    """单线程 zmq IO 循环:poller 轮询全部 socket(含唤醒管道),入站解帧后
    on_msg 回调分发,出站经有界队列由本线程发送。

    on_msg(key, routing_key, msg):IO 线程内调用,key 为 sockets 构造键
    (边侧=link),routing_key 为 ROUTER 信封剥出的对端 identity(边侧恒
    None)。回调内只做入队/投递,不做长活。"""

    def __init__(
        self,
        sockets: dict[Any, LwdControlCommunicator],
        *,
        decoder: Callable[[bytes], Any],
        on_msg: Callable[[Any, bytes | None, Any], None],
        encoder: Callable[[Any], bytes] = lwd_encode_notify,
        routing: bool = False,
        queue_max: int = LWD_PUBLISH_QUEUE_MAX,
    ) -> None:
        self._sockets = dict(sockets)
        self._routing = routing
        self._decoder = decoder
        self._encoder = encoder
        self._on_msg = on_msg
        self._queue: queue.Queue = queue.Queue(maxsize=queue_max)
        self._closed = False
        # 唤醒管道:send() 入队后写一字节打断 IO 线程的 poll,使出站即时
        # (每轮循环先无条件 drain 队列,唤醒丢失亦无正确性影响)
        self._wake_context = zmq.Context()
        self._wake_in = self._wake_context.socket(zmq.PAIR)
        self._wake_out = self._wake_context.socket(zmq.PAIR)
        endpoint = f"inproc://lwd-wake-{id(self)}"
        self._wake_in.bind(endpoint)
        self._wake_out.connect(endpoint)
        self._thread = threading.Thread(
            target=self._run, name="lwd-io-loop", daemon=True
        )

    @property
    def closed(self) -> bool:
        """已关停为 True。"""
        return self._closed

    def send(self, key: Any, msg: Any) -> bool:
        """出站入队;False = 队满未发(可预期失败,不抛异常不丢已发消息)。
        ROUTER 模式下 key 为对端 identity,DEALER 模式下为构造键(link)。"""
        if self._closed:
            return False
        try:
            self._queue.put_nowait((key, msg))
        except queue.Full:
            return False
        with contextlib.suppress(zmq.ZMQError):
            self._wake_in.send(b"", flags=zmq.NOBLOCK)
        return True

    def start(self) -> None:
        """启动 IO 线程(全部状态就绪后调用,只许一次)。"""
        self._thread.start()

    def stop(self) -> None:
        """关停(幂等):关停令入队+唤醒 -> join -> 残留线程以 term 收尾。"""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(queue.Full):
            self._queue.put(_LWD_IO_SHUTDOWN, timeout=1.0)
        with contextlib.suppress(zmq.ZMQError):
            self._wake_in.send(b"", flags=zmq.NOBLOCK)
        self._thread.join(timeout=2.0)
        for communicator in self._sockets.values():
            if self._thread.is_alive():
                communicator.terminate()
            communicator.close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        with contextlib.suppress(zmq.ZMQError):
            self._wake_out.close(0)
            self._wake_in.close(0)
        self._wake_context.term()

    def _run(self) -> None:
        poller = zmq.Poller()
        poller.register(self._wake_out, zmq.POLLIN)
        for communicator in self._sockets.values():
            poller.register(communicator.zmq_socket, zmq.POLLIN)
        while True:
            if not self._drain_queue():
                break  # ETERM/关停打断发送,socket 由 stop() 统一收尾
            try:
                events = dict(poller.poll(_LWD_IO_POLL_TIMEOUT_MS))
            except zmq.ZMQError:
                break  # ETERM(关停打断阻塞 poll)
            for key, communicator in self._sockets.items():
                if events.get(communicator.zmq_socket) != zmq.POLLIN:
                    continue
                while True:
                    try:
                        routing_key, payload = communicator.recv_unit(block=False)
                    except zmq.ZMQError:
                        # NOBLOCK 竞态 Again / ETERM(关停打断)均止步本 socket
                        break
                    self._dispatch_inbound(key, routing_key, payload)
            if events.get(self._wake_out) == zmq.POLLIN:
                with contextlib.suppress(zmq.ZMQError):
                    self._wake_out.recv(zmq.NOBLOCK)

    def _drain_queue(self) -> bool:
        """False = 关停令或 ETERM,IO 线程退出;True = 队列已排空。"""
        while True:
            try:
                key, msg = self._queue.get_nowait()
            except queue.Empty:
                return True
            if msg is None:
                # 关停令:不再 drain,socket 由 stop() 统一收尾
                return False
            if not self._send_one(key, msg):
                return False

    def _send_one(self, key: Any, msg: Any) -> bool:
        """单帧发送;False = ETERM(关停打断),调用方停止 drain。
        ROUTER_MANDATORY 的 EHOSTUNREACH(peer 掉线/未注册)在此告警并
        丢弃该帧——控制面载荷不可丢,由调用方重试/重发兜底。"""
        if self._routing:
            # ROUTER 单 socket:key 即对端 identity(定向回包/主动下发)
            communicator = next(iter(self._sockets.values()))
            routing_key = key
        else:
            communicator = self._sockets.get(key)
            routing_key = None
            if communicator is None:
                logger.error("[Lwd][zmq] drop outbound frame: unknown link %r", key)
                return True
        try:
            communicator.send_unit(routing_key, self._encoder(msg))
        except zmq.ContextTerminated:
            return False
        except zmq.ZMQError as exc:
            logger.error(
                "[Lwd][zmq] outbound frame dropped (peer %r unreachable): %s",
                key, exc,
            )
        return True

    def _dispatch_inbound(self, key: Any, routing_key: bytes | None, payload: bytes) -> None:
        try:
            msg = self._decoder(payload)
        except Exception:
            logger.warning("[Lwd][zmq] drop malformed notify frame")
            return
        self._on_msg(key, routing_key, msg)
