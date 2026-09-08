"""控制面通信收发器:ZMQ socket 句柄,只做收发,无线程(§9.1)。

socket 单线程亲和:send/recv 只能由持有线程调用;close/term 是允许
跨线程的关停调用。term 是跨线程打断阻塞 recv/send 的可靠手段
(使其以 ETERM 返回);close(0) 不保证唤醒。关停编排由持有方负责。
本文件不知道任何通知类型;通知协议住 lwd_notify.py(§10.3)。
"""

from __future__ import annotations

import zmq


class LwdControlCommunicator:
    """PRE_OUT 平面收发器:endpoint socket 的收发句柄,无线程。"""

    def __init__(self, endpoint: str, socket_type: int, *, bind: bool) -> None:
        self._context = zmq.Context()
        self._socket = self._context.socket(socket_type)
        # 退出时不因未发完的帧挂死进程;显式 close(0) 可覆盖
        self._socket.setsockopt(zmq.LINGER, 2000)
        if bind:
            self._socket.bind(endpoint)
        else:
            self._socket.connect(endpoint)

    def send(self, data: bytes) -> None:
        """阻塞发送;无对端时阻塞,由调用方有界队列形成对端背压。"""
        self._socket.send(data)

    def recv(self) -> bytes:
        """阻塞接收;已 close 的 socket 再 recv 抛 zmq.ZMQError。"""
        return self._socket.recv()

    def close(self) -> None:
        """close(0) 立即返回并丢弃未发帧(pyzmq 重复 close 安全)。

        注意:close 不保证唤醒其他线程阻塞中的 recv(平台相关)。
        """
        self._socket.close(0)

    def terminate(self) -> None:
        """term context:使该 context 上阻塞的收发以 ETERM 返回,并阻塞至其收尾。

        这是跨线程打断阻塞 recv/send 的可靠手段;持有方在 join 超时后
        以此收尾,term 返回后线程必已(或即将)退出。
        """
        self._context.term()
