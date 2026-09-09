"""控制面通信收发器:ZMQ socket 句柄,只做收发,无线程(§9.1)。

socket 单线程亲和:send/recv 只能由持有线程调用;close/term 是允许
跨线程的关停调用。term 是跨线程打断阻塞 recv/send 的可靠手段
(使其以 ETERM 返回);close(0) 不保证唤醒。关停编排由持有方负责。
本文件不知道任何通知类型;通知协议住 lwd_notify.py(§10.3)。
"""

from __future__ import annotations

import zmq


class LwdControlCommunicator:
    """单向平面收发器:endpoint socket 的收发句柄,无线程。

    endpoint=None 构造延迟连接态(PUSH 无对端只入队不报错),
    由持有方线程经 retarget 补连——边侧 PRE_OUT 的目标来自云侧
    HELLO 通告,装配期不可知。
    """

    def __init__(self, endpoint: str | None, socket_type: int, *, bind: bool) -> None:
        self._context = zmq.Context()
        self._socket = self._context.socket(socket_type)
        # 退出时不因未发完的帧挂死进程;显式 close(0) 可覆盖
        self._socket.setsockopt(zmq.LINGER, 2000)
        self._endpoint: str | None = None
        if endpoint is not None:
            if bind:
                self._socket.bind(endpoint)
            else:
                self._socket.connect(endpoint)
            self._endpoint = endpoint

    @property
    def endpoint(self) -> str | None:
        """当前连接/bind 的端点;延迟连接态为 None。"""
        return self._endpoint

    def retarget(self, endpoint: str) -> None:
        """换连接目标(仅持有线程调用):先连新再断旧,换址期消息不丢。

        ZMQ connect 惰性且可并存多管道,先连新端点使后续消息有路可
        走,再摘除旧端点注册;同址重复调用幂等(重复 HELLO 误发时只连一次)。
        """
        if endpoint == self._endpoint:
            return
        self._socket.connect(endpoint)
        if self._endpoint is not None:
            self._socket.disconnect(self._endpoint)
        self._endpoint = endpoint

    def send(self, data: bytes) -> None:
        """阻塞发送;无对端时阻塞,由调用方有界队列形成对端背压。"""
        self._socket.send(data)

    def poll(self, timeout: int) -> bool:
        """等待 socket 可读;timeout 毫秒,就绪返回 True,超时返回 False。"""
        return self._socket.poll(timeout) != 0

    def recv(self, block: bool = True) -> bytes:
        """接收;block=False 非阻塞,无消息时抛 zmq.Again。"""
        return self._socket.recv(flags=0 if block else zmq.NOBLOCK)

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
