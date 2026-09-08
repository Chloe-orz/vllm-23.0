"""控制面通信公共基类:ZMQ socket 的线程所有权与生命周期约定(§9.1)。

本文件只管生命周期,不知道任何消息类型;线上协议住 lwd_message.py
(协议/生命周期两条变化轴分文件,§10.3)。共享骨架收敛于此:
context/socket/线程的建立与统一关停(socket 只被所属线程或 shutdown
路径关闭,关停幂等、join 上限 2s)。派生类只实现方向语义
(publish / drain)、线程主体(_communicator_thread)与退出触发
(_request_stop)。
"""

from __future__ import annotations

import threading

import zmq


class LwdControlCommunicator:
    """PRE_OUT 平面收发器公共骨架:endpoint + 后台线程 + 幂等关停。"""

    def __init__(
        self,
        endpoint: str,
        socket_type: int,
        *,
        bind: bool,
        thread_name: str,
    ) -> None:
        """建 socket(按 bind 参数决定 bind/connect)与后台线程;线程立即启动。

        派生类自有状态(队列/锁)须在 super().__init__ 之前初始化。
        """
        self._closed = False
        self._context = zmq.Context()
        self._socket = self._context.socket(socket_type)
        # 退出时不因未发完的帧挂死进程;显式 close(0) 可覆盖
        self._socket.setsockopt(zmq.LINGER, 2000)
        if bind:
            self._socket.bind(endpoint)
        else:
            self._socket.connect(endpoint)
        self._thread = threading.Thread(
            target=self._communicator_thread, name=thread_name, daemon=True
        )
        self._thread.start()

    def shutdown(self) -> None:
        """关停(幂等):触发线程退出 -> join(2s) -> 线程已死才 term context。"""
        if self._closed:
            return
        self._closed = True
        self._request_stop()
        self._thread.join(timeout=2.0)
        if not self._thread.is_alive():
            self._context.term()

    def _communicator_thread(self) -> None:
        """线程主体;socket 由基类创建,派生类只写循环与退出路径。"""
        raise NotImplementedError

    def _request_stop(self) -> None:
        """触发线程退出(关停令/关 socket 等);必须幂等。"""
        raise NotImplementedError
