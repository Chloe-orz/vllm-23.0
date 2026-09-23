"""控制面通信收发器:ZMQ socket 句柄,只做收发,无线程;通知协议住 lwd_notify。
send/recv 仅持有线程可调(zmq 单线程亲和);term 是跨线程打断阻塞收发的可靠手段。"""

from __future__ import annotations

import zmq
from vllm.logger import init_logger

logger = init_logger(__name__)

_SOCKET_TYPE_NAMES = {
    zmq.PUSH: "PUSH",
    zmq.PULL: "PULL",
    zmq.ROUTER: "ROUTER",
    zmq.DEALER: "DEALER",
}


class LwdControlCommunicator:
    """单向/双向平面收发器:endpoint socket 的收发句柄,无线程。
    ROUTER/DEALER 由 LwdControlLoop 持有使用;信封差异(routing frame)由
    recv_unit/send_unit 吸收,调用方对 socket 类型无感知。"""

    def __init__(
        self,
        endpoint: str | None,
        socket_type: int,
        *,
        bind: bool,
        identity: bytes | None = None,
    ) -> None:
        self._context = zmq.Context()
        self._socket = self._context.socket(socket_type)
        # 退出时不因未发完的帧挂死进程;显式 close(0) 可覆盖
        self._socket.setsockopt(zmq.LINGER, 2000)
        if identity is not None:
            # DEALER 必须在 connect 前设稳定身份:断线重连后 ROUTER 按
            # identity 认回本连接(随机 id 重连即失联)
            self._socket.setsockopt(zmq.IDENTITY, identity)
        if socket_type == zmq.ROUTER:
            # 未知 identity 的定向 send 默认静默丢弃;改为抛
            # EHOSTUNREACH,让发送侧立即感知 peer 掉线(载荷不可丢)
            self._socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self._endpoint: str | None = None
        if endpoint is not None:
            if bind:
                self._socket.bind(endpoint)
            else:
                self._socket.connect(endpoint)
            self._endpoint = endpoint
        logger.info(
            "[Lwd][zmq] communicator up: type=%s mode=%s endpoint=%s",
            _SOCKET_TYPE_NAMES.get(socket_type, str(socket_type)),
            "bind" if bind else "connect",
            endpoint,
        )

    @property
    def endpoint(self) -> str | None:
        """当前连接/bind 的端点。"""
        return self._endpoint

    def send_unit(self, routing_key: bytes | None, payload: bytes) -> None:
        """统一发送:ROUTER 组 [routing_key, payload] 信封定向回包(DEALER
        协议两帧;REQ 的空分隔符三帧形态不适用);其余类型单帧直发。"""
        if routing_key is None:
            self._socket.send(payload)
        else:
            self._socket.send_multipart([routing_key, payload])

    def poll(self, timeout: int) -> bool:
        """等待 socket 可读;timeout 毫秒,就绪返回 True,超时返回 False。"""
        return self._socket.poll(timeout) != 0

    def recv_unit(self, block: bool = True) -> tuple[bytes | None, bytes]:
        """统一接收,返回 (routing_key, payload):ROUTER 收 [identity,
        payload] 两帧信封并剥出对端 identity;DEALER/PULL 收单帧
        (routing_key=None)。
        block=False 无消息时抛 zmq.Again,由调用方处理。"""
        if self._socket.type == zmq.ROUTER:
            identity, payload = self._socket.recv_multipart(
                flags=0 if block else zmq.NOBLOCK
            )
            return identity, payload
        return None, self._socket.recv(flags=0 if block else zmq.NOBLOCK)

    def close(self) -> None:
        """close(0) 立即返回并丢弃未发帧;不保证唤醒阻塞中的 recv(平台相关)。"""
        self._socket.close(0)

    def terminate(self) -> None:
        """term context:使阻塞的收发以 ETERM 返回并阻塞至收尾,供持有方
        join 超时后兜底关停。"""
        self._context.term()

    @property
    def zmq_socket(self) -> zmq.Socket:
        """底层 socket 句柄,仅供持有方线程注册 Poller 使用。"""
        return self._socket
