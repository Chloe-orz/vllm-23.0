"""云侧 L3 引擎子类:类选择点注入的云 EngineCore(§10.14,替代已删的
lwd_cloud_assemble 装配层)。

接入点只有一个:覆写原生 socket IO 线程入口 process_input_sockets ——
父线程照跑父类原版(前端消息零复制零改动),PRE_OUT 循环独立成
lwd-pre-out 专属线程,通道与门状态在该线程内先建后用(无线程竞态)。
云侧双面(§9.1):PRE_OUT bind 收边;POST_OUT 经 master_addr connect
边,周期 HELLO 通告云端点(边阻塞等待,决策 B 的唯一事实源),该面
数据传输保留给后续云->边扩展。
两类消息都汇入 input_queue(多生产者-单消费者):
父类线程产前端消息,PRE_OUT 线程产 (ADD, (Request, 0)) / (ABORT, [rid]) ——
主循环 _handle_client_request 原生分发,零改动。
空闲唤醒由原生机制自然解决:PRE_OUT 线程阻塞在 zmq 上,消息转成 ADD 塞进
input_queue,主循环的 input_queue.get() 随即被唤醒 —— 无轮询、无
空闲切换、core.py 零改动。

PRE_OUT 三类消息是协议:request 预告进门池;range(offset==0)开门
(边侧真派发了首块才开算);abort 终结。数据面接缝(hint 转发,§9.12)
随数据面落位时再接。

本类仅在 mode=prefill_only 且云角色时经类选择点构造;部署需
max_concurrent_batches > 1(异步流水线,同步步路径兼容但慢)。
空批契约垫片已按裁定移除(原防相位调度器刻意空步触发 fork runner
0-token 批回 None,复现表现 = core.py:576 RuntimeError;防护挂
runner 侧契约修复)。
"""

from __future__ import annotations

import threading

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
    LwdControlPublisher,
)
from vllm.v1.lwd_control.control_communication.lwd_control_subscriber import (
    LwdControlSubscriber,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LwdAbortNotify,
    LwdHelloNotify,
    LwdRangeNotify,
    LwdRequestNotify,
    lwd_encode_cloud_notify,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import LwdConfig
from vllm.v1.request import Request

logger = init_logger(__name__)

# HELLO 周期重发间隔(挂 PRE_OUT 循环的 recv 超时拍上,不另起定时器)
LWD_HELLO_RESEND_INTERVAL_MS = 5000


class LwdCloudEngineCore(EngineCoreProc):
    """云 PO 引擎:覆写 socket IO 线程入口,其余全走原生。"""

    def _lwd_setup_zmq(self) -> None:
        """介入 ZMQ 双面(§9.1):PRE_OUT bind 收边 + POST_OUT connect 通告边。

        POST_OUT 经 master_addr 连边(边 bind),承载周期 HELLO——
        云端点(pre_out_*)的唯一事实源;该面当前仅发现用途,数据
        传输保留给后续云->边扩展。建站失败仍走 EXECUTOR_FAILED 升级。
        """
        config = LwdConfig.from_env_and_config(self.vllm_config)
        self._lwd_config = config
        self._lwd_subscriber = LwdControlSubscriber(
            config.lwd_pre_out_endpoint(), bind=True
        )
        master_addr = self.vllm_config.parallel_config.master_addr
        self._lwd_post_out = LwdControlPublisher(
            f"tcp://{master_addr}:{config.post_out_port}",
            bind=False,
            encoder=lwd_encode_cloud_notify,
        )
        self._lwd_hello = LwdHelloNotify(
            pre_out_host=config.pre_out_host, pre_out_port=config.pre_out_port
        )
        # 首拍即通告(边可能已 bind 等待);此后周期重发覆盖边重启场景
        self._lwd_announce()
        # 首预告门:request 进门池,range(offset==0)开门;乱序防御 =
        # 双侧检查(先 range 后 request 到达同样放行)。门归 socket IO
        # 线程独占;调度器只经 input_queue 被主循环碰。
        self._lwd_gate_pending: dict[str, LwdRequestNotify] = {}
        self._lwd_gate_ready: set[str] = set()
        logger.info(
            "[Lwd] cloud engine assembled: PRE_OUT bind %s, POST_OUT announce -> "
            "%s:%s via master %s",
            config.lwd_pre_out_endpoint(),
            config.pre_out_host,
            config.pre_out_port,
            master_addr,
        )

    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ) -> None:
        """原生 IO 线程入口的云侧版:父线程照跑父类原版,PRE_OUT 循环
        独立成线程 —— 两个生产者共用 input_queue。"""
        threading.Thread(
            target=self._lwd_pre_out_loop, daemon=True, name="lwd-pre-out"
        ).start()
        super().process_input_sockets(
            input_addresses, coord_input_address, identity, ready_event
        )

    def _lwd_pre_out_loop(self) -> None:
        """边侧 PRE_OUT 接收循环:subscriber 与门状态在本线程内先建后用,
        无构造期竞态,socket 建用同线程(zmq 单线程亲和);建站失败经
        EXECUTOR_FAILED 通道升级为引擎致命错误,不静默降级。

        recv 挂超时拍:空闲拍上重发 HELLO(边重启后重新发现的载体),
        关停(closed)退出。"""
        try:
            self._lwd_setup_zmq()
        except Exception:
            logger.exception("[Lwd] cloud PRE_OUT setup failed")
            self.input_queue.put_nowait((EngineCoreRequestType.EXECUTOR_FAILED, b""))
            return
        while True:
            msg = self._lwd_subscriber.recv(timeout_ms=LWD_HELLO_RESEND_INTERVAL_MS)
            if msg is None:
                if self._lwd_subscriber.closed:
                    break
                self._lwd_announce()
                continue
            self._lwd_dispatch(msg)

    def _lwd_announce(self) -> None:
        """HELLO 通告(幂等):队满失败不重试,周期拍自愈。"""
        self._lwd_post_out.publish(self._lwd_hello)

    def shutdown(self) -> None:
        """两面关停后走原生(幂等;装配失败路径两面可能未建,容忍缺省)。"""
        subscriber = getattr(self, "_lwd_subscriber", None)
        if subscriber is not None:
            subscriber.shutdown()
        publisher = getattr(self, "_lwd_post_out", None)
        if publisher is not None:
            publisher.shutdown()
        super().shutdown()

    def _lwd_dispatch(self, msg) -> None:
        """PRE_OUT 三类分派(本 IO 线程):门/转换/终结。"""
        if isinstance(msg, LwdRangeNotify):
            if msg.offset == 0:
                # 首预告门:offset==0 = 边侧派发首块,开门放行
                self._lwd_gate_ready.add(msg.request_id)
                self._lwd_promote(msg.request_id)
        elif isinstance(msg, LwdAbortNotify):
            self._lwd_gate_pending.pop(msg.request_id, None)
            self._lwd_gate_ready.discard(msg.request_id)
            # 双队列与原生 ABORT 同款:eager 处理 + 保持 input_queue 次序
            self.aborts_queue.put_nowait([msg.request_id])
            self.input_queue.put_nowait((EngineCoreRequestType.ABORT, [msg.request_id]))
        else:
            rid = msg.request_id
            if rid in self._lwd_gate_pending:
                logger.warning("[Lwd] duplicate request metadata %s ignored", rid)
                return
            self._lwd_gate_pending[rid] = msg
            self._lwd_promote(rid)

    def _lwd_promote(self, request_id: str) -> None:
        """过门:门池取 wire,转 Request 投 input_queue 走原生 ADD 分发。"""
        wire = self._lwd_gate_pending.pop(request_id, None)
        if wire is not None:
            request = self._lwd_build_request(wire)
            self.input_queue.put_nowait((EngineCoreRequestType.ADD, (request, 0)))
            logger.info("[Lwd] cloud request %s admitted via gate", request_id)

    def _lwd_build_request(self, wire: LwdRequestNotify) -> Request:
        """请求构建(唯一建请求点,Request/SamplingParams 留 L3)。"""
        local_hasher = self.request_block_hasher
        if local_hasher is None:
            # prefix caching 未启用:请求不挂 hasher,整链机制不激活
            return Request(
                request_id=wire.request_id,
                # 占位 token:云侧调度只看长度,真值由边侧 embeds 提供(§9.5)
                prompt_token_ids=[0] * wire.num_prompt_tokens,
                sampling_params=SamplingParams(max_tokens=wire.max_tokens),
                pooling_params=None,
            )
        hash_block_size = resolve_kv_cache_block_sizes(
            self.scheduler.kv_cache_config, self.vllm_config
        )[1]

        def block_hasher(request: Request) -> list[bytes]:
            # wire 链优先(§10.13):prompt 首建用边侧预告链,decode 续算
            # 回本地 hasher;缺链/长度不符回退本地(fail-open)
            if len(request.block_hashes) == 0 and request.num_output_tokens == 0:
                expected = request.num_prompt_tokens // hash_block_size
                if len(wire.block_hashes) == expected:
                    return wire.block_hashes
            return local_hasher(request)

        return Request(
            request_id=wire.request_id,
            prompt_token_ids=[0] * wire.num_prompt_tokens,
            sampling_params=SamplingParams(max_tokens=wire.max_tokens),
            pooling_params=None,
            block_hasher=block_hasher,
        )
