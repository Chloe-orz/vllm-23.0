"""云侧 L3 引擎子类:类选择点注入的云 EngineCore(§10.14,替代已删的
lwd_cloud_assemble 装配层)。

__init__ 一处云侧介入:ZMQ 收发。接收在 _process_input_queue(循环
线程):空闲阻塞等 PRE_OUT,收到经首预告门/请求转换直接进调度器,
门/暂存/调度簿记全单线程化。PRE_OUT 三类消息是协议:request 预告
进门池;range(offset==0)开门(边侧真派发了首块才开算);abort 终结。
数据面接缝(hint 转发,§9.12)随数据面落位时再接。

本类仅在 mode=prefill_only 且云角色时经类选择点构造;部署需
max_concurrent_batches > 1(异步流水线,同步步路径兼容但慢)。
空批契约垫片已按裁定移除(原防相位调度器刻意空步触发 fork runner
0-token 批回 None,复现表现 = core.py:576 RuntimeError;防护挂
runner 侧契约修复)。
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.lwd_control.control_communication.lwd_control_subscriber import (
    LwdControlSubscriber,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LwdAbortNotify,
    LwdRangeNotify,
    LwdRequestNotify,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import LwdConfig
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)

# 空闲等 PRE_OUT 的单轮超时(s):上界 = 关停信号/客户端 UTILITY 消息的
# 最坏感知延迟,下界无关紧要(poll 就绪即醒)
_LWD_IDLE_RECV_TIMEOUT_S = 0.1


class LwdCloudEngineCore(EngineCoreProc):
    """云 PO 引擎:ZMQ 收发一处介入,其余全走原生。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lwd_setup_zmq()

    def shutdown(self) -> None:
        self._lwd_subscriber.shutdown()
        super().shutdown()

    def _lwd_setup_zmq(self) -> None:
        """介入 ZMQ:订阅通道(bind)+ 首预告门状态。"""
        self._lwd_subscriber = LwdControlSubscriber(
            LwdConfig.from_env_and_config(self.vllm_config).lwd_pre_out_endpoint(),
            bind=True,
        )
        # 首预告门:request 进门池,range(offset==0)开门;乱序防御 =
        # 双侧检查(先 range 后 request 到达同样放行)
        self._lwd_gate_pending: dict[str, LwdRequestNotify] = {}
        self._lwd_gate_ready: set[str] = set()
        logger.info("[Lwd] cloud engine assembled: main-loop PRE_OUT pump")

    def _process_input_queue(self) -> None:
        """结构对齐原生 _process_input_queue,唯一差异是空闲阻塞点:
        原生阻塞在 input_queue.get() 等客户端请求,headless 云没有
        客户端,改为阻塞等边侧 PRE_OUT(0.1s 轮询,响应 signal 关停);
        input_queue 有消息时同样处理(UTILITY/EXECUTOR_FAILED 等照常走)。
        退出循环后 has_work=True 或已请求关停,此时调 super() 只会做
        input_queue 非阻塞排空,不会再阻塞。"""
        self._lwd_pump_pre_out(timeout=0)
        while self.is_running() and not self.has_work():
            self._notify_idle_state_callbacks()
            if not self.input_queue.empty():
                req = self.input_queue.get_nowait()
                self._handle_client_request(*req)
            else:
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
                self._lwd_pump_pre_out(timeout=_LWD_IDLE_RECV_TIMEOUT_S)
        super()._process_input_queue()

    def _lwd_pump_pre_out(self, timeout: float) -> None:
        """收 PRE_OUT 并分派(循环线程;阻塞至多有消息或超时)。"""
        for msg in self._lwd_subscriber.recv_available(timeout):
            if isinstance(msg, LwdRangeNotify):
                if msg.offset == 0:
                    # 首预告门:offset==0 = 边侧派发首块,开门放行
                    self._lwd_gate_ready.add(msg.request_id)
                    self._lwd_promote(msg.request_id)
            elif isinstance(msg, LwdAbortNotify):
                self._lwd_gate_pending.pop(msg.request_id, None)
                self._lwd_gate_ready.discard(msg.request_id)
                self.scheduler.finish_requests(
                    [msg.request_id], RequestStatus.FINISHED_ABORTED
                )
            else:
                rid = msg.request_id
                if rid in self._lwd_gate_pending:
                    logger.warning("[Lwd] duplicate request metadata %s ignored", rid)
                    return
                self._lwd_gate_pending[rid] = msg
                self._lwd_promote(rid)

    def _lwd_promote(self, request_id: str) -> None:
        """过门:门池取 wire,转 Request 直接进调度器(暂存池/原生路径)。"""
        wire = self._lwd_gate_pending.pop(request_id, None)
        if wire is not None:
            self.scheduler.add_request(self._lwd_build_request(wire))

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
