"""云侧 EngineCore 子类:覆写 socket IO 线程入口,PRE_OUT 循环独立成线程,
边侧预告与步内元数据经 input_queue 走原生分发;仅 prefill_only 云角色启用。"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.engine import EngineCoreRequestType, FinishReason
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
    LwdControlPublisher,
)
from vllm.v1.lwd_control.control_communication.lwd_control_subscriber import (
    LwdControlSubscriber,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LWD_NOT_FINISHED,
    LwdAbortNotify,
    LwdC2eNotify,
    LwdHelloNotify,
    LwdRangeNotify,
    LwdRequestNotify,
    lwd_encode_cloud_notify,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import LwdConfig
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.outputs import LwdC2eMeta, ModelRunnerOutput

logger = init_logger(__name__)

# PRE_OUT recv 超时拍:仅作关停响应上限(HELLO 首拍一次,无重发)
LWD_PRE_OUT_RECV_TIMEOUT_MS = 5000

# 步元数据队满重试小睡:元数据不可丢(边侧据此预挂精确尺寸 recv)
_LWD_C2E_SEND_RETRY_SLEEP_S = 0.05


class LwdCloudEngineCore(EngineCoreProc):
    """云 PO 引擎:覆写 socket IO 线程入口,其余全走原生。"""

    def _lwd_setup_zmq(self) -> None:
        """介入 ZMQ 双面:PRE_OUT bind 收边;POST_OUT connect 边,承载首拍
        HELLO 通告与步内元数据。建站失败走 EXECUTOR_FAILED 升级。"""
        config = LwdConfig.from_env_and_config(self.vllm_config)
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
        # 首拍即通告(边侧可能已 bind 等待)
        self._lwd_announce()
        # 门池:元数据查重与暂存,到达即构建放行;仅本 IO 线程独占
        self._lwd_gate_pending: dict[str, LwdRequestNotify] = {}
        # UP 链 seqno 登记(边→云→云 worker 的最后一跳,§9.12 接缝):
        # rid -> [chunk 序 seqno 列表],RangeNotify 到达即登记;调度器
        # 出 prefill 批时取快照挂 SO.lwd_up_seqnos 随批下发云 worker
        # (数据面 UP recv 配对键,与边侧 SO.lwd_batch.seqno 同源同值)。
        # registry 引用交付调度器(IO 线程登记 / 主循环读,dict 赋值原子)。
        self._lwd_seqno_registry: dict[str, list[int]] = {}
        self.scheduler.lwd_seqno_registry = self._lwd_seqno_registry
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
        """父线程照跑父类原版,PRE_OUT 循环独立成线程,两生产者共用 input_queue。"""
        threading.Thread(
            target=self._lwd_pre_out_loop, daemon=True, name="lwd-pre-out"
        ).start()
        super().process_input_sockets(
            input_addresses, coord_input_address, identity, ready_event
        )

    def _lwd_pre_out_loop(self) -> None:
        """PRE_OUT 接收循环:socket 与门状态在本线程内先建后用(zmq 单线程
        亲和);recv 挂超时拍仅作关停响应上限,关停(closed)退出。"""
        try:
            self._lwd_setup_zmq()
        except Exception:
            logger.exception("[Lwd] cloud PRE_OUT setup failed")
            self.input_queue.put_nowait((EngineCoreRequestType.EXECUTOR_FAILED, b""))
            return
        while True:
            msg = self._lwd_subscriber.recv(timeout_ms=LWD_PRE_OUT_RECV_TIMEOUT_MS)
            if msg is None:
                if self._lwd_subscriber.closed:
                    break
                continue
            self._lwd_dispatch(msg)

    def _lwd_announce(self) -> None:
        """首拍 HELLO 通告一次;队满不重试,由边侧等待超时 fail-fast 兜底。"""
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
        """PRE_OUT 三类分派(本 IO 线程):元数据转 Request / abort 终结 /
        范围预告登记 seqno。"""
        if isinstance(msg, LwdRangeNotify):
            # 范围预告:登记 UP 链 seqno(数据面配对键,§9.12)。幂等去重
            # 按"单调性"(seqno 不大于该请求已登记尾号即重复,边侧队满
            # 重试天然产生重复预告,重试复用同一号不产生新登记)。
            seqnos = self._lwd_seqno_registry.setdefault(msg.request_id, [])
            if not seqnos or msg.seqno > seqnos[-1]:
                seqnos.append(msg.seqno)
            # 每条预告都整条入队(重复预告即重复点名,剔除-调度-拼回幂等,
            # 无副作用;PRE_OUT 只 append,调度主线程单独 popleft,deque
            # 单操作原子;预告自带 seqno,出批时作 UP 链配对号)
            self.scheduler.prefill_notify_queue.append(msg)
            return
        if isinstance(msg, LwdAbortNotify):
            self._lwd_gate_pending.pop(msg.request_id, None)
            # 双队列与原生 ABORT 同款:eager 处理 + 保持 input_queue 次序
            self.aborts_queue.put_nowait([msg.request_id])
            self.input_queue.put_nowait((EngineCoreRequestType.ABORT, [msg.request_id]))
            return
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
        """请求构建(唯一建请求点,Request/SamplingParams 留 L3)。

        采样参数自边侧 LwdRequestNotify 透传(采样核/惩罚/EOS 策略/
        min_tokens),云侧按客户端真实参数采样,不再落默认值。"""
        sampling_params = SamplingParams(
            max_tokens=wire.max_tokens,
            temperature=wire.temperature,
            top_p=wire.top_p,
            top_k=wire.top_k,
            min_p=wire.min_p,
            seed=wire.seed,
            repetition_penalty=wire.repetition_penalty,
            presence_penalty=wire.presence_penalty,
            frequency_penalty=wire.frequency_penalty,
            ignore_eos=wire.ignore_eos,
            stop_token_ids=list(wire.stop_token_ids),
            min_tokens=wire.min_tokens,
        )
        local_hasher = self.request_block_hasher
        if local_hasher is None:
            # prefix caching 未启用:请求不挂 hasher,整链机制不激活
            return Request(
                request_id=wire.request_id,
                # 占位 token:云侧调度只看长度,真值由边侧提供
                prompt_token_ids=[0] * wire.num_prompt_tokens,
                sampling_params=sampling_params,
                pooling_params=None,
            )
        hash_block_size = resolve_kv_cache_block_sizes(
            self.scheduler.kv_cache_config, self.vllm_config
        )[1]

        def block_hasher(request: Request) -> list[bytes]:
            # prompt 首建用边侧预告链,续算/缺链/长度不符回退本地(fail-open)
            if len(request.block_hashes) == 0 and request.num_output_tokens == 0:
                expected = request.num_prompt_tokens // hash_block_size
                if len(wire.block_hashes) == expected:
                    return wire.block_hashes
            return local_hasher(request)

        return Request(
            request_id=wire.request_id,
            prompt_token_ids=[0] * wire.num_prompt_tokens,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=block_hasher,
        )
    
    def lwd_handle_model_output(
        self,
        model_output: ModelRunnerOutput,
        engine_core_outputs: dict[int, EngineCoreOutputs],
    ) -> ModelRunnerOutput:
        """步元数据 lwd_c2e_meta 经 POST_OUT 先于隐藏张量发边;逐请求
        finish_reasons 完成码取自本步 engine_core_outputs 的 finish_reason
        (原生停止条件即云侧 decode 终结的事实源),其余原样透传。"""
        meta = model_output.lwd_c2e_meta
        if meta is not None:
            self._lwd_publish_c2e(
                meta, self._lwd_c2e_finish_reasons(meta, engine_core_outputs)
            )
        return model_output

    @staticmethod
    def _lwd_c2e_finish_reasons(
        meta: LwdC2eMeta,
        engine_core_outputs: dict[int, EngineCoreOutputs],
    ) -> list[int]:
        """req_ids 对齐的逐请求完成码:本步任一 EngineCoreOutputs 里带
        finish_reason 的输出取其码;仅进 finished_requests 的缺口按 ABORT
        兜底;其余 LWD_NOT_FINISHED(边侧保持 awaiting,由后续步通告收口)。"""
        reasons: dict[str, int] = {}
        for outputs in engine_core_outputs.values():
            for out in outputs.outputs:
                if out.finish_reason is not None:
                    reasons.setdefault(out.request_id, int(out.finish_reason))
            for request_id in outputs.finished_requests or ():
                reasons.setdefault(request_id, int(FinishReason.ABORT))
        return [reasons.get(request_id, LWD_NOT_FINISHED)
                for request_id in meta.req_ids]

    def _lwd_publish_c2e(self, meta: LwdC2eMeta, finish_reasons: list[int]) -> None:
        """步元数据(含逐请求 finish_reasons 完成码)发边:队满小睡重试
        (元数据不可丢),关停(closed)退出。"""
        notify = LwdC2eNotify(
            hidden_num_elements=meta.hidden_num_elements,
            top_id_ths=meta.top_id_ths,
            num_accepted_tokens=meta.num_accepted_tokens,
            req_ids=meta.req_ids,
            finish_reasons=finish_reasons,
            down_seqno=meta.down_seqno,
        )
        while not self._lwd_post_out.closed:
            if self._lwd_post_out.publish(notify):
                return
            time.sleep(_LWD_C2E_SEND_RETRY_SLEEP_S)
