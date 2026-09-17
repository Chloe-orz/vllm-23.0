"""云侧 EngineCore 子类:构造期建 ZMQ 双面并起 PRE_OUT 接收线程,边侧
预告与步内元数据经 input_queue 走原生分发;仅 prefill_only 云角色启用。"""

from __future__ import annotations

import threading

import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.engine import EngineCoreRequestType
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
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_scheduler import (
    LwdCloudScheduler,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import LwdBaseEngineCore
from vllm.v1.request import Request

logger = init_logger(__name__)

# PRE_OUT recv 超时拍:仅作关停响应上限(HELLO 首拍一次,无重发)
LWD_PRE_OUT_RECV_TIMEOUT_MS = 5000


class LwdCloudEngineCore(LwdBaseEngineCore):
    """云 PO 引擎:构造期建 ZMQ 双面 + 起 PRE_OUT 接收线程,其余全走原生。"""

    lwd_scheduler_cls = LwdCloudScheduler

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lwd_setup_zmq()
        threading.Thread(
            target=self._lwd_pre_out_loop, daemon=True, name="lwd-pre-out"
        ).start()

    def _lwd_setup_zmq(self) -> None:
        """建 ZMQ 双面:PRE_OUT bind 收边;POST_OUT connect 边,承载
        HELLO 通告与步元数据。建站失败即构造失败(fail-fast)。"""
        config = self.lwd_config
        self._subscriber = LwdControlSubscriber(
            f"tcp://{config.pre_out_host}:{config.pre_out_port}", bind=True
        )
        master_addr = self.vllm_config.parallel_config.master_addr
        self._publisher = LwdControlPublisher(
            f"tcp://{master_addr}:{config.post_out_port}",
            bind=False,
            encoder=lwd_encode_cloud_notify,
        )
        # 步元数据发布面交付调度器(update_from_output 覆写消费)
        self.scheduler.lwd_cloud_publisher = self._publisher
        # 首拍即通告(边侧可能已 bind 等待);队满不重试,由边侧
        # 等待超时 fail-fast 兜底
        hello = LwdHelloNotify(
            pre_out_host=config.pre_out_host, pre_out_port=config.pre_out_port
        )
        self._publisher.publish(hello)
        logger.info(
            "[Lwd][cloud] HELLO announced: pre_out=%s:%s",
            config.pre_out_host, config.pre_out_port,
        )
        # 门池:元数据查重与暂存,到达即构建放行;仅接收线程独占
        self._lwd_gate_pending: dict[str, LwdRequestNotify] = {}
        logger.info(
            "[Lwd] cloud engine assembled: PRE_OUT bind %s, POST_OUT announce -> "
            "%s:%s via master %s",
            f"tcp://{config.pre_out_host}:{config.pre_out_port}",
            config.pre_out_host,
            config.pre_out_port,
            master_addr,
        )

    def _lwd_pre_out_loop(self) -> None:
        """PRE_OUT 接收循环(本线程独占 recv;socket 构造期建立后移交,
        与边侧同款模式)。recv 超时拍仅作关停响应上限,closed 退出。"""
        while True:
            msg = self._subscriber.recv(timeout_ms=LWD_PRE_OUT_RECV_TIMEOUT_MS)
            if msg is None:
                if self._subscriber.closed:
                    break
                continue
            self._lwd_dispatch(msg)

    def _lwd_dispatch(self, msg) -> None:
        """PRE_OUT 三类分派(本 IO 线程):元数据转 Request / abort 终结 /
        范围预告入队。"""
        if isinstance(msg, LwdRangeNotify):
            logger.info(
                "[Lwd][cloud-ctrl] RangeNotify req=%s num=%s seqno=%s",
                msg.request_id, msg.num_tokens, msg.seqno,
            )
            # 每条预告都整条入队(重复预告即重复点名,剔除-调度-拼回幂等,
            # 无副作用;PRE_OUT 只 append,调度主线程单独 popleft,deque
            # 单操作原子;预告自带 seqno,出批时作 UP 链配对号)
            self.scheduler.prefill_notify_queue.append(msg)
            return
        if isinstance(msg, LwdAbortNotify):
            logger.info("[Lwd][cloud-ctrl] AbortNotify req=%s", msg.request_id)
            self._lwd_gate_pending.pop(msg.request_id, None)
            # 双队列与原生 ABORT 同款:eager 处理 + 保持 input_queue 次序
            self.aborts_queue.put_nowait([msg.request_id])
            self.input_queue.put_nowait((EngineCoreRequestType.ABORT, [msg.request_id]))
            return
        rid = msg.request_id
        if rid in self._lwd_gate_pending:
            logger.warning("[Lwd] duplicate request metadata %s ignored", rid)
            return
        logger.info("[Lwd][cloud-ctrl] RequestNotify req=%s", rid)
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
        # eos_token_id 非构造入参,走原生回填入口(同边侧前端
        # input_processor):设 _eos_token_id 并计入 _all_stop_token_ids
        # 供 min_tokens 判定;云侧无客户端 generation_config,传空。
        sampling_params.update_from_generation_config({}, wire.eos_token_id)
        LwdDebug.cloud_request_admitted(wire, sampling_params)  # [lwd-debug]
        # 不传真实 ids 也不造占位:按原生 prompt-embeds 语义挂零缓冲,
        # 行数即 prompt 长度(UP chunk 注入直接写该缓冲的对应窗口);
        # ids=None 时 input_batch 自动把 prompt 段 is_token_ids 置 False,
        # M-RoPE 走纯文本直通构造(与扫描结果逐值一致)。
        prompt_ids: list[int] | None = None
        prompt_embeds = torch.zeros(
            wire.num_prompt_tokens,
            self.vllm_config.model_config.get_hidden_size(),
            dtype=self.vllm_config.model_config.dtype,
        )
        logger.info(
            "[Lwd][cloud-ctrl] build request req=%s prompt=%d ids=none+embeds_buf",
            wire.request_id, wire.num_prompt_tokens,
        )
        local_hasher = self.request_block_hasher
        if local_hasher is None:
            # prefix caching 未启用:请求不挂 hasher,整链机制不激活
            request = Request(
                request_id=wire.request_id,
                prompt_token_ids=prompt_ids,
                prompt_embeds=prompt_embeds,
                sampling_params=sampling_params,
                pooling_params=None,
            )
            # 占位 embeds 不随 SO 上传运输层(NewRequestData 只带形状,
            # worker 本地分配):35MB 零 buffer 走 MQ overflow 通道实测
            # 单程 240ms+,是 prefill SO 开工延迟的主因。
            request.lwd_embeds_placeholder = True
            return request
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

        request = Request(
            request_id=wire.request_id,
            prompt_token_ids=prompt_ids,
            prompt_embeds=prompt_embeds,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=block_hasher,
        )
        request.lwd_embeds_placeholder = True  # 同上:占位 embeds 不上 MQ
        return request
