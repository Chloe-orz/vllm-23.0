"""云侧 EngineCore 子类:构造期建 ZMQ 双面并起 PRE_OUT 接收线程,边侧
预告与步内元数据经 input_queue 走原生分发;仅 prefill_only 云角色启用。"""

from __future__ import annotations


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

class LwdCloudEngineCore(LwdBaseEngineCore):
    """云 PO 引擎:构造期建 ZMQ 双面 + 起 PRE_OUT 接收线程,其余全走原生。"""

    lwd_scheduler_cls = LwdCloudScheduler

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lwd_setup_planes()
        # 首拍即通告(边侧可能已 bind 等待);队满不重试,由边侧等待
        # 超时 fail-fast 兜底
        config = self.lwd_config
        self._publisher.publish(LwdHelloNotify(
            pre_out_host=config.pre_out_host, pre_out_port=config.pre_out_port
        ))
        logger.info(
            "[Lwd][cloud] HELLO announced: pre_out=%s:%s",
            config.pre_out_host, config.pre_out_port,
        )
        master_addr = self.vllm_config.parallel_config.master_addr
        logger.info(
            "[Lwd] cloud engine assembled: PRE_OUT bind %s, POST_OUT announce -> "
            "%s:%s via master %s",
            f"tcp://{config.pre_out_host}:{config.pre_out_port}",
            config.pre_out_host,
            config.pre_out_port,
            master_addr,
        )

    def _lwd_build_subscriber(self) -> LwdControlSubscriber:
        """PRE_OUT bind 收边。"""
        config = self.lwd_config
        return LwdControlSubscriber(
            f"tcp://{config.pre_out_host}:{config.pre_out_port}", bind=True
        )

    def _lwd_build_publisher(self) -> LwdControlPublisher:
        """POST_OUT connect 边(master_addr),承载 HELLO 与步元数据。"""
        config = self.lwd_config
        master_addr = self.vllm_config.parallel_config.master_addr
        return LwdControlPublisher(
            f"tcp://{master_addr}:{config.post_out_port}",
            bind=False,
            encoder=lwd_encode_cloud_notify,
        )

    def _lwd_on_message(self, msg) -> None:
        """PRE_OUT 三类分派:范围预告入调度器队列 / abort 终结 /
        元数据即时建 Request 投 ADD(重复元数据不会到达:发布重试仅
        发生在未入队时)。"""
        if isinstance(msg, LwdRangeNotify):
            logger.info(
                "[Lwd][cloud-ctrl] RangeNotify req=%s num=%s seqno=%s",
                msg.request_id, msg.num_tokens, msg.seqno,
            )
            # 整条入队,一步一条点名;预告自带 seqno 即 UP 链配对号
            self.scheduler.prefill_notify_queue.append(msg)
            return
        if isinstance(msg, LwdAbortNotify):
            logger.info("[Lwd][cloud-ctrl] AbortNotify req=%s", msg.request_id)
            # 双队列与原生 ABORT 同款:eager 处理 + 保持 input_queue 次序
            self.aborts_queue.put_nowait([msg.request_id])
            self.input_queue.put_nowait((EngineCoreRequestType.ABORT, [msg.request_id]))
            return
        logger.info("[Lwd][cloud-ctrl] RequestNotify req=%s", msg.request_id)
        request = self._lwd_build_request(msg)
        self.input_queue.put_nowait((EngineCoreRequestType.ADD, (request, 0)))

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
