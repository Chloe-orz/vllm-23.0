"""云侧 EngineCore 子类:控制面 ROUTER 收发归单线程 IO 循环,边侧预告
经 input_queue 走原生分发;仅 prefill_only 云角色启用。

连接模型:云侧 ROUTER bind 本 dp 的 ctrl_port(唯一 bind 方),边侧
DEALER connect 并以 register 帧注册;ack/步元数据(C2e)按 identity 从
同一 socket 定向回发。多边(fan-in)时按 identity 区分来源,请求键以
"{edge_id}#{dp_idx}#{rid}" 前缀隔离(边侧协议上的 req_id 保持透传不变,
C2e 回发时还原)。"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import torch
import zmq

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.engine import EngineCoreRequestType, FinishReason
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.lwd_control.control_communication.lwd_control_communicator import (
    LwdControlCommunicator,
)
from vllm.v1.lwd_control.control_communication.lwd_control_loop import (
    LwdControlLoop,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LWD_NOT_FINISHED,
    LWD_WIRE_VERSION,
    LwdAbortNotify,
    LwdC2eNotify,
    LwdRangeNotify,
    LwdRegisterAckNotify,
    LwdRegisterNotify,
    LwdRequestNotify,
    lwd_check_peer_topology,
    lwd_decode_notify,
    lwd_encode_cloud_notify,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import LwdConfig
from vllm.v1.lwd_debug import LwdDebug
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_phase_scheduler import (
    LwdCloudPhaseScheduler,
)
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.outputs import LwdC2eMeta, ModelRunnerOutput

logger = init_logger(__name__)

# 步元数据队满重试小睡:元数据不可丢(边侧据此预挂精确尺寸 recv)
_LWD_C2E_SEND_RETRY_SLEEP_S = 0.05


def _lwd_rid_key(edge_id: int, dp_idx: int, request_id: str) -> str:
    """云侧内部请求键:边/前缀隔离(两条边的裸 req_id 与 seqno 不再撞号),
    C2e 回发时按 '#' 反解还原边侧原始 id。"""
    return f"{edge_id}#{dp_idx}#{request_id}"


def _lwd_rid_split(rid_key: str) -> tuple[int, int, str]:
    """rid_key -> (edge_id, dp_idx, 原始 request_id);非法键即断言失败。"""
    edge_id, dp_idx, request_id = rid_key.split("#", 2)
    return int(edge_id), int(dp_idx), request_id


class LwdCloudEngineCore(EngineCoreProc):
    """云 PO 引擎:控制面 IO 循环接管 ROUTER 收发,其余全走原生。"""

    def __init__(self, *args, **kwargs) -> None:
        # 调度器自注入须赶在 super() 之前(与边侧 LwdEdgeEngineCore 同款):
        # super 构建 self.scheduler 时一次性消费 scheduler_cls,后设无效。
        # 依赖 lwd_serve_guard 注入不可靠——guard 只在 headless serve 入口
        # 执行,完整 serve 路径的 EngineCore 子进程不经 guard,缺注入会让
        # IO 线程把 RangeNotify 写进裸 AsyncScheduler 而崩溃。
        vllm_config = kwargs["vllm_config"]
        vllm_config.scheduler_config.scheduler_cls = LwdCloudPhaseScheduler
        super().__init__(*args, **kwargs)

    def _lwd_setup_control_plane(self) -> None:
        """介入控制面:ROUTER bind 本 dp 的 ctrl_port,单线程 IO 循环收发。
        建站失败走 EXECUTOR_FAILED 升级。"""
        config = LwdConfig.from_vllm_config(self.vllm_config)
        router = LwdControlCommunicator(
            config.router_bind_endpoint, zmq.ROUTER, bind=True
        )
        self._lwd_config = config
        self._lwd_io = LwdControlLoop(
            {None: router},
            routing=True,
            decoder=lwd_decode_notify,
            encoder=lwd_encode_cloud_notify,
            on_msg=self._lwd_on_edge_msg,
        )
        # 已注册对端:identity <-> (edge_id, dp_idx) 双向(fan-in 时一端口
        # 多连接,按 identity 区分来源与定向回发)
        self._lwd_peers: dict[bytes, tuple[int, int]] = {}
        self._lwd_peer_ids: dict[tuple[int, int], bytes] = {}
        # 门池:元数据查重与暂存,到达即构建放行;仅本 IO 线程独占
        self._lwd_gate_pending: dict[str, LwdRequestNotify] = {}
        # UP 链 seqno 登记(边→云→云 worker 的最后一跳,§9.12 接缝):
        # rid -> [chunk 序 seqno 列表],RangeNotify 到达即登记;调度器
        # 出 prefill 批时取快照挂 SO.lwd_up_seqnos 随批下发云 worker
        # (数据面 UP recv 配对键,与边侧 SO.lwd_batch.seqno 同源同值)。
        # registry 引用交付调度器(IO 线程登记 / 主循环读,dict 赋值原子)。
        self._lwd_seqno_registry: dict[str, list[int]] = {}
        self.scheduler.lwd_seqno_registry = self._lwd_seqno_registry
        self._lwd_io.start()
        logger.info(
            "[Lwd] cloud engine assembled: ROUTER bind %s, waiting for edges",
            config.router_bind_endpoint,
        )

    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ) -> None:
        """父线程照跑父类原版,控制面 ROUTER 循环独立成线程,两生产者共用
        input_queue。"""
        try:
            self._lwd_setup_control_plane()
        except Exception:
            logger.exception("[Lwd] cloud ROUTER setup failed")
            self.input_queue.put_nowait((EngineCoreRequestType.EXECUTOR_FAILED, b""))
        super().process_input_sockets(
            input_addresses, coord_input_address, identity, ready_event
        )

    def _lwd_on_edge_msg(self, _key, identity: bytes, msg) -> None:
        """控制面入站分发(IO 线程回调):register 注册/互校;已注册来源的
        三类预告按 rid 前缀键投递。"""
        if isinstance(msg, LwdRegisterNotify):
            self._lwd_handle_register(identity, msg)
            return
        peer = self._lwd_peers.get(identity)
        if peer is None:
            logger.warning(
                "[Lwd] drop frame from unregistered edge %r (%r); "
                "waiting for register",
                identity, type(msg),
            )
            return
        edge_id, dp_idx = peer
        # 信封/载荷交叉校验(§3.3.4):消息冗余携带的 edge_id/dp_idx 必须
        # 与 identity 注册值一致——不一致即 identity 配错或串线,fail-fast
        # 优于静默按错误归属处理
        msg_edge, msg_dp = getattr(msg, "edge_id", None), getattr(msg, "dp_idx", None)
        if (msg_edge is not None and msg_edge >= 0 and msg_edge != edge_id) or (
            msg_dp is not None and msg_dp >= 0 and msg_dp != dp_idx
        ):
            logger.error(
                "[Lwd] identity/payload mismatch from %r: registered "
                "(edge=%d, dp=%d) but frame claims (edge=%s, dp=%s); drop",
                identity, edge_id, dp_idx, msg_edge, msg_dp,
            )
            return
        if isinstance(msg, LwdRangeNotify):
            self._lwd_handle_range(edge_id, dp_idx, msg)
        elif isinstance(msg, LwdAbortNotify):
            self._lwd_handle_abort(edge_id, dp_idx, msg)
        elif isinstance(msg, LwdRequestNotify):
            self._lwd_handle_request(edge_id, dp_idx, msg)
        else:
            logger.warning("[Lwd] drop unexpected edge frame %r", type(msg))

    def _lwd_handle_register(self, identity: bytes, msg: LwdRegisterNotify) -> None:
        """注册入口:cloud_id 连对端口 + 公共互校(版本/卡数/digest);
        通过即登记对端并定向回 ack(队满丢弃可接受——边侧周期重发,幂等)。"""
        config = self._lwd_config
        link = (msg.edge_id, msg.cloud_id, msg.dp_idx)
        expected_identity = f"edge-{msg.edge_id}-{msg.dp_idx}".encode()
        if link not in config.my_links or identity != expected_identity:
            logger.error(
                "[Lwd] register rejected: identity=%r link=%s is not bound "
                "to this cloud control endpoint",
                identity, link,
            )
            return
        if msg.cloud_id != config.instance_id:
            logger.error(
                "[Lwd] register from edge %d targets cloud %d but reached "
                "cloud %d (check topology ctrl_port assignment)",
                msg.edge_id, msg.cloud_id, config.instance_id,
            )
            return
        error = lwd_check_peer_topology(
            exp_wire_version=LWD_WIRE_VERSION,
            exp_edge_npu_count=config.edge_npu_count,
            exp_cloud_npu_count=config.cloud_npu_count,
            exp_topology_digest=config.topology_digest,
            got_wire_version=msg.wire_version,
            got_edge_npu_count=msg.edge_npu_count,
            got_cloud_npu_count=msg.cloud_npu_count,
            got_topology_digest=msg.topology_digest,
        )
        if error is not None:
            logger.error("[Lwd] register rejected from %r: %s", identity, error)
            return
        self._lwd_peers[identity] = (msg.edge_id, msg.dp_idx)
        self._lwd_peer_ids[(msg.edge_id, msg.dp_idx)] = identity
        self._lwd_io.send(
            identity,
            LwdRegisterAckNotify(
                cloud_id=config.instance_id,
                dp_idx=msg.dp_idx,
                wire_version=LWD_WIRE_VERSION,
            ),
        )
        logger.info(
            "[Lwd][cloud-ctrl] edge registered: identity=%r edge=%d dp=%d",
            identity, msg.edge_id, msg.dp_idx,
        )

    def _lwd_handle_range(self, edge_id: int, dp_idx: int, msg: LwdRangeNotify) -> None:
        """范围预告:登记 UP 链 seqno(数据面配对键,§9.12)。幂等去重按
        "单调性"(seqno 不大于该请求已登记尾号即重复,边侧队满重试天然
        产生重复预告,重试复用同一号不产生新登记)。"""
        rid_key = _lwd_rid_key(edge_id, dp_idx, msg.request_id)
        seqnos = self._lwd_seqno_registry.setdefault(rid_key, [])
        if not seqnos or msg.seqno > seqnos[-1]:
            seqnos.append(msg.seqno)
        logger.info(
            "[Lwd][cloud-ctrl] RangeNotify req=%s num=%s seqno=%s edge=%d dp=%d",
            msg.request_id, msg.num_tokens, msg.seqno, edge_id, dp_idx,
        )
        # 入队即换内部键:调度器按 notify.request_id 点名 requests 表
        # (Request/registry 均为 rid_key 形态),LwdEmbedBatch.req_ids 同
        # 键下发 worker——云侧全链 rid_key,边侧协议保持原始 id
        inner = LwdRangeNotify(
            request_id=rid_key,
            offset=msg.offset,
            num_tokens=msg.num_tokens,
            seqno=msg.seqno,
            has_mrope=msg.has_mrope,
            edge_id=edge_id,
            dp_idx=dp_idx,
        )
        self.scheduler.lwd_cloud_enqueue_range(inner, edge_id, dp_idx)

    def _lwd_handle_abort(self, edge_id: int, dp_idx: int, msg: LwdAbortNotify) -> None:
        rid_key = _lwd_rid_key(edge_id, dp_idx, msg.request_id)
        logger.info(
            "[Lwd][cloud-ctrl] AbortNotify req=%s edge=%d dp=%d",
            msg.request_id, edge_id, dp_idx,
        )
        self._lwd_gate_pending.pop(rid_key, None)
        # 双队列与原生 ABORT 同款:eager 处理 + 保持 input_queue 次序
        self.aborts_queue.put_nowait([rid_key])
        self.input_queue.put_nowait((EngineCoreRequestType.ABORT, [rid_key]))

    def _lwd_handle_request(
        self, edge_id: int, dp_idx: int, msg: LwdRequestNotify
    ) -> None:
        rid_key = _lwd_rid_key(edge_id, dp_idx, msg.request_id)
        if rid_key in self._lwd_gate_pending:
            logger.warning("[Lwd] duplicate request metadata %s ignored", rid_key)
            return
        logger.info(
            "[Lwd][cloud-ctrl] RequestNotify req=%s edge=%d dp=%d",
            msg.request_id, edge_id, dp_idx,
        )
        self._lwd_gate_pending[rid_key] = msg
        self._lwd_promote(rid_key)

    def _lwd_promote(self, rid_key: str) -> None:
        """过门:门池取 wire,转 Request 投 input_queue 走原生 ADD 分发。"""
        wire = self._lwd_gate_pending.pop(rid_key, None)
        if wire is not None:
            request = self._lwd_build_request(rid_key, wire)
            self.input_queue.put_nowait((EngineCoreRequestType.ADD, (request, 0)))
            logger.info("[Lwd] cloud request %s admitted via gate", rid_key)

    def _lwd_build_request(self, rid_key: str, wire: LwdRequestNotify) -> Request:
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
            rid_key, wire.num_prompt_tokens,
        )
        local_hasher = self.request_block_hasher
        if local_hasher is None:
            # prefix caching 未启用:请求不挂 hasher,整链机制不激活
            request = Request(
                request_id=rid_key,
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
            request_id=rid_key,
            prompt_token_ids=prompt_ids,
            prompt_embeds=prompt_embeds,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=block_hasher,
        )
        request.lwd_embeds_placeholder = True  # 同上:占位 embeds 不上 MQ
        return request

    def step_with_batch_queue(self):
        """步骤执行时长打点(开始执行→执行结束;不含引擎空等)。"""
        _t0 = time.monotonic()
        out = super().step_with_batch_queue()
        logger.info(
            "[Lwd][perf] cloud-step exec=%.2fms",
            (time.monotonic() - _t0) * 1000,
        )
        return out

    def step(self):
        """同步步路径同款打点。"""
        _t0 = time.monotonic()
        out = super().step()
        logger.info(
            "[Lwd][perf] cloud-step exec=%.2fms",
            (time.monotonic() - _t0) * 1000,
        )
        return out

    def lwd_handle_model_output(
        self,
        model_output: ModelRunnerOutput,
        engine_core_outputs: dict[int, EngineCoreOutputs],
    ) -> ModelRunnerOutput:
        """rank-replay:解码 worker 主流末尾 pinned 物化的步 meta(就绪由
        响应入队处的 event synchronize 保证),组 c2e 通告经 ROUTER 先于
        hidden 发边;finish 码取自 engine_core_outputs。

        pinned 布局:[ranks(各调度段行)..., counts(accepted/请求)...,
        seg_lens(段长/请求)...];top_id_ths 按段长切,被拒行一并携带,
        边侧按 num_accepted 取有效前缀。fan-in 时请求按 rid 前缀切组,
        每组一条 notify 定向发回所属边。"""
        # [Lwd][perf] cloud-step dt 已由 exec 时长替代(见 step_with_batch_queue
        # 覆写):开始执行→执行结束,不含无请求的空等。
        carrier = getattr(model_output, "lwd_down_carrier", None)
        if carrier is not None:
            pinned, req_ids, hidden_numel, seqno, connection_key = carrier
            n_req = len(req_ids)
            vals = pinned.tolist()
            counts = vals[-2 * n_req : -n_req]
            seg_lens = vals[-n_req:]
            ranks_flat = vals[: -2 * n_req]
            top_id_ths: list[list[int]] = []
            off = 0
            for seg_len in seg_lens:
                top_id_ths.append(ranks_flat[off : off + seg_len])
                off += seg_len
            from vllm.v1.outputs import LwdC2eMeta

            meta = LwdC2eMeta(
                hidden_num_elements=hidden_numel,
                top_id_ths=top_id_ths,
                num_accepted_tokens=list(counts),
                req_ids=list(req_ids),
                down_seqno=seqno,
                connection_key=connection_key,
            )
            logger.info(
                "[Lwd][cloud-ctrl] publish c2e(rank-replay): reqs=%s "
                "rows=%d seqno=%d",
                meta.req_ids, off, seqno,
            )
            LwdDebug.cloud_step(self.scheduler, meta, engine_core_outputs)  # [lwd-debug]
            _t = time.monotonic()
            self._lwd_publish_c2e(
                meta, self._lwd_c2e_finish_reasons(meta, engine_core_outputs)
            )
            # [Lwd][perf] 云侧 LWD 税:finish 码推导 + ZMQ publish
            logger.info(
                "[Lwd][perf] publish reqs=%d dur=%.2fms",
                len(meta.req_ids), (time.monotonic() - _t) * 1000,
            )
        else:
            logger.info(
                "[Lwd][cloud-ctrl] handle_model_output: no down carrier this step"
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
        """步元数据发边:按 rid 前缀切组(fan-in 拆发;单边退化为一条),
        每组定向发所属边;组级重试(成功的组不再重发——重复 C2e 会让
        边侧对同一 down_seqno 二次派发 unembed),队满小睡,关停退出。"""
        pending: list[tuple[bytes, LwdC2eNotify]] = []
        groups = self._lwd_c2e_split(meta)
        key = meta.connection_key
        if (
            key not in self._lwd_config.my_links
            or set(groups) != {(key[0], key[2])}
        ):
            raise ValueError(
                "[LWD] C2e metadata must describe one matching DOWN packet; "
                "cloud reuse requires splitting hidden rows before sending"
            )
        for (edge_id, dp_idx), indexes in groups.items():
            identity = self._lwd_peer_ids.get((edge_id, dp_idx))
            if identity is None:
                # 对端未注册(重启窗口):本组丢弃,由边侧超时兜底;
                # 不对未知 identity 重试(ROUTER_MANDATORY 会拒发)
                logger.warning(
                    "[Lwd] no registered edge for (%d,%d); drop c2e group",
                    edge_id, dp_idx,
                )
                continue
            notify = self._lwd_c2e_build(meta, finish_reasons,
                                         indexes, edge_id, dp_idx)
            logger.info(
                "[Lwd][cloud-ctrl] publish C2eNotify->%r reqs=%d "
                "down_seqno=%s hidden_elems=%s",
                identity, len(notify.req_ids),
                notify.down_seqno, notify.hidden_num_elements,
            )
            pending.append((identity, notify))
        while pending and not self._lwd_io.closed:
            pending = [
                (identity, notify)
                for identity, notify in pending
                if not self._lwd_io.send(identity, notify)
            ]
            if pending:
                time.sleep(_LWD_C2E_SEND_RETRY_SLEEP_S)
        for identity, notify in pending or ():
            logger.error("[Lwd] c2e group dropped on shutdown: %r", identity)

    @staticmethod
    def _lwd_c2e_split(meta: LwdC2eMeta) -> dict[tuple[int, int], list[int]]:
        """meta 行按 rid 前缀切组:组键 (edge_id, dp_idx),值为行下标。"""
        groups: dict[tuple[int, int], list[int]] = {}
        for index, rid_key in enumerate(meta.req_ids):
            edge_id, dp_idx, _rid = _lwd_rid_split(rid_key)
            groups.setdefault((edge_id, dp_idx), []).append(index)
        return groups

    @staticmethod
    def _lwd_c2e_build(
        meta: LwdC2eMeta,
        finish_reasons: list[int],
        indexes: list[int],
        edge_id: int,
        dp_idx: int,
    ) -> LwdC2eNotify:
        """切出行子集构造定向 notify;rid 还原为边侧原始 id。"""
        return LwdC2eNotify(
            hidden_num_elements=meta.hidden_num_elements,
            top_id_ths=[meta.top_id_ths[i] for i in indexes],
            num_accepted_tokens=[meta.num_accepted_tokens[i] for i in indexes],
            req_ids=[_lwd_rid_split(meta.req_ids[i])[2] for i in indexes],
            finish_reasons=[finish_reasons[i] for i in indexes],
            down_seqno=meta.down_seqno,
            edge_id=edge_id,
            dp_idx=dp_idx,
        )

    def shutdown(self) -> None:
        """控制面关停后走原生(幂等;装配失败路径可能未建,容忍缺省)。"""
        io_loop = getattr(self, "_lwd_io", None)
        if io_loop is not None:
            io_loop.stop()
        super().shutdown()
