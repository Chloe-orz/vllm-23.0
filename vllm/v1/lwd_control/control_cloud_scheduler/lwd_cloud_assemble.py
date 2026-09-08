"""云侧 L3 装配:空批契约垫片 + PRE_OUT 桥线程 + 请求构建(§10.12)。

EngineCore 的构建仍走原生 headless 路径(serve.py 守卫负责),本文件在
真实 EngineCore 建成后完成 Lwd 装配;step_wrapper 不再使用 —— 步体走
原生(垫片恢复空批契约),泵由桥线程承担,调度语义全在相位调度器。
LwdConfig/模式判定复用 lwd_edge_assemble;违禁 import 只允许本文件
(§7.2:vllm.v1.engine 请求元组类型 / v1.outputs 契约值 / Request 构建)。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils.system_utils import set_process_title
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_phase_scheduler import (
    LwdCloudPhaseScheduler,
)
from vllm.v1.lwd_control.control_communication.lwd_control_subscriber import (
    LwdControlSubscriber,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LwdAbortNotify,
    LwdRangeNotify,
    LwdRequestNotify,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
    LwdConfig,
    is_lwd_prefill_only,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.request import Request

logger = init_logger(__name__)


class LwdCloudBridge:
    """PRE_OUT 桥线程(§10.12,方案 §2.4 桥 A;POST_OUT 桥随 §9.1 裁)。

    两线程分工:桥线程独占门状态变更与数据面接缝(seqno/hint/drop);
    调度状态只经 input_queue 的 ADD/ABORT 原生分发在循环线程变更 ——
    跨线程直改调度器内部即竞态,零锁的代价是这条纪律。
    """

    _LWD_IDLE_SLEEP_SECONDS = 0.001

    def __init__(
        self,
        subscriber: LwdControlSubscriber,
        scheduler: LwdCloudPhaseScheduler,
        engine_core,
    ) -> None:
        self._subscriber = subscriber
        self._scheduler = scheduler
        self._engine_core = engine_core
        self._hint_missing_warned = False
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._pump, name="lwd-cloud-bridge", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """幂等停桥:停泵后关停订阅通道(S1 关停幂等)。"""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._subscriber.shutdown()

    def _pump(self) -> None:
        while not self._stop_event.is_set():
            for msg in self._subscriber.drain():
                try:
                    self._dispatch(msg)
                except Exception:
                    # 泵线程不允许带病退出:单条翻译失败记日志丢消息,
                    # 上层超时/abort 路径兜底(源 zombie 观测同语义)。
                    logger.exception("[Lwd] cloud bridge dispatch failed")
            self._stop_event.wait(self._LWD_IDLE_SLEEP_SECONDS)

    def _dispatch(self, msg) -> None:
        if isinstance(msg, LwdRangeNotify):
            self._on_range_notify(msg)
        elif isinstance(msg, LwdAbortNotify):
            self._on_abort_notify(msg.request_id)
        else:
            self._scheduler.lwd_cloud_on_request_notify(msg)

    def _on_range_notify(self, notify: LwdRangeNotify) -> None:
        # 数据面接缝(§10.1):seqno 接收登记 + hint 转发
        registry = self._engine_core._po_chunk_seqnos
        seqnos = registry.get(notify.request_id)
        if seqnos is None:
            seqnos = []
            registry[notify.request_id] = seqnos
        seqnos.append(notify.seqno)
        self._forward_hint(notify)
        if notify.offset == 0:
            # 首预告门判定属线上语义(offset==0 = 派发首块),留桥侧
            self._scheduler.lwd_cloud_on_range_notify(notify.request_id)

    def _on_abort_notify(self, request_id: str) -> None:
        self._scheduler.lwd_cloud_on_abort_notify(request_id)
        self._engine_core._po_chunk_seqnos.pop(request_id, None)
        self._forward_drop(request_id)

    def _forward_hint(self, notify: LwdRangeNotify) -> None:
        hint_mq = getattr(self._engine_core.model_executor, "cloud_recv_hint_mq", None)
        if hint_mq is None:
            if not self._hint_missing_warned:
                logger.warning(
                    "[Lwd] chunk notifications arriving but cloud_recv_hint_mq "
                    "is not wired (worker recv manager pending)"
                )
                self._hint_missing_warned = True
            return
        hint = {
            "prefill_only": True,
            "request_id": notify.request_id,
            "seqno": notify.seqno,
            "num_tokens": notify.num_tokens,
        }
        hint_mq.enqueue((b"irecv_hint", (hint,), {}, None))

    def _forward_drop(self, request_id: str) -> None:
        hint_mq = getattr(self._engine_core.model_executor, "cloud_recv_hint_mq", None)
        if hint_mq is not None:
            hint_mq.enqueue((b"prefill_only_drop", (request_id,), {}, None))


def lwd_cloud_install_empty_batch_contract(engine_core) -> None:
    """空批契约垫片:fork runner 对 0-token 批回 None,原生步体
    future.result() 即崩(core.py:576)。0-token 派发在执行器接缝处
    短路为预完成 EMPTY_MODEL_RUNNER_OUTPUT(上游契约值;原步体增强 1
    的下沉形态,附带省一次 worker 往返)—— runner 侧契约修复落地后
    可整体拆除,非 PO 形态不安装。
    """
    executor = engine_core.model_executor
    original_execute = executor.execute_model

    def execute_model(scheduler_output, *args, **kwargs):
        if scheduler_output.total_num_scheduled_tokens > 0:
            return original_execute(scheduler_output, *args, **kwargs)
        future: Future = Future()
        future.set_result(EMPTY_MODEL_RUNNER_OUTPUT)
        return future

    executor.execute_model = execute_model


def lwd_cloud_try_assemble(engine_core) -> bool:
    """云侧装配点(幂等):非 PO / 边角色立即返回 False,零副作用。

    装配三段:空批契约垫片安装 → PRE_OUT 订阅通道与桥线程 → 请求
    工厂/出口绑定调度器(§10.12)。step_wrapper 不再使用,core.py
    的关停经原生 scheduler.shutdown 钩子转发停桥。
    """
    # 先角色判定再触达调度器:非 PO 引擎是原生调度器,无 Lwd 接口
    if not is_lwd_prefill_only(engine_core.vllm_config):
        return False
    config = LwdConfig.from_env_and_config(engine_core.vllm_config)
    if config.is_edge_node:
        return False
    scheduler = engine_core.scheduler
    if scheduler.lwd_cloud_control_plane_bound():
        return True
    if engine_core.batch_queue is None:
        # 云侧按异步流水线部署验证(源装配断言同款:requires the batch
        # queue, max_concurrent_batches > 1),缺位即部署错误显式失败
        raise RuntimeError(
            "[Lwd] cloud prefill_only requires the batch queue "
            "(async_scheduling / max_concurrent_batches > 1)"
        )
    # 角色标记(原 lwd_cloud_main 进程初始化段随删除折叠至此)
    set_process_title("vllm::EngineCore::LwdCloud")
    lwd_cloud_install_empty_batch_contract(engine_core)
    subscriber = _lwd_cloud_connect_planes(config)
    bridge = LwdCloudBridge(subscriber, scheduler, engine_core)
    scheduler.lwd_cloud_bind_request_factory(_lwd_cloud_build_request(engine_core))
    scheduler.lwd_cloud_bind_bridge(
        admit_sink=_lwd_cloud_admit_sink(engine_core),
        abort_sink=_lwd_cloud_abort_sink(engine_core),
        stop=bridge.stop,
    )
    bridge.start()
    logger.info("[Lwd] cloud assembled: native step + bridge pump (no step_wrapper)")
    return True


def _lwd_cloud_connect_planes(config: LwdConfig) -> LwdControlSubscriber:
    """建控制面订阅通道(单向,仅 PRE_OUT;无结果面,§9.1)。

    bind/connect 是装配期 wiring(传输层 side-agnostic):云侧 bind,
    边侧 connect 于同一端点。
    """
    return LwdControlSubscriber(config.lwd_pre_out_endpoint(), bind=True)


def _lwd_cloud_build_request(engine_core) -> Callable[[LwdRequestNotify], Request]:
    """唯一请求构建点:Request 构建/block_hasher 全收于此(§7.3-C1)。

    哈希链获取(§10.13):云 prompt token 是占位零值,本地算不出真实
    链 —— prompt 首建直接用边侧预告里的 block_hashes,decode 续算回
    本地 hasher;边侧未提供或长度不符回退本地(占位链,命中无效但不
    崩)。数据面挂载(prompt_embeds 视图)由数据面落位侧在此对接
    (§9.12);产物经提升出口进 input_queue,循环线程原生 ADD 分发到
    调度器暂存(§10.10/§10.12)。
    """
    local_hasher = engine_core.request_block_hasher
    if local_hasher is not None:
        # 与 core.py 构造期同函数同输入,结果确定一致
        _, hash_block_size = resolve_kv_cache_block_sizes(
            engine_core.scheduler.kv_cache_config, engine_core.vllm_config
        )
    else:
        hash_block_size = 0

    def _build(wire: LwdRequestNotify) -> Request:
        return Request(
            request_id=wire.request_id,
            # 占位 token:云侧调度只看长度,真值由边侧 embeds 经数据面提供(§9.5)
            prompt_token_ids=[0] * wire.num_prompt_tokens,
            sampling_params=SamplingParams(max_tokens=wire.max_tokens),
            pooling_params=None,
            block_hasher=(
                _lwd_cloud_wire_hasher(wire, local_hasher, hash_block_size)
                if local_hasher is not None
                else None
            ),
        )

    return _build


def _lwd_cloud_wire_hasher(
    wire: LwdRequestNotify, local_hasher: Callable, hash_block_size: int
) -> Callable[[Request], list[bytes]]:
    """wire 链优先的请求 hasher:prompt 首建用边侧预告链,decode 续算
    回本地 hasher;边侧未提供或长度不符回退本地(fail-open,§10.13)。"""

    def hasher(request: Request) -> list[bytes]:
        if len(request.block_hashes) == 0 and request.num_output_tokens == 0:
            expected = request.num_prompt_tokens // hash_block_size
            if len(wire.block_hashes) == expected:
                return wire.block_hashes
        return local_hasher(request)

    return hasher


def _lwd_cloud_admit_sink(engine_core) -> Callable[[Request], None]:
    """提升出口:过门请求 marshal 到循环线程(input_queue 原生 ADD 分发)。"""
    input_queue = engine_core.input_queue

    def _admit(request: Request) -> None:
        input_queue.put_nowait((EngineCoreRequestType.ADD, (request, 0)))

    return _admit


def _lwd_cloud_abort_sink(engine_core) -> Callable[[str], None]:
    """abort 出口:终结 marshal 到循环线程(原生 ABORT 分发 → finish_requests)。"""
    input_queue = engine_core.input_queue

    def _abort(request_id: str) -> None:
        input_queue.put_nowait((EngineCoreRequestType.ABORT, [request_id]))

    return _abort
