"""云侧执行类:源 ActiveEdgeCloudEngineCore 的 EngineCore 强耦合残部(§10.11)。

控制面功能接口(首预告门/三类通知处理/消费水位/统计)已二次收编进相位
调度器(§10.11);本文件只留与 EngineCore 模块强耦合的部分:PRE_OUT 泵、
数据面接缝(seqno 登记/hint/drop)、步体与驱动。结构基线 = 源类全方法
照搬(§10.8),收敛裁剪逐条见差异清单。

照搬差异清单(逐条,改造时先读):
  1. 消息名映射:EdgeEmbedChunkNotify/EdgePrefillAbort/EngineCoreRequest →
     LwdRangeNotify/LwdAbortNotify/LwdRequestNotify;
  2. 源 chunk_idx 无对应字段:首预告门判 offset == 0(门状态已入调度器,
     泵只转发就绪信号);seqno 登记表为 rid -> [seqno 有序表](源为
     chunk_idx 槽位表);
  3. 源 __init__ 的 step_fn MethodType 绑定不搬:守卫入口已驱动增强步体,
     再绑会让原生外层与 wrapper 各走一遍 = 双步进;
  4. 步体调用点:源 self.engine_core.step_fn() →
     lwd_native_step_bq_prefill_only(self.engine_core)(step_fn 若走
     原生绑定会经守卫递归回 wrapper);
  5. POST_OUT/结果发布器:results 经构造可选注入(§9.1 裁,缺省 None);
     None 时发布段整体短路 —— 源方法体保留,传输回接后即生效;
  6. 消费水位推导移入调度器(lwd_cloud_publish_consumed_watermarks,
     §10.11),本类不再持有 _consumed_sent;
  7. fast path 不接:subscriber 组合化后无 set_fast_handler,源绑定段与
     _try_fast_forward 惰性方法删除(§10.11);
  8. 请求构建经调度器 request factory(装配层绑定,§10.11);准入纪律
     与首预告门均在调度器(§10.10);
  9. post_step 与 GIL 让出 sleep 不搬:外层原生 _process_engine_step
     (core.py:1267)承担,本类嵌套其 step_fn 委托链内,双份有害;
 10. run_busy_loop 照搬但当前不启用(驱动来自原生 EngineCoreProc 循环;
     自驱启用时需自补 output_queue 发布与 post_step,见差异 5/9);
 11. 函数超 50 行为照搬暂态,改造收敛时按标准拆分(check_functions 暂豁免)。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LwdAbortNotify,
    LwdRangeNotify,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_step_core import LwdStepCore

if TYPE_CHECKING:
    from vllm.v1.lwd_control.control_communication.lwd_control_subscriber import (
        LwdControlSubscriber,
    )
    from vllm.v1.lwd_control.control_communication.lwd_notify import (
        EngineCoreOutputs,
    )
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_step_core import (
        LwdEnginePort,
        LwdStepSettings,
    )

logger = init_logger(__name__)


class LwdCloudCore(LwdStepCore):
    """云侧 prefill_only core:EngineCore 强耦合残部,PRE_OUT 泵为控制面入口。"""

    def __init__(
        self,
        subscriber: LwdControlSubscriber,
        engine_port: LwdEnginePort,
        settings: LwdStepSettings,
        results: object | None = None,
    ) -> None:
        self._lwd_subscriber = subscriber
        self._settings = settings
        self.engine_core = engine_port.lwd_engine_core()
        self._results = results  # §9.1:缺省无发布器,发布段短路(差异 5)
        # 数据面接缝状态(接收登记,§10.1);门/水位状态已入调度器(差异 6/8)
        self.engine_core._po_chunk_seqnos: dict[str, list[int]] = {}
        self._step_index = 0
        self._idle_sleep_seconds = 0.001
        self._zombie_log_ts = 0.0
        self._hint_mq = getattr(
            self.engine_core.model_executor, "cloud_recv_hint_mq", None
        )
        self._hint_missing_warned = False

    # ------------------------------------------------------------------ #
    # 控制面(PRE_OUT 泵;门/准入/请求构建均在调度器,§10.10/§10.11)     #
    # ------------------------------------------------------------------ #
    def _record_chunk_notify(self, notify: LwdRangeNotify) -> None:
        """登记 seqno(rid -> 有序表;数据面接收登记接缝,§10.1)。"""
        registry = self.engine_core._po_chunk_seqnos
        seqnos = registry.get(notify.request_id)
        if seqnos is None:
            seqnos = []
            registry[notify.request_id] = seqnos
        seqnos.append(notify.seqno)

    def _drain_control_plane(self) -> None:
        scheduler = self.engine_core.scheduler
        for msg in self._lwd_subscriber.drain():
            if isinstance(msg, LwdRangeNotify):
                self._record_chunk_notify(msg)
                self._forward_chunk_hint(msg)
                if msg.offset == 0:
                    # 首预告门判定属线上语义(offset==0 = 派发首块),留泵侧
                    scheduler.lwd_cloud_on_range_notify(msg.request_id)
            elif isinstance(msg, LwdAbortNotify):
                self._handle_abort(msg.request_id)
            else:
                scheduler.lwd_cloud_on_request_notify(msg)

    def _handle_abort(self, request_id: str) -> None:
        """abort:门/暂存/已准入清理由调度器接口承担,此处只留数据面接缝。"""
        self.engine_core.scheduler.lwd_cloud_on_abort_notify(request_id)
        self.engine_core._po_chunk_seqnos.pop(request_id, None)
        self._forward_drop(request_id)

    def _forward_chunk_hint(self, notify: LwdRangeNotify) -> None:
        if self._hint_mq is None:
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
        self._hint_mq.enqueue((b"irecv_hint", (hint,), {}, None))

    def _forward_drop(self, request_id: str) -> None:
        if self._hint_mq is None:
            return
        self._hint_mq.enqueue((b"prefill_only_drop", (request_id,), {}, None))

    # ------------------------------------------------------------------ #
    # 引擎步进(EngineCore 强耦合:调度/批队列/执行器触达)                #
    # ------------------------------------------------------------------ #
    def _has_work(self) -> bool:
        return self.engine_core.scheduler.has_requests() or bool(
            self.engine_core.batch_queue
        )

    def _process_engine_step(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        super()._process_engine_step()

    def run_busy_loop(self) -> None:
        """源自有循环,当前不启用(差异 10);自驱启用需自补差异 5/9 两段。"""
        logger.info("[Lwd] cloud busy loop starting")
        while True:
            self._drain_control_plane()
            if self._has_work():
                self._process_engine_step()
            else:
                time.sleep(self._idle_sleep_seconds)


def lwd_native_step_bq_prefill_only(engine_core) -> tuple[dict | None, bool]:
    """源步内路径(裁剪版):调度 → 执行 → 采样的异步流水线。

    = 原生 step_with_batch_queue(core.py:491)逐字 + 空批预完成 future
    (增强 1)+ [PO-RPC] 预检;增强 2/3 与 POST_OUT/水位已裁。
    """
    from concurrent.futures import Future
    from typing import cast

    batch_queue = engine_core.batch_queue
    assert batch_queue is not None
    assert len(batch_queue) < engine_core.batch_queue_size

    model_executed = False
    deferred_scheduler_output = None
    if engine_core.scheduler.has_requests():
        scheduler_output = engine_core.scheduler.schedule()
        # [Lwd-STEP] 相位形状:P=prefill / D=decode / E=empty
        _n_reqs = len(scheduler_output.num_scheduled_tokens)
        _n_toks = scheduler_output.total_num_scheduled_tokens
        logger.info(
            "[Lwd-STEP] schedule phase=%s reqs=%d tokens=%d queue_len=%d",
            "P" if _n_toks > _n_reqs else ("D" if _n_toks > 0 else "E"),
            _n_reqs,
            _n_toks,
            len(batch_queue),
        )
        # 增强 1(空批):0-token 批不派 worker,用预完成 future 占位,
        # 批仍入队 —— 否则 batch queue 冻结,末请求 finish 帧永不执行
        if _n_toks == 0:
            from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

            exec_future = Future()
            exec_future.set_result(EMPTY_MODEL_RUNNER_OUTPUT)
        if _n_toks > 0:
            with engine_core.log_error_detail(scheduler_output):
                exec_future = engine_core.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
        if engine_core.is_ec_consumer:
            model_executed = scheduler_output.total_num_scheduled_tokens > 0

        if engine_core.is_pooling_model or not model_executed:
            future = cast(Future, exec_future)
        elif not scheduler_output.pending_structured_output_tokens:
            grammar_output = engine_core.scheduler.get_grammar_bitmask(scheduler_output)
            future = engine_core.model_executor.sample_tokens(
                grammar_output, non_block=True
            )
        else:
            deferred_scheduler_output = scheduler_output

        if not deferred_scheduler_output:
            batch_queue.appendleft((future, scheduler_output, exec_future))
            if len(batch_queue) < engine_core.batch_queue_size and (
                model_executed or engine_core.scheduler.has_requests()
            ):
                return None, model_executed
    elif not batch_queue:
        return None, False

    future, scheduler_output, exec_model_fut = batch_queue.pop()
    with (
        engine_core.log_error_detail(scheduler_output),
        engine_core.log_iteration_details(scheduler_output),
    ):
        model_output = future.result()
        if model_output is None:
            exec_model_fut.result()
            raise RuntimeError("unexpected error")

    engine_core._process_aborts_queue()
    # [PO-RPC] 批/输出一致性预检:错位在此显式失败,而非 update_from_output
    # 深处的 KeyError
    _so_reqs = list(scheduler_output.num_scheduled_tokens.keys())
    _out_reqs = list(model_output.req_ids or ())
    _missing = [r for r in _so_reqs if r not in set(_out_reqs)]
    if _missing:
        raise RuntimeError(
            "[Lwd] batch/output mismatch before update_from_output: "
            f"scheduler_output has {_so_reqs}, model_output has {_out_reqs} "
            f"(missing {_missing})"
        )
    engine_core_outputs = engine_core.scheduler.update_from_output(
        scheduler_output, model_output
    )

    if deferred_scheduler_output:
        if engine_core.use_spec_decode:
            draft_token_ids = engine_core.model_executor.take_draft_token_ids()
            assert draft_token_ids is not None
            engine_core.scheduler.update_draft_token_ids_in_output(
                draft_token_ids, deferred_scheduler_output
            )
        grammar_output = engine_core.scheduler.get_grammar_bitmask(
            deferred_scheduler_output
        )
        future = engine_core.model_executor.sample_tokens(
            grammar_output, non_block=True
        )
        batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

    return engine_core_outputs, model_executed
