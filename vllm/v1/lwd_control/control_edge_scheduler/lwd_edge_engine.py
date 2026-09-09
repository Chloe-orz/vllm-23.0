"""边侧 L3 引擎子类:类选择点注入的边 EngineCore(对齐云侧 §10.14,替代
已删的装配层 lwd_edge_try_assemble/step_wrapper/LwdEnginePort)。

接入点 = run_engine_core 类选择点(lwd_resolve_engine_cls 边角色分支):
引擎出生即 LwdEdgeEngineCore,通信面(POST_OUT bind/发现线程/HELLO 等待)
与调度器注入(LwdEdgeScheduler)全部收进子类 __init__,引擎属性零外挂。

装配时序(§9.1/§9.10):
  ① kv_transfer_config 预检:在场则降级原生(不建任何 lwd 状态,覆写
     方法按 _lwd_active=False 走 super(),语义同旧装配降级)。
  ② 通信面先建:bind POST_OUT、延迟连接的 PRE_OUT publisher、结果队列、
     lwd-post-in 发现线程,阻塞等首条 HELLO(hello_timeout_s fail-fast,
     失败自清理两面后抛)。
  ③ scheduler_cls 注入裸类后 super().__init__():调度器出生即
     LwdEdgeScheduler(零整实例重建);publisher 构造后回填(装配期完成,
     早于任何请求,等价构造注入——get_scheduler_cls 无法携带实参)。

步进编排(原 LwdEdgeCore 迁入):step/step_with_batch_queue 覆写 =
调度器 schedule(单请求组批) -> lwd_edge_notify 发预告(队满整步回退)
-> 原生 executor 同步执行 -> 步末对账;输出恒 (None, 是否有工作),
云结果回传占用返回值首位待后续 Step 启用。
"""

from __future__ import annotations

import queue
import threading

from vllm.logger import init_logger
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequestType,
    FinishReason,
)
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
    LwdControlPublisher,
)
from vllm.v1.lwd_control.control_communication.lwd_control_subscriber import (
    LwdControlSubscriber,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LwdHelloNotify,
    LwdResultNotify,
    lwd_decode_cloud_notify,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
    LWD_RESULT_QUEUE_MAX,
    LwdConfig,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
    lwd_build_unembed_batch,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_step_core import LwdLog

logger = init_logger(__name__)


class LwdEdgeEngineCore(EngineCoreProc):
    """边 PO 引擎:通信面装配 + 调度器注入 + step/add/abort/shutdown 覆写。"""

    def __init__(self, *args, **kwargs) -> None:
        self._lwd_active = False
        vllm_config = args[0]
        config = LwdConfig.from_env_and_config(vllm_config)
        self._lwd_config = config
        self._lwd_log = LwdLog(config.debug)
        if vllm_config.kv_transfer_config is not None:
            # 调度器重建会丢失 connector 握手态:PO 不支持 kv_connector,
            # 降级原生(旧装配点同款语义,预检 = connector 创建条件)
            logger.warning(
                "[Lwd] kv_connector enabled on edge: degrade to native engine"
            )
            super().__init__(*args, **kwargs)
            return

        # 通信面(§9.1):bind POST_OUT 订阅面 + 延迟连接的 PRE_OUT 发布面,
        # 发现线程消费 POST_OUT,HELLO -> retarget PRE_OUT(云端点唯一事实源)
        self._lwd_post_out_receiver = self._lwd_build_post_out(config)
        self._lwd_publisher = LwdControlPublisher(
            None, bind=False, queue_max=config.publish_queue_max
        )
        self.lwd_result_queue = queue.Queue(maxsize=LWD_RESULT_QUEUE_MAX)
        hello_event = threading.Event()
        discovery = threading.Thread(
            target=self._lwd_discovery_loop,
            args=(hello_event,),
            name="lwd-post-in",
            daemon=True,
        )
        discovery.start()
        if not hello_event.wait(config.hello_timeout_s):
            self._lwd_shutdown_planes()
            raise RuntimeError(
                f"[Lwd] edge engine init failed: no cloud HELLO within "
                f"{config.hello_timeout_s}s on POST_OUT "
                f"(bind {config.lwd_post_out_bind_endpoint()}; check cloud "
                f"master_addr connectivity and POST_OUT port)"
            )

        # 调度器注入(§9.10):裸类注入出生即 LwdEdgeScheduler,publisher
        # 于引擎构造完成后回填(构造期完成,早于任何请求,等价构造注入)
        vllm_config.scheduler_config.scheduler_cls = LwdEdgeScheduler
        self._lwd_active = True
        super().__init__(*args, **kwargs)
        self.scheduler.lwd_edge_publisher = self._lwd_publisher
        logger.info(
            "[Lwd] edge engine assembled: POST_OUT bind %s, PRE_OUT discovered",
            config.lwd_post_out_bind_endpoint(),
        )

    # ------------------------------------------------------------------ #
    # 装配支撑(自旧装配文件迁入,逻辑不变)                                #
    # ------------------------------------------------------------------ #
    def _lwd_build_post_out(self, config: LwdConfig) -> LwdControlSubscriber:
        """bind POST_OUT 订阅面(云经 master_addr 主动来连;边不预知云地址)。"""
        return LwdControlSubscriber(
            config.lwd_post_out_bind_endpoint(),
            bind=True,
            decoder=lwd_decode_cloud_notify,
        )

    def _lwd_discovery_loop(self, hello_event: threading.Event) -> None:
        """发现线程(lwd-post-in):消费 POST_OUT,按类型分发。

        HELLO -> retarget PRE_OUT(云端点唯一事实源,§9.1):常驻运行,
        云换址重启后周期 HELLO 仍能驱动先连新断旧;retarget 队满失败
        靠周期重发自愈。
        LwdResultNotify -> 结果队列(不丢)+ WAKEUP 唤醒主循环:引擎可能
        阻塞在 input_queue.get()(prefill 全部完成后 awaiting 无排程
        工作),结果只进 lwd_result_queue 不会唤醒任何线程,必须敲门;
        WAKEUP 分支原生即丢弃消息体,数据与唤醒分离,多投无害(空
        drain 一步即返回)。
        """
        receiver = self._lwd_post_out_receiver
        publisher = self._lwd_publisher
        while not receiver.closed:
            msg = receiver.recv(timeout_ms=5000)
            if msg is None:
                continue
            if isinstance(msg, LwdHelloNotify):
                endpoint = f"tcp://{msg.pre_out_host}:{msg.pre_out_port}"
                if not hello_event.is_set():
                    logger.info(
                        "[Lwd] cloud discovered via HELLO: PRE_OUT -> %s", endpoint
                    )
                if not publisher.retarget(endpoint):
                    # 队满丢令:周期重发(5s)会再来,下条 HELLO 重试
                    logger.warning(
                        "[Lwd] PRE_OUT retarget deferred: publish queue full"
                    )
                hello_event.set()
            elif isinstance(msg, LwdResultNotify):
                self._lwd_enqueue_result(msg)
            else:
                # 协议分面:POST_OUT 只承载 HELLO 与结果,坏帧已被订阅层丢弃
                logger.warning("[Lwd] drop unexpected POST_OUT frame %r", type(msg))

    def _lwd_enqueue_result(self, msg: LwdResultNotify) -> None:
        """云结果入队(唯一数据通道,不丢)+ WAKEUP 唤醒主循环。

        队满自旋重试(结果不可失,与 publisher 背压可丢语义相反):
        消费侧每步取空,积压只来自引擎长步;重试间隔与接收线程 5s
        超时拍同量级,不阻塞 HELLO retarget 之外的职责。
        """
        while not self._lwd_post_out_receiver.closed:
            try:
                self.lwd_result_queue.put_nowait(msg)
                break
            except queue.Full:
                logger.warning("[Lwd] result queue full, retrying (rid=%s)", msg.request_id)
                threading.Event().wait(0.01)
        self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

    def _lwd_shutdown_planes(self) -> None:
        """两面关停(幂等):receiver 先关(断输入),publisher 收尾。"""
        receiver = getattr(self, "_lwd_post_out_receiver", None)
        if receiver is not None:
            receiver.shutdown()
        publisher = getattr(self, "_lwd_publisher", None)
        if publisher is not None:
            publisher.shutdown()

    # ------------------------------------------------------------------ #
    # 引擎接口覆写(原 core.py 守卫 + LwdEdgeCore 编排迁入)               #
    # ------------------------------------------------------------------ #
    def add_request(self, request, request_wave: int = 0) -> None:
        """边校验 + 云预告 + 本地入队(原 core.py add_request 守卫语义)。"""
        if not self._lwd_active:
            super().add_request(request, request_wave)
            return
        self.scheduler.lwd_edge_add_request(request)

    def abort_requests(self, request_ids: list[str]) -> None:
        """abort 先出云再走原生清理(原 core.py abort 守卫语义)。"""
        if self._lwd_active:
            self.scheduler.lwd_edge_abort(request_ids)
        super().abort_requests(request_ids)

    def step(self):
        if not self._lwd_active:
            return super().step()
        return self._lwd_edge_step()

    def step_with_batch_queue(self):
        if not self._lwd_active:
            return super().step_with_batch_queue()
        return self._lwd_edge_step()

    def _lwd_edge_step(self) -> tuple[dict[int, object] | None, bool]:
        """编排:步首云结果消费(unembed→deliver) -> prefill 编排 ->
        步末僵尸检查;返回 (云结果输出 | None, prefill 是否有工作)。"""
        outputs, finished_reqs = self._lwd_edge_consume_results()
        executed: dict[str, int] = {}
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule()
            executed = self._lwd_edge_dispatch(scheduler_output)
            self._lwd_log.phase("edge step: %d reqs executed", len(executed))
        self.scheduler.lwd_edge_update_progress(executed)
        for request_id in self.scheduler.lwd_edge_zombie_check():
            outputs.append(
                EngineCoreOutput(
                    request_id, [], finish_reason=FinishReason.ABORT
                )
            )
            finished_reqs.add(request_id)
        if outputs:
            return (
                EngineCoreOutputs(
                    outputs=outputs,
                    finished_requests=finished_reqs or None,
                ),
                bool(executed),
            )
        return None, bool(executed)

    def _lwd_edge_consume_results(self) -> tuple[list, set]:
        """drain 结果队列 -> 组 UNEMBED 批提交 worker(lm_head,数据面按
        batch_type 分流)-> deliver 到 awaiting -> 组前端输出。

        worker 返回按批型约定为 dict[request_id -> token_ids](假定接口);
        LwdResultNotify 当前仅 request_id(占位),finished 暂按 True 处理,
        消息补 finished 字段后改为透传。迟到结果 deliver 返回 False,
        丢弃告警(幂等);UNEMBED 批与 prefill 排程串行,无顺序耦合。
        """
        notifies: list[LwdResultNotify] = []
        while True:
            try:
                notifies.append(self.lwd_result_queue.get_nowait())
            except queue.Empty:
                break
        if not notifies:
            return [], set()
        unembed_batch = lwd_build_unembed_batch([n.request_id for n in notifies])
        result = self.model_executor.execute_model(unembed_batch).result()
        token_map = result if isinstance(result, dict) else {}
        if not isinstance(result, dict):
            logger.warning(
                "[Lwd] unembed batch returned %r (expected token map), "
                "delivering empty tokens",
                type(result),
            )
        outputs: list = []
        finished_reqs: set = set()
        for notify in notifies:
            token_ids = list(token_map.get(notify.request_id, []))
            if not self.scheduler.lwd_edge_deliver_tokens(
                notify.request_id, token_ids, finished=True
            ):
                logger.warning(
                    "[Lwd] drop stale cloud result for %s (not awaiting)",
                    notify.request_id,
                )
                continue
            outputs.append(
                EngineCoreOutput(
                    notify.request_id,
                    token_ids,
                    finish_reason=FinishReason.STOP,
                )
            )
            finished_reqs.add(notify.request_id)
        return outputs, finished_reqs

    def _lwd_edge_dispatch(self, scheduler_output) -> dict[str, int]:
        """调度器 lwd_edge_notify 发预告 + 原生 executor 同步提交。

        预告失败(队满)本步不派发,返回空执行量 -> 进度回退、下一步重试。
        """
        if not self.scheduler.lwd_edge_notify(scheduler_output):
            return {}
        self.model_executor.execute_model(scheduler_output).result()
        # 同步执行:调度量即执行量;数据面落位后由此接缝改报实际量(§9.12)
        return dict(scheduler_output.num_scheduled_tokens)

    def shutdown(self) -> None:
        """两面关停后走原生(幂等;降级路径两面未建,容忍缺省)。"""
        if self._lwd_active:
            self._lwd_shutdown_planes()
        super().shutdown()
