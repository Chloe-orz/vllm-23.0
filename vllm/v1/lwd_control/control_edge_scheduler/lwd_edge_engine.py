"""边侧引擎子类:run_engine_core 类选择点注入的边 EngineCore。

边引擎职责:通信面装配(POST_OUT bind/接收线程/PRE_OUT 延迟连接)、
调度器注入(LwdEdgeScheduler)、请求入口/步进/关停的引擎接口覆写。
全部状态收进子类字段,不外挂引擎属性。

构造约束:
  ① 通信面先于引擎主体构建:bind POST_OUT、建云载荷队列与接收线程,
     阻塞等云首拍 HELLO——HELLO 在云引擎全量初始化(权重/KV/图编译)
     完成后才发出,等到了它才允许边侧对外就绪;超时(hello_timeout_s,
     默认 600s,须覆盖云全量启动时长)自清理两面后 fail-fast。
     HELLO 首拍一次、无周期重发,不考虑任一侧重启自愈:重启即整组重拉,
     构造期等待是边侧唯一的发现窗口。
  ② 调度器经 scheduler_cls 注入裸类,引擎构造完成后回填 publisher
     (早于任何请求,等价构造注入)。

步进编排:步首消费云载荷(c2e -> UNEMBED 批 -> token 交付/请求终结)
+ prefill 编排(单请求组批 -> 范围预告 -> 原生 executor 同步执行 ->
步末对账)+ 步末 awaiting 僵尸检查。
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
    LWD_NOT_FINISHED,
    LwdC2eNotify,
    LwdHelloNotify,
    lwd_decode_cloud_notify,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
    LwdConfig,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
    lwd_build_unembed_batch,
)

logger = init_logger(__name__)


class LwdLog:
    """按 debug 开关分级的极简诊断面;error 留给真正的业务中断点。"""

    def __init__(self, debug: bool = False) -> None:
        self._debug = debug

    def phase(self, message: str, *args) -> None:
        """步进/相位轨迹;默认关,生产路径零输出。"""
        if self._debug:
            logger.info("[Lwd] %s", message % args if args else message)

# 云->边载荷队列容量:队满时接收线程阻塞在 put,背压沿 ZMQ 直达云侧
# 步发送循环(载荷不可丢)
LWD_C2E_META_QUEUE_MAX = 1000


class LwdEdgeEngineCore(EngineCoreProc):
    """边 PO 引擎:通信面装配 + 调度器注入 + step/add/abort/shutdown 覆写。"""

    def __init__(self, *args, **kwargs) -> None:
        vllm_config = args[0]
        config = LwdConfig.from_env_and_config(vllm_config)
        self._lwd_log = LwdLog(config.debug)
        # 通信面:bind POST_OUT 订阅面 + 延迟连接的 PRE_OUT 发布面;
        # 云端点由 HELLO 通告决定(边不预知云地址)
        self._edge_receiver = self._lwd_build_post_out(config)
        self._edge_sender = LwdControlPublisher(
            None, bind=False, queue_max=config.publish_queue_max
        )
        # 云->边唯一载荷队列:生产端接收线程,消费端引擎步;数据面经
        # UNEMBED 批的 lwd_c2e_notifies 拿元数据,不直接读队列
        # (单消费者语义)
        self.lwd_c2e_meta_queue = queue.Queue(maxsize=LWD_C2E_META_QUEUE_MAX)
        hello_event = threading.Event()
        discovery = threading.Thread(
            target=self._receive_thread,
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

        vllm_config.scheduler_config.scheduler_cls = LwdEdgeScheduler
        super().__init__(*args, **kwargs)
        self.scheduler.lwd_edge_publisher = self._edge_sender
        logger.info(
            "[Lwd] edge engine assembled: POST_OUT bind %s, PRE_OUT discovered",
            config.lwd_post_out_bind_endpoint(),
        )

    # ------------------------------------------------------------------ #
    # 通信面                                                              #
    # ------------------------------------------------------------------ #
    def _lwd_build_post_out(self, config: LwdConfig) -> LwdControlSubscriber:
        """bind POST_OUT 订阅面;云经 master_addr 主动来连。"""
        return LwdControlSubscriber(
            config.lwd_post_out_bind_endpoint(),
            bind=True,
            decoder=lwd_decode_cloud_notify,
        )

    def _receive_thread(self, hello_event: threading.Event) -> None:
        """POST_OUT 接收线程体,按消息类型分发。

        HELLO -> retarget PRE_OUT:云端点唯一事实源,首拍一次通告;
        retarget 队满时无下条 HELLO 可等,须本线程自旋重试到成功
        (构造期发布队列必空,该路径仅防御性保活)。
        LwdC2eNotify(云->边唯一载荷)-> 载荷队列(阻塞 put,不可丢)
        + WAKEUP 唤醒主循环:prefill 全部完成后请求转入 awaiting,
        引擎无排程工作、阻塞在 input_queue.get(),载荷只进队列不会
        唤醒任何线程,必须向 input_queue 敲门;WAKEUP 原生语义即丢弃
        消息体,数据与唤醒分离,多投无害(空 drain 一步即返回)。
        其余帧(坏帧已被订阅层丢弃后仍不认识的类型)告警丢弃。
        """
        receiver = self._edge_receiver
        publisher = self._edge_sender
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
                while not publisher.retarget(endpoint):
                    if receiver.closed:
                        break
                    logger.warning(
                        "[Lwd] PRE_OUT retarget deferred (queue full), retrying"
                    )
                    threading.Event().wait(0.05)
                hello_event.set()
            elif isinstance(msg, LwdC2eNotify):
                self.lwd_c2e_meta_queue.put(msg)
                self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
            else:
                logger.warning("[Lwd] drop unexpected POST_OUT frame %r", type(msg))

    def _lwd_shutdown_planes(self) -> None:
        """两面关停(幂等):receiver 先关断输入,publisher 收尾。"""
        receiver = getattr(self, "_edge_receiver", None)
        if receiver is not None:
            receiver.shutdown()
        publisher = getattr(self, "_edge_sender", None)
        if publisher is not None:
            publisher.shutdown()

    # ------------------------------------------------------------------ #
    # 引擎接口覆写                                                        #
    # ------------------------------------------------------------------ #
    def add_request(self, request, _request_wave: int = 0) -> None:
        """边校验 + 云预告 + 本地入队。

        _request_wave:原生主循环按位置传入(core.py ADD 分发),本模式
        无 DP wave 语义,仅保形参契约,不消费。"""
        self.scheduler.lwd_edge_add_request(request)

    def abort_requests(self, request_ids: list[str]) -> None:
        """abort 信号先出云,再走原生本地清理。"""
        self.scheduler.lwd_edge_abort(request_ids)
        super().abort_requests(request_ids)

    def step(self):
        return self._lwd_edge_step()

    def step_with_batch_queue(self):
        return self._lwd_edge_step()

    def _lwd_edge_step(self) -> tuple[dict[int, object] | None, bool]:
        """单步编排:云载荷消费 -> prefill 编排 -> 僵尸检查;
        返回 (云侧结果输出 | None, prefill 是否有工作)。"""
        outputs, finished_reqs = self._lwd_edge_consume_c2e()
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

    def _lwd_edge_consume_c2e(self) -> tuple[list, set]:
        """消费云载荷(LwdC2eNotify,云->边唯一载荷)并产出前端输出。

        逐条通告交付(批间顺序 = 到达顺序),分流与交付逻辑收在
        _lwd_deliver_notify:先以完成码判 finish——完结通告走本地
        终结流程(不下发 worker);未完结组 UNEMBED 批提交 worker 做
        lm_head(一条通告对应一个 DOWN 张量,一一批配对无需切分拼接)。

        token 逐条交付(finish_reason=None),终结步透传云侧完成码
        (STOP/LENGTH 等),流式/非流式由原生前端透明处理,引擎层恒为
        增量;finish_reasons 与 req_ids 构造上严格对齐(云侧逐位推导),
        错配即 IndexError fail-fast,无缺省兜底。

        应答契约:worker 经原生 future 返回 ModelRunnerResult 形态,
        token ids 取 lwd_token_ids(request_id -> list[int]);缺失/为空
        即该请求 unembed 失败,以 FinishReason.ERROR 终结(原生 ERROR
        通道转 5xx),不静默降级为空 STOP 输出。迟到载荷(请求不在
        awaiting)丢弃告警,幂等不复活;UNEMBED 批与 prefill 排程在
        同一线程序行,无顺序耦合。
        """
        notifies: list[LwdC2eNotify] = []
        while True:
            try:
                notifies.append(self.lwd_c2e_meta_queue.get_nowait())
            except queue.Empty:
                break
        if not notifies:
            return [], set()

        outputs: list = []
        finished_reqs: set = set()

        for notify in notifies:
            # 通告级以完成码分流,一条通告一次交付
            self._lwd_deliver_notify(notify, outputs, finished_reqs)
        return outputs, finished_reqs

    def _lwd_deliver_notify(
        self, notify: LwdC2eNotify, outputs: list, finished_reqs: set,
    ) -> None:
        """单条 c2e 通告的完整交付流程。

        先以完成码判 finish:finish_reasons 全非哨兵 = 完结通告,走
        本地终结流程——不下发 worker,逐请求以云侧完成码空输出终结
        (原样透传 STOP/LENGTH 等);未完结则组 UNEMBED 批下发 worker
        做 unembed,应答后逐请求交付 token。行序 = req_ids 序,与
        DOWN 张量行序一致;一条通告对应一个 DOWN 张量,一一批配对
        无需切分拼接。

        finish_reasons 与 req_ids 逐位严格对齐(云侧逐位推导),错配
        IndexError fail-fast,无缺省兜底;协约:完结通告不携带
        hidden 行,违约丢弃并告警;迟到载荷(请求不在 awaiting)幂等
        丢弃。应答契约:token ids 取 ModelRunnerOutput.lwd_token_ids,
        缺失/为空 = unembed 失败,ERROR 优先于云侧完成码,不静默降级
        为空输出。"""

        # ---- finish 流程:不下发 worker,本地终结 ----
        if notify.req_ids and all(
            code != LWD_NOT_FINISHED for code in notify.finish_reasons
        ):
            if notify.hidden_num_elements > 0:
                logger.warning(
                    "[Lwd] finish-marked notify carries hidden rows, "
                    "rows dropped (cloud protocol violation)"
                )
            for index, request_id in enumerate(notify.req_ids):
                if not self.scheduler.lwd_edge_deliver_tokens(
                    request_id, [], finished=True
                ):
                    logger.warning(
                        "[Lwd] drop stale cloud payload for %s (not awaiting)",
                        request_id,
                    )
                    continue
                outputs.append(
                    EngineCoreOutput(
                        request_id,
                        [],
                        finish_reason=self._lwd_finish_code(notify, index),
                    )
                )
                finished_reqs.add(request_id)
            return

        # ---- 未完结:下发 worker 处理 ----
        if not notify.req_ids or notify.hidden_num_elements <= 0:
            # 无行且未完结:无内容可交付,防御跳过
            return
        future = self._lwd_dispatch_unembed(notify)
        self._lwd_deliver_unembed(notify, future, outputs, finished_reqs)

    def _lwd_dispatch_unembed(self, notify: LwdC2eNotify):
        """UNEMBED 批提交侧:组批并以 non_block 提交 worker,立即返回
        future(不等执行)。同步路径提交后立即收割;异步路径
        (batch_queue)将 (notify, future) 入队,收割阶段再等 future——
        提交与等待的边界即本函数返回处。"""
        unembed_batch = lwd_build_unembed_batch(notify)
        return self.model_executor.execute_model(
            unembed_batch, non_block=True
        )

    def _lwd_deliver_unembed(
        self, notify: LwdC2eNotify, future, outputs: list, finished_reqs: set,
    ) -> None:
        """UNEMBED 批收割侧:等 worker 应答取 token_map,逐请求交付
        token。不感知提交时机,只消费 (notify, future)。应答契约:
        token ids 取 lwd_token_ids,缺失/为空 = unembed 失败,ERROR
        优先于云侧完成码;迟到载荷幂等丢弃。"""
        result = future.result()
        token_map = getattr(result, "lwd_token_ids", None)
        if token_map is None:
            logger.warning(
                "[Lwd] unembed batch answer missing lwd_token_ids (%r), "
                "finishing requests with ERROR",
                type(result),
            )
        for index, request_id in enumerate(notify.req_ids):
            finish_reason = self._lwd_finish_code(notify, index)
            finished = finish_reason is not None
            token_ids = list(token_map.get(request_id, [])) if token_map else []
            if not token_ids:
                # 行在批里但无 token = unembed 失败,ERROR 优先于云侧码
                finish_reason = FinishReason.ERROR
                finished = True
            if not self.scheduler.lwd_edge_deliver_tokens(
                request_id, token_ids, finished=finished
            ):
                logger.warning(
                    "[Lwd] drop stale cloud payload for %s (not awaiting)",
                    request_id,
                )
                continue
            if not token_ids and not finished:
                # 无内容且未完结:不出空输出
                continue
            outputs.append(
                EngineCoreOutput(
                    request_id, token_ids, finish_reason=finish_reason
                )
            )
            if finished:
                finished_reqs.add(request_id)

    @staticmethod
    def _lwd_finish_code(
        notify: LwdC2eNotify, index: int
    ) -> FinishReason | None:
        """逐请求完成码:哨兵 = 未完结(None);否则透传 FinishReason。
        与 req_ids 按位严格对齐,错配 IndexError fail-fast。"""
        code = notify.finish_reasons[index]
        return None if code == LWD_NOT_FINISHED else FinishReason(code)

    def _lwd_edge_dispatch(self, scheduler_output) -> dict[str, int]:
        """范围预告 + 原生 executor 同步提交,返回每请求执行 token 数。

        预告失败(发布队列满)本步不派发,返回空执行量,调度器步末
        回退乐观推进量,下一步重试同一范围。"""
        if not self.scheduler.lwd_edge_notify(scheduler_output):
            return {}
        self.model_executor.execute_model(scheduler_output).result()
        # 同步执行:调度量即执行量;数据面异步化后由此改报实际量
        return dict(scheduler_output.num_scheduled_tokens)

    def shutdown(self) -> None:
        """两面关停后走原生;初始化失败路径两面可能未建,容忍缺省。"""
        self._lwd_shutdown_planes()
        super().shutdown()
