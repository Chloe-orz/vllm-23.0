"""边侧引擎子类:run_engine_core 类选择点注入的边 EngineCore。

边引擎职责:通信面装配(POST_OUT bind/接收线程/PRE_OUT 延迟连接)、
调度器注入(LwdEdgeScheduler)、请求入口/步进/关停的引擎接口覆写。
全部状态收进子类字段,不外挂引擎属性。

构造约束:
  ① 引擎主体先行(super() 最先调):接收线程 WAKEUP 敲门的
     self.input_queue 由 super 创建,后启动线程即无构造期竞态;代价是
     云缺失时引擎主体启动白费,超时路径只清理通信面,主体随进程退出
     兜底回收。通信面随后构建:bind POST_OUT、建云载荷队列与接收线程,
     阻塞等云首拍 HELLO——HELLO 在云引擎全量初始化(权重/KV/图编译)
     完成后才发出,等到了它才允许边侧对外就绪;超时(hello_timeout_s,
     默认 600s,须覆盖云全量启动时长)fail-fast。
     HELLO 首拍一次、无周期重发,不考虑任一侧重启自愈:重启即整组重拉,
     构造期等待是边侧唯一的发现窗口。
  ② 调度器经 scheduler_cls 注入裸类,须在 super() 之前设值——super 内
     构建调度器时一次性消费该配置,后设无效;引擎构造完成后回填
     publisher(早于任何请求,等价构造注入)。

步进编排:步首消费云载荷(c2e -> UNEMBED 批 -> token 交付/请求终结)
+ prefill 编排(单请求组批 -> 范围预告 -> 原生 executor 同步执行 ->
步末完结登记)。
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque

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
from vllm.v1.lwd_debug import LwdDebug, LwdLogBase
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
)

logger = init_logger(__name__)


# 云->边载荷队列容量:队满时接收线程阻塞在 put,背压沿 ZMQ 直达云侧
# 步发送循环(载荷不可丢)
LWD_C2E_META_QUEUE_MAX = 1000

# 批队列在飞深度上限:派发(embed+unembed 合计)不收割的批数上限,
# 突发积压时连续派发喂饱 worker(参照仓经验 >=4 才能维持流水对齐)
LWD_EDGE_BATCH_QUEUE_DEPTH = 4


class LwdEdgeEngineCore(EngineCoreProc):
    """边 PO 引擎:通信面装配 + 调度器注入 + step/add/abort/shutdown 覆写。"""

    def __init__(self, *args, **kwargs) -> None:
        vllm_config = kwargs["vllm_config"]
        # 调度器注入须赶在 super() 之前:super 内构建 self.scheduler 时
        # 一次性消费 scheduler_cls,后设无效(注入失效,首请求即崩)
        vllm_config.scheduler_config.scheduler_cls = LwdEdgeScheduler
        super().__init__(*args, **kwargs)
        config = LwdConfig.from_env_and_config(vllm_config)
        # 层日志总开关:env 已开则不动,config 段开则补开(仅本进程)
        LwdLogBase.set_debug(config.debug)
        # 通信面:bind POST_OUT 订阅面 + 延迟连接的 PRE_OUT 发布面;
        # 云端点由 HELLO 通告决定(边不预知云地址)
        self._edge_receiver = self._lwd_build_post_out(config)
        self._edge_sender = LwdControlPublisher(
            None, bind=False, queue_max=config.publish_queue_max
        )
        # 云->边唯一载荷队列:生产端接收线程,消费端交付线程
        self.lwd_c2e_meta_queue = queue.Queue(maxsize=LWD_C2E_META_QUEUE_MAX)
        # token 交付线程产物缓冲:交付在独立线程做(与引擎步的模型批
        # 完全解耦——embed 收割阻塞多久都不耽误 token 到达即付),
        # 引擎步首在锁内一把换出,经 step() 返回值走原生输出通道
        self._lwd_delivery_lock = threading.Lock()
        self._lwd_delivery_outputs: list = []
        self._lwd_delivery_finished: set = set()
        self._lwd_delivery_stop = threading.Event()
        # 已派发待收割的批队列 (kind, payload, t_dispatch, future):
        # kind="embed" payload=scheduler_output。派发不收割,队首 FIFO
        # 收割——必须全局单队列:executor 的 FutureWrapper 按底层队列序
        # 排水,交错收割会连带等错批。t_dispatch 供 [Lwd][sched]
        # harvest wait(派发→收割)度量批在队列里的滞留时长
        self._lwd_batch_queue: deque = deque()
        hello_event = threading.Event()
        discovery = threading.Thread(
            target=self._receive_thread,
            args=(hello_event,),
            name="lwd-post-in",
            daemon=True,
        )
        discovery.start()
        delivery = threading.Thread(
            target=self._lwd_delivery_loop,
            name="lwd-token-deliver",
            daemon=True,
        )
        delivery.start()
        if not hello_event.wait(config.hello_timeout_s):
            self._lwd_shutdown_planes()
            raise RuntimeError(
                f"[Lwd] edge engine init failed: no cloud HELLO within "
                f"{config.hello_timeout_s}s on POST_OUT "
                f"(bind {config.lwd_post_out_bind_endpoint()}; check cloud "
                f"master_addr connectivity and POST_OUT port)"
            )

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
                logger.info(
                    "[Lwd][edge-ctrl] C2eNotify reqs=%d tokens=%s",
                    len(getattr(msg, "req_ids", []) or []),
                    [len(t) for t in getattr(msg, "token_ids", [])],
                )
                # 队列元素带到达时间戳:消费滞后(c2e_wait=到达→派发)
                # 是"边消费不动 → 云 publisher 队满小睡"闭环的前置指标
                self.lwd_c2e_meta_queue.put((msg, time.monotonic()))
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

    def has_work(self) -> bool:
        """c2e 载荷队列与待收割批队列都计入工作:WAKEUP 只能打断阻塞的
        get,轮询循环是否退出由本判据决定(core.py while not has_work)。
        批队列非空必须计入——否则最后一批派发后引擎睡眠,最终 token
        永不收割交付。"""
        # 交付缓冲非空也计入:交付线程独立消费 c2e,引擎可能无排程
        # 工作时缓冲里已有 token——漏计会让主循环睡眠,token 滞留
        return (super().has_work() or not self.lwd_c2e_meta_queue.empty()
                or self._lwd_batch_queue or self._lwd_delivery_outputs)

    def _lwd_edge_step(self) -> tuple[dict[int, object] | None, bool]:
        """单步编排(token_id 版 + 交付线程):换出交付线程产物 -> 收割
        embed 批队列直至清空 -> prefill 连续派发 -> 返回({0: 云侧结果
        输出} | None,prefill 是否有派发)。

        token 交付在独立线程(到达即付,不受 embed 收割/派发阻塞);
        引擎步只做模型批编排。embed 批派发入队不收割,队首 FIFO 收割
        (序与 executor 排水序同构)。

        [Lwd][sched] 三类日志:交付线程 deliver、每批 harvest(含队列
        滞留 wait)、每步汇总。"""
        with self._lwd_delivery_lock:
            outputs = self._lwd_delivery_outputs
            finished_reqs = self._lwd_delivery_finished
            self._lwd_delivery_outputs = []
            self._lwd_delivery_finished = set()
        prefill_work = False
        h_emb = 0
        while self._lwd_batch_queue:
            _kind, payload, t_dispatch, future = self._lwd_batch_queue.popleft()
            self.scheduler.lwd_edge_update_progress(
                dict(payload.num_scheduled_tokens)
            )
            future.result()
            prefill_work = True
            h_emb += 1
            logger.info(
                "[Lwd][sched] edge harvest embed seqno=%s wait=%.2fms",
                self._lwd_batch_seqno(payload),
                (time.monotonic() - t_dispatch) * 1000,
            )
        d_emb = 0
        while (len(self._lwd_batch_queue) < LWD_EDGE_BATCH_QUEUE_DEPTH
               and self.scheduler.has_requests()):
            scheduler_output = self.scheduler.schedule()
            if not scheduler_output.num_scheduled_tokens:
                break
            future = self._lwd_dispatch_embed(scheduler_output)
            if future is None:
                break
            self._lwd_batch_queue.append(
                ("embed", scheduler_output, time.monotonic(), future)
            )
            prefill_work = True
            d_emb += 1
        logger.info(
            "[Lwd][sched] edge step deliver=%d harvest_emb=%d "
            "disp_emb=%d queue=%demb c2e_pending=%d",
            len(outputs), h_emb, d_emb, len(self._lwd_batch_queue),
            self.lwd_c2e_meta_queue.qsize(),
        )
        if outputs:
            step_outputs = EngineCoreOutputs(
                outputs=outputs,
                finished_requests=finished_reqs or None,
            )
            return {0: step_outputs}, prefill_work
        return None, prefill_work

    @staticmethod
    def _lwd_batch_seqno(payload) -> str:
        """harvest 日志的配对号:embed 批 seqno(与 RangeNotify/UP 同号)。"""
        batch = getattr(payload, "lwd_batch", None)
        return str(getattr(batch, "seqno", "?")) if batch else "?"

    def _lwd_delivery_loop(self) -> None:
        """token 交付线程体:排空 c2e 队列,到达即付。

        与引擎步(模型批收割/派发)完全解耦——embed future 阻塞多久都
        不影响 token 交付;产物进缓冲并敲 WAKEUP,引擎步首换出经原生
        通道发给前端。awaiting 台账在调度器侧以幂等操作跨线程访问。
        """
        while not self._lwd_delivery_stop.is_set():
            try:
                notify, t_arrive = self.lwd_c2e_meta_queue.get(
                    timeout=0.5
                )
            except queue.Empty:
                continue
            logger.info(
                "[Lwd][sched] edge deliver-unembed reqs=%d tokens=%s "
                "c2e_wait=%.2fms c2e_pending=%d",
                len(notify.req_ids),
                [len(t) for t in notify.token_ids],
                (time.monotonic() - t_arrive) * 1000,
                self.lwd_c2e_meta_queue.qsize(),
            )
            outputs: list = []
            finished_reqs: set = set()
            self._lwd_deliver_notify(notify, outputs, finished_reqs)
            if outputs:
                with self._lwd_delivery_lock:
                    self._lwd_delivery_outputs.extend(outputs)
                    self._lwd_delivery_finished |= finished_reqs
                # 引擎可能已无其他工作在睡:交付本身不产生排程事件,
                # 须主动唤醒主循环换出缓冲(WAKEUP 多投无害)
                self.input_queue.put_nowait(
                    (EngineCoreRequestType.WAKEUP, None)
                )

    def _lwd_dispatch_embed(self, scheduler_output):
        """EMBED 批提交侧(唯一提交点,同步/异步共用):范围预告(发布
        成功才组批挂 lwd_batch)后以 non_block 提交,返回 future;
        发布失败返回 None,本轮不再提交。

        正确性前提:发布通道不丢消息。失败分支无回退可施——进度
        已按排程乐观推进,该 chunk 随 SO 废弃即永久丢失(进度虚高),
        正确性由通道不丢保证;同步/异步失败语义镜像(均为弃批)。"""
        if not self.scheduler.lwd_edge_notify(scheduler_output):
            return None
        batch = scheduler_output.lwd_batch
        logger.info(
            "[Lwd][sched] edge dispatch-embed seqno=%d req=%s tokens=%d "
            "queue=%demb c2e_pending=%d",
            batch.seqno, batch.batch_meta.req_ids[0],
            scheduler_output.total_num_scheduled_tokens,
            len(self._lwd_batch_queue), self.lwd_c2e_meta_queue.qsize(),
        )
        return self.model_executor.execute_model(
            scheduler_output, non_block=True
        )

    def _lwd_deliver_notify(
        self, notify: LwdC2eNotify, outputs: list, finished_reqs: set,
    ) -> None:
        """c2e 通告交付侧(token_id 版):token ids 云侧已随通告到达,
        无 worker/future,逐请求直接交付。语义与旧 unembed 收割一致:
        行无 token = 云侧本步无产出(discard/缺席),ERROR 优先于云侧
        完成码;迟到载荷幂等丢弃。"""
        token_map = dict(zip(notify.req_ids, notify.token_ids))
        for index, request_id in enumerate(notify.req_ids):
            finish_reason = self._lwd_finish_code(notify, index)
            finished = finish_reason is not None
            sampled_token_ids = list(token_map.get(request_id, []))
            LwdDebug.edge_tokens_delivered(  # [lwd-debug]
                request_id, sampled_token_ids, finish_reason,
                self.vllm_config,
            )
            if not sampled_token_ids:
                # 行在通告里但无 token = 本步无产出,ERROR 优先于云侧码
                finish_reason = FinishReason.ERROR
                finished = True
            if not self.scheduler.lwd_edge_deliver_tokens(
                request_id, sampled_token_ids, finished=finished
            ):
                logger.warning(
                    "[LWD] drop stale cloud payload for %s (not awaiting)",
                    request_id,
                )
                continue
            if not sampled_token_ids and not finished:
                # 无内容且未完结:不出空输出
                continue
            outputs.append(
                EngineCoreOutput(
                    request_id, sampled_token_ids,
                    finish_reason=finish_reason,
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

    def shutdown(self) -> None:
        """交付线程关停 + 两面关停后走原生;初始化失败路径容忍缺省。"""
        getattr(self, "_lwd_delivery_stop", threading.Event()).set()
        self._lwd_shutdown_planes()
        super().shutdown()
