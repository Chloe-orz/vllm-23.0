"""边侧引擎子类:run_engine_core 类选择点注入的边 EngineCore。

边引擎职责:通信面装配(POST_OUT bind/接收线程/PRE_OUT 延迟连接)、
调度器注入(LwdEdgeScheduler)、请求入口/步进/关停的引擎接口覆写。
全部状态收进子类字段,不外挂引擎属性。

构造约束:
  ① 引擎主体先行(super() 最先调):接收线程 WAKEUP 敲门的
     self.input_queue 由 super 创建,后启动线程即无构造期竞态;代价是
     云缺失时引擎主体启动白费,超时路径只清理通信面,主体随进程退出
     兜底回收。通信面随后构建:bind POST_OUT、起接收线程,阻塞等云
     首拍 HELLO——HELLO 在云引擎全量初始化(权重/KV/图编译)完成后
     才发出,等到了它才允许边侧对外就绪;超时(hello_timeout_s,默认
     600s,须覆盖云全量启动时长)fail-fast。HELLO 首拍一次、无周期
     重发,不考虑任一侧重启自愈:重启即整组重拉,构造期等待是边侧
     唯一的发现窗口。
  ② 调度器经 scheduler_cls 注入裸类,须在 super() 之前设值——super 内
     构建调度器时一次性消费该配置,后设无效;引擎构造完成后回填
     publisher(早于任何请求,等价构造注入)。

步进编排:步首收割在飞批(UNEMBED 经原生 update_from_output 入账
销欠/判停/终结,EMBED 收割即无事)→ 连续调度派发至深度上限(调度器
相位模板决定批型:prefill 步产 EMBED 批,decode 步产 UNEMBED 批)。
c2e 通告由接收线程直入调度器 decode_notify_queue,WAKEUP 唤醒主循环。
"""

from __future__ import annotations

import threading
from collections import deque
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.sched.output import LwdBatchType
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
    LwdControlPublisher,
)
from vllm.v1.lwd_control.control_communication.lwd_control_subscriber import (
    LwdControlSubscriber,
)
from vllm.v1.lwd_debug import LwdLogBase
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LwdC2eNotify,
    LwdHelloNotify,
    lwd_decode_cloud_notify,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
)

if TYPE_CHECKING:
    from vllm.config.lwd import LwdConfig

logger = init_logger(__name__)


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
        config = vllm_config.lwd_config
        # 层日志总开关:env 已开则不动,config 段开则补开(仅本进程)
        LwdLogBase.set_debug(config.debug)
        # 通信面:bind POST_OUT 订阅面 + 延迟连接的 PRE_OUT 发布面;
        # 云端点由 HELLO 通告决定(边不预知云地址)
        self._edge_receiver = self._lwd_build_post_out(config)
        self._edge_sender = LwdControlPublisher(
            None, bind=False, queue_max=config.publish_queue_max
        )
        # 已派发待收割的批队列 (kind, payload, future):
        # kind="unembed"/"embed" payload=scheduler_output。
        # 派发不收割,队首 FIFO 收割——必须全局单队列:executor 的 FutureWrapper
        # 按底层队列序排水,交错收割会连带等错批。
        self._lwd_batch_queue: deque = deque()
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
                f"(bind tcp://{config.post_out_bind}:{config.post_out_port}; "
                f"check cloud master_addr connectivity and POST_OUT port)"
            )

        self.scheduler.lwd_edge_publisher = self._edge_sender
        logger.info(
            "[Lwd] edge engine assembled: POST_OUT bind %s, PRE_OUT discovered",
            f"tcp://{config.post_out_bind}:{config.post_out_port}",
        )

    # ------------------------------------------------------------------ #
    # 通信面                                                              #
    # ------------------------------------------------------------------ #
    def _lwd_build_post_out(self, config: LwdConfig) -> LwdControlSubscriber:
        """bind POST_OUT 订阅面;云经 master_addr 主动来连。"""
        return LwdControlSubscriber(
            f"tcp://{config.post_out_bind}:{config.post_out_port}",
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
                    "[Lwd][edge-ctrl] C2eNotify reqs=%d down_seqno=%s",
                    len(msg.req_ids), msg.down_seqno,
                )
                # decode 通告入调度器队列(decode 步弹队首点名);
                # WAKEUP 打断主循环的阻塞 get
                self.scheduler.decode_notify_queue.append(msg)
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
        """decode 通告队列与待收割批队列都计入工作:WAKEUP 只能打断阻塞
        的 get,轮询循环是否退出由本判据决定(core.py while not has_work)。
        批队列非空必须计入——否则最后一批派发后引擎睡眠,最终 token
        永不收割交付。"""
        return (super().has_work() or bool(self.scheduler.decode_notify_queue)
                or self._lwd_batch_queue)

    def _lwd_edge_step(self) -> tuple[dict[int, object] | None, bool]:
        """单步编排:收割队首直至清空 -> 连续派发至深度上限(调度器
        相位选择决定 EMBED/UNEMBED 批型)-> 返回({0: 输出} | None,
        是否有模型执行)。

        批型由调度器模板决定:prefill 步产 EMBED 批(发布 RangeNotify
        后提交),decode 步产 UNEMBED 批(调度器已挂 c2e 载荷,直接
        提交)。收割:UNEMBED 批经原生 update_from_output 入账销欠、
        判停、finish;EMBED 批收割即无事(进度原生推进,完结请求留
        running 等 c2e)。批队列全局单 FIFO,序与 executor 排水序
        同构,不可交错。"""
        outputs: list = []
        finished_reqs: set = set()
        model_executed = False
        while self._lwd_batch_queue:
            kind, payload, future = self._lwd_batch_queue.popleft()
            if kind == "unembed":
                result = future.result()
                if result is not None:
                    for outputs_per_rank in self.scheduler.update_from_output(
                        payload, result
                    ).values():
                        outputs.extend(outputs_per_rank.outputs)
                        finished_reqs.update(
                            outputs_per_rank.finished_requests or ()
                        )
                model_executed = True
            else:
                model_executed = True
        while (len(self._lwd_batch_queue) < LWD_EDGE_BATCH_QUEUE_DEPTH
               and self.scheduler.has_requests()):
            scheduler_output = self.scheduler.schedule()
            if not scheduler_output.num_scheduled_tokens:
                break
            batch = scheduler_output.lwd_batch
            is_unembed = (
                batch is not None
                and batch.batch_type is LwdBatchType.LWD_UNEMBED
            )
            if is_unembed:
                future = self._lwd_dispatch_unembed(scheduler_output)
            else:
                future = self._lwd_dispatch_embed(scheduler_output)
            if future is None:
                break
            self._lwd_batch_queue.append(
                ("unembed" if is_unembed else "embed", scheduler_output, future)
            )
            model_executed = True
        if outputs:
            step_outputs = EngineCoreOutputs(
                outputs=outputs,
                finished_requests=finished_reqs or None,
            )
            return {0: step_outputs}, model_executed
        return None, model_executed

    def _lwd_dispatch_embed(self, scheduler_output):
        """EMBED 批提交侧(唯一提交点,同步/异步共用):范围预告(发布
        成功才组批挂 lwd_batch)后以 non_block 提交,返回 future;
        发布失败返回 None,本轮不再提交。

        正确性前提:发布通道不丢消息。失败分支无回退可施——进度
        已按排程乐观推进,该 chunk 随 SO 废弃即永久丢失(进度虚高),
        正确性由通道不丢保证;同步/异步失败语义镜像(均为弃批)。"""
        if not self.scheduler.lwd_edge_notify(scheduler_output):
            return None
        return self.model_executor.execute_model(
            scheduler_output, non_block=True
        )

    def _lwd_dispatch_unembed(self, scheduler_output):
        """UNEMBED 批提交侧:调度器 decode 步已在 SO 挂好 c2e 载荷
        (lwd_batch),以 non_block 提交 worker,立即返回 future(不等
        执行);收割阶段经原生 update_from_output 入账销欠。"""
        return self.model_executor.execute_model(
            scheduler_output, non_block=True
        )

    def shutdown(self) -> None:
        """两面关停后走原生;初始化失败路径两面可能未建,容忍缺省。"""
        self._lwd_shutdown_planes()
        super().shutdown()
