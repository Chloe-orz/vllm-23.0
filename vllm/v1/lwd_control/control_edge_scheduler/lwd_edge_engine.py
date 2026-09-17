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

步进全走父类(core.py step / step_with_batch_queue):调度器相位模板
出批并自带载荷(EMBED 批在 schedule_prefill 内发布 RangeNotify 后挂载,
UNEMBED 批在 schedule_decode 内挂 c2e 载荷),收割与 update_from_output
均原生。c2e 通告由接收线程直入调度器 unembed_notify_queue,WAKEUP 唤醒
主循环;异步调度深度由原生 batch_queue 机制承担。
"""

from __future__ import annotations

import threading
from collections import deque
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequestType
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
from vllm.v1.lwd_control.lwd_base_engine import LwdBaseEngineCore

if TYPE_CHECKING:
    from vllm.config.lwd import LwdConfig

logger = init_logger(__name__)

# 步进流水深度:对齐旧手搓批队列的深度,由原生 batch_queue 机制承担
LWD_EDGE_BATCH_DEPTH = 4


class LwdEdgeEngineCore(LwdBaseEngineCore):
    """边 PO 引擎:通信面装配 + add/abort 覆写。

    步进全走父类(schedule → execute → update_from_output):调度器
    相位模板出 EMBED/UNEMBED 批并自带载荷,worker 契约不变;流水深度
    强制 4,由原生 batch_queue 机制承担。"""

    lwd_scheduler_cls = LwdEdgeScheduler

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 流水深度强制 4:切换到原生 batch_queue 异步调度路径(worker 侧
        # non_block 提交契约与旧手搓流水线一致;原生 async_scheduling 标志
        # 仅剩 spec-decode 消费点,边侧无 spec,强行切换无冲突)
        self.batch_queue_size = LWD_EDGE_BATCH_DEPTH
        self.batch_queue = deque(maxlen=LWD_EDGE_BATCH_DEPTH)
        self.step_fn = self.step_with_batch_queue
        config = self.lwd_config
        # 层日志总开关:env 已开则不动,config 段开则补开(仅本进程)
        LwdLogBase.set_debug(config.debug)
        # 通信面:bind POST_OUT 订阅面 + 延迟连接的 PRE_OUT 发布面;
        # 云端点由 HELLO 通告决定(边不预知云地址)
        self._subscriber = self._lwd_build_post_out(config)
        self._publisher = LwdControlPublisher(
            None, bind=False, queue_max=config.publish_queue_max
        )
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

        self.scheduler.lwd_edge_publisher = self._publisher
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
        receiver = self._subscriber
        publisher = self._publisher
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
                self.scheduler.unembed_notify_queue.append(msg)
                self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
            else:
                logger.warning("[Lwd] drop unexpected POST_OUT frame %r", type(msg))

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
