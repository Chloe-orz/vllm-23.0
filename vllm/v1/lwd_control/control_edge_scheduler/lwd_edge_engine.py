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
import time
from collections import deque
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequestType, RequestStatus
from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
    LwdControlPublisher,
)
from vllm.v1.lwd_control.control_communication.lwd_control_subscriber import (
    LwdControlSubscriber,
)
from vllm.v1.lwd_debug import LwdLogBase
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LWD_WIRE_SAMPLING_FIELDS,
    LwdAbortNotify,
    LwdC2eNotify,
    LwdHelloNotify,
    LwdRequestNotify,
    lwd_decode_cloud_notify,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import LwdBaseEngineCore

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)

# 步进流水深度:对齐旧手搓批队列的深度,由原生 batch_queue 机制承担
LWD_EDGE_BATCH_DEPTH = 4

# 请求元数据预告发布重试:次数 x 递增间隔(共约 3s),耗尽即请求级报错
_LWD_ADD_RETRY_STEPS = 5
_LWD_ADD_RETRY_INTERVAL_S = 0.2


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
        # HELLO 首拍一次、无重发,不考虑任一侧重启自愈:重启即整组重拉,
        # 构造期等待是边侧唯一的发现窗口
        self._hello_event = threading.Event()
        self._lwd_setup_planes()
        # 发布面回填调度器(schedule_prefill 的 RangeNotify 出口)
        self.scheduler.lwd_publisher = self._publisher
        if not self._hello_event.wait(config.hello_timeout_s):
            self._lwd_shutdown_planes()
            raise RuntimeError(
                f"[LWD] edge engine init failed: no cloud HELLO within "
                f"{config.hello_timeout_s}s on POST_OUT "
                f"(bind tcp://{config.post_out_bind}:{config.post_out_port}; "
                f"check cloud master_addr connectivity and POST_OUT port)"
            )
        logger.info(
            "[Lwd] edge engine assembled: POST_OUT bind %s, PRE_OUT discovered",
            f"tcp://{config.post_out_bind}:{config.post_out_port}",
        )

    def _lwd_build_subscriber(self) -> LwdControlSubscriber:
        """bind POST_OUT 订阅面;云经 master_addr 主动来连。"""
        config = self.lwd_config
        return LwdControlSubscriber(
            f"tcp://{config.post_out_bind}:{config.post_out_port}",
            bind=True,
            decoder=lwd_decode_cloud_notify,
        )

    def _lwd_build_publisher(self) -> LwdControlPublisher:
        """PRE_OUT 延迟连接:端点由 HELLO 通告后 retarget。"""
        return LwdControlPublisher(
            None, bind=False, queue_max=self.lwd_config.publish_queue_max
        )

    # ------------------------------------------------------------------ #
    # 通信面                                                              #
    # ------------------------------------------------------------------ #
    def _lwd_on_message(self, msg) -> None:
        """云消息路由:HELLO 发现(retarget + 放行构造等待)→ c2e 入
        调度器通告队列并 WAKEUP → 其余告警丢弃。

        retarget 队满时无下条 HELLO 可等,须自旋重试到成功(构造期
        发布队列必空,该路径仅防御性保活);WAKEUP 原生语义即丢弃
        消息体,多投无害(空 drain 一步即返回)。"""
        if isinstance(msg, LwdHelloNotify):
            endpoint = f"tcp://{msg.pre_out_host}:{msg.pre_out_port}"
            if not self._hello_event.is_set():
                logger.info(
                    "[Lwd] cloud discovered via HELLO: PRE_OUT -> %s", endpoint
                )
            while not self._publisher.retarget(endpoint):
                if self._subscriber.closed:
                    break
                logger.warning(
                    "[Lwd] PRE_OUT retarget deferred (queue full), retrying"
                )
                threading.Event().wait(0.05)
            self._hello_event.set()
        elif isinstance(msg, LwdC2eNotify):
            logger.info(
                "[Lwd][edge-ctrl] C2eNotify reqs=%d down_seqno=%s",
                len(msg.req_ids), msg.down_seqno,
            )
            self.scheduler.unembed_notify_queue.append(msg)
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
        else:
            logger.warning("[Lwd] drop unexpected POST_OUT frame %r", type(msg))

    def add_request(self, request, _request_wave: int = 0) -> None:
        """云侧元数据预告 → 原生入队(预告先行,云只能准备不能开算)。
        abort_immediately 走 finish + abort 出口。

        _request_wave:原生主循环按位置传入(core.py ADD 分发),本模式
        无 DP wave 语义,仅保形参契约,不消费。"""
        self._lwd_notify_request_meta(
            request_id=request.request_id,
            num_prompt_tokens=len(request.prompt_token_ids),
            sampling_params=request.sampling_params,
            block_hashes=list(request.block_hashes),
        )
        self.scheduler.add_request(request)
        if request.abort_immediately:
            self.scheduler.finish_requests(
                [request.request_id], RequestStatus.FINISHED_ABORTED
            )
            self._lwd_abort_notify([request.request_id])

    def abort_requests(self, request_ids: list[str]) -> None:
        """abort 信号先出云,再走原生本地清理。"""
        self._lwd_abort_notify(request_ids)
        super().abort_requests(request_ids)

    def _lwd_notify_request_meta(
        self,
        request_id: str,
        num_prompt_tokens: int,
        sampling_params: SamplingParams,
        block_hashes: list[bytes] | None = None,
    ) -> None:
        """发 LwdRequestNotify:采样参数/满块哈希链透传(云侧占位
        prompt 的前缀缓存只能靠边侧哈希链命中;stop 字符串等
        detokenizer 层参数不上 wire)。队满短退避重试,耗尽抛
        RuntimeError 回客户端(请求未入队,云侧零残留)。"""
        sp = sampling_params
        message = LwdRequestNotify(
            request_id=request_id,
            num_prompt_tokens=num_prompt_tokens,
            max_tokens=sp.max_tokens if sp.max_tokens is not None else 16,
            block_hashes=block_hashes if block_hashes is not None else [],
            eos_token_id=sp.eos_token_id,
            stop_token_ids=list(sp.stop_token_ids or []),
            **{f: getattr(sp, f) for f in LWD_WIRE_SAMPLING_FIELDS},
        )
        for attempt in range(_LWD_ADD_RETRY_STEPS):
            if self._publisher.publish(message):
                logger.info(
                    "[Lwd][edge-notify] request meta announced: req=%s "
                    "prompt=%d",
                    request_id, num_prompt_tokens,
                )
                return
            time.sleep(_LWD_ADD_RETRY_INTERVAL_S * (attempt + 1))
        raise RuntimeError(
            f"[LWD] add-request notify for {request_id} dropped: publish "
            f"queue full after {_LWD_ADD_RETRY_STEPS} retries "
            f"(cloud PRE_OUT consumption stalled?)"
        )

    def _lwd_abort_notify(self, request_ids: list[str]) -> None:
        """发 LwdAbortNotify;本地清理走原生(请求在三队列内)。"""
        for request_id in request_ids:
            if not self._publisher.publish(
                LwdAbortNotify(request_id=request_id)
            ):
                logger.warning(
                    "[LWD] drop abort signal for %s: publish queue full", request_id
                )
            else:
                logger.info(
                    "[Lwd][edge-notify] AbortNotify req=%s", request_id
                )
