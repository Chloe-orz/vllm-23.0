"""边侧 L3 引擎子类:类选择点注入的边 EngineCore(对齐云侧 §10.14,替代
已删的装配层 lwd_edge_try_assemble/step_wrapper/LwdEnginePort)。

接入点 = run_engine_core 类选择点(lwd_resolve_engine_cls 边角色分支):
引擎出生即 LwdEdgeEngineCore,通信面(POST_OUT bind/发现线程/HELLO 等待)
与调度器注入(LwdEdgeScheduler)全部收进子类 __init__,引擎属性零外挂。

装配时序(§9.1/§9.10):
  ① kv_transfer_config 预检:在场则降级原生(不建任何 lwd 状态,覆写
     方法按 _lwd_active=False 走 super(),语义同旧装配降级)。
  ② 通信面先建:bind POST_OUT、延迟连接的 PRE_OUT publisher、云载荷
     队列(lwd_c2e_meta_queue)、lwd-post-in 发现线程,阻塞等首条 HELLO
     (hello_timeout_s fail-fast,失败自清理两面后抛)。
  ③ scheduler_cls 注入裸类后 super().__init__():调度器出生即
     LwdEdgeScheduler(零整实例重建);publisher 构造后回填(装配期完成,
     早于任何请求,等价构造注入——get_scheduler_cls 无法携带实参)。

步进编排(原 LwdEdgeCore 迁入):step/step_with_batch_queue 覆写 =
步首云载荷消费(c2e 元数据 -> UNEMBED 批 -> deliver -> 前端输出)+
调度器 schedule(单请求组批) -> lwd_edge_notify 发预告(队满整步回退)
-> 原生 executor 同步执行 -> 步末对账。
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
from vllm.v1.lwd_control.control_edge_scheduler.lwd_step_core import LwdLog

logger = init_logger(__name__)

# 云->边步元数据接缝队列容量(§9.12):生产端 lwd-post-in,消费端随数据面落位
LWD_C2E_META_QUEUE_MAX = 1000


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
        # 云->边唯一载荷队列(§9.12):LwdC2eNotify 步元数据(兼结果回传
        # 驱动);生产端 lwd-post-in,消费端引擎步(UNEMBED 派发)与数据面
        # (预挂 recv);队满阻塞:元数据不可丢,背压沿 zmq 直达云侧
        self.lwd_c2e_meta_queue = queue.Queue(maxsize=LWD_C2E_META_QUEUE_MAX)
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
        """接收线程(lwd-post-in):消费 POST_OUT,按类型分发。

        HELLO -> retarget PRE_OUT(云端点唯一事实源,§9.1):首拍一次通告,
        无周期重发(裁定:不考虑云换址重启/边重启重连的自愈——任一
        侧重启即整组重拉,边侧装配期 30s 等待是唯一发现窗口);retarget
        队满无下条 HELLO 可等,必须本线程内自旋重试到成功。
        LwdC2eNotify(云->边唯一载荷)-> 元数据队列(阻塞 put:不可丢,
        背压沿 zmq 直达云侧步循环)+ WAKEUP 唤醒主循环:引擎可能阻塞
        在 input_queue.get()(prefill 全部完成后 awaiting 无排程工作),
        元数据只进 lwd_c2e_meta_queue 不会唤醒任何线程,必须敲门;
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
                while not publisher.retarget(endpoint):
                    # 首拍唯一通告:无下条 HELLO 可等,队满只能本线程自旋
                    # 重试(构造期队列为空,此处几乎不可达,防御性保活)
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
                # 协议分面:POST_OUT 只承载 HELLO 与步元数据(唯一载荷),
                # 坏帧已被订阅层丢弃
                logger.warning("[Lwd] drop unexpected POST_OUT frame %r", type(msg))

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
        """编排:步首云载荷消费(c2e 元数据 -> unembed -> deliver) ->
        prefill 编排 -> 步末僵尸检查;返回 (云结果输出 | None, prefill
        是否有工作)。"""
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
        """drain 云载荷队列(LwdC2eNotify,云->边唯一载荷)-> 逐条处理:
        带 hidden 行的通告组 UNEMBED 批提交 worker(c2e 全量随批下发)
        取 token ids;纯终结通告(hidden_num_elements == 0)本地终结,
        **不派发 worker**。token 逐条交付(finish_reason=None),仅在
        finished 标记的请求上置 finish——流式/非流式由原生前端透明
        处理(引擎层恒为增量)。

        应答契约(定型):worker 经原生 future 返回 ModelRunnerResult
        形态,token ids 取 lwd_token_ids(request_id -> list[int];缺失/
        None/空 = 该请求 unembed 失败),失败以 FinishReason.ERROR
        终结(原生 ERROR 通道转 5xx),不静默降级空 STOP。
        finished 与 req_ids 对齐透传;缺省空列表 = 兼容"全部完结"
        旧语义。迟到载荷 deliver 返回 False,丢弃告警(幂等);
        UNEMBED 批与 prefill 排程串行,无顺序耦合。
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

        # 带 hidden 行的通告:组批一次提交 worker
        rowed = [n for n in notifies if n.req_ids and n.hidden_num_elements > 0]
        token_map: dict | None = None
        if rowed:
            unembed_batch = lwd_build_unembed_batch(rowed)
            result = self.model_executor.execute_model(unembed_batch).result()
            token_map = getattr(result, "lwd_token_ids", None)
            if token_map is None:
                logger.warning(
                    "[Lwd] unembed batch answer missing lwd_token_ids (%r), "
                    "finishing requests with ERROR",
                    type(result),
                )

        def _lwd_finish_flag(notify: LwdC2eNotify, index: int) -> bool:
            # finished 与 req_ids 对齐;缺省/未对齐回退旧语义(全部完结)
            if len(notify.finished) == len(notify.req_ids):
                return bool(notify.finished[index])
            return True

        for notify in notifies:
            has_rows = bool(notify.req_ids) and notify.hidden_num_elements > 0
            for index, request_id in enumerate(notify.req_ids):
                finished = _lwd_finish_flag(notify, index)
                token_ids: list[int] = []
                if has_rows:
                    token_ids = (
                        list(token_map.get(request_id, [])) if token_map else []
                    )
                    if not token_ids:
                        # 行在批里但无 token = unembed 失败(错误路径)
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
                    # 空 token 且未完结:无内容可交付,跳过(不出空输出)
                    continue
                if token_ids:
                    finish_reason = (
                        FinishReason.STOP if finished else None
                    )
                else:
                    # 纯终结通告(无行)或失败:只发 finish
                    finish_reason = (
                        FinishReason.ERROR if has_rows else FinishReason.STOP
                    )
                outputs.append(
                    EngineCoreOutput(
                        request_id,
                        token_ids,
                        finish_reason=finish_reason,
                    )
                )
                if finished:
                    finished_reqs.add(request_id)
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
