"""边侧引擎子类:run_engine_core 类选择点注入的边 EngineCore。

边引擎职责:通信面装配(边连云 DEALER mesh + 单线程 IO 循环)、
调度器注入(LwdEdgeScheduler)、请求入口/步进/关停的引擎接口覆写。
全部状态收进子类字段,不外挂引擎属性。

构造约束:
  ① 引擎主体先行(super() 最先调):IO 循环回调 WAKEUP 敲门的
     self.input_queue 由 super 创建,后启动循环即无构造期竞态;代价是
     云缺失时引擎主体启动白费,超时路径只清理通信面,主体随进程退出
     兜底回收。通信面随后构建:对每条 dp 级连接开一条 DEALER(带稳定
     identity,端点自拓扑推导)并立即发 register;ack 在云引擎全量初始
     化(权重/KV/图编译)完成后才回——等齐全部连接的 ack 才允许边侧
     对外就绪;超时(hello_timeout_s,默认 600s,须覆盖云全量启动时长)
     fail-fast,未齐的连接逐条报出。register 未确认期间周期重发
     (连接级自愈);云引擎级重启丢调度状态,仍需整组重拉(与现状一致)。
  ② 调度器经 scheduler_cls 注入裸类,须在 super() 之前设值——super 内
     构建调度器时一次性消费该配置,后设无效;引擎构造完成后回填
     IO 循环与 link(早于任何请求,等价构造注入)。

步进编排(embed 优先):步首收割在飞批(unembed 交付 token/请求终结,
embed 登记进度)-> prefill 编排(单请求组批 -> 范围预告 -> executor
异步执行,优先占深度配额)-> 云载荷消费(c2e -> UNEMBED 批,剩余配额)。
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque

import zmq

from vllm.logger import init_logger
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequestType,
    FinishReason,
)
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.lwd_control.control_communication.lwd_control_communicator import (
    LwdControlCommunicator,
)
from vllm.v1.lwd_control.control_communication.lwd_control_loop import (
    LwdControlLoop,
)
from vllm.v1.lwd_debug import LwdDebug
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LWD_NOT_FINISHED,
    LWD_WIRE_VERSION,
    LwdC2eNotify,
    LwdRegisterAckNotify,
    LwdRegisterNotify,
    lwd_decode_cloud_notify,
    lwd_encode_notify,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
    LwdConfig,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
    lwd_build_unembed_batch,
)

logger = init_logger(__name__)


# 云->边载荷队列容量:队满时 IO 循环出站阻塞,背压沿 ZMQ 直达云侧
# 步发送循环(载荷不可丢)
LWD_C2E_META_QUEUE_MAX = 1000

# 批队列在飞深度上限:派发(embed+unembed 合计)不收割的批数上限,
# 突发积压时连续派发喂饱 worker(参照仓经验 >=4 才能维持流水对齐)
LWD_EDGE_BATCH_QUEUE_DEPTH = 4

# register 未确认期间的重发间隔(幂等;云未起或互校拒绝均由重发覆盖)。
# 注意 DEALER 与旧 PUSH 的行为差异:connect 到无人 bind 的端口时,
# DEALER 的 send 静默丢帧(zmq 无可用 pipe)而非排队——本侧靠 register
# 周期重发覆盖"云未起"窗口;请求/预告只在 ack 后发送,连接已建立,
# 帧进入已建 pipe 排队/投递,不丢。
_LWD_REGISTER_RETRY_S = 5.0


class LwdEdgeEngineCore(EngineCoreProc):
    """边 PO 引擎:DEALER mesh 装配 + 调度器注入 + step/add/abort/shutdown 覆写。"""

    def __init__(self, *args, **kwargs) -> None:
        vllm_config = kwargs["vllm_config"]
        # 调度器注入须赶在 super() 之前:super 内构建 self.scheduler 时
        # 一次性消费 scheduler_cls,后设无效(注入失效,首请求即崩)
        vllm_config.scheduler_config.scheduler_cls = LwdEdgeScheduler
        super().__init__(*args, **kwargs)
        config = LwdConfig.from_vllm_config(vllm_config)
        # 通信面:边连云——对每条 dp 级连接一条 DEALER connect(端点自
        # 拓扑推导,带稳定 identity),单线程 IO 循环收发;边不 bind 任何端口
        self._edge_mesh = self._lwd_build_mesh(config)
        # 每条连接的 register-ack 事件:全齐 = 对外就绪;error = 互校失败
        self._lwd_ack: dict[tuple, threading.Event] = {
            link: threading.Event() for link in config.my_links
        }
        self._lwd_ack_error: str | None = None
        self._edge_mesh.start()
        self._lwd_await_registers(config)
        self.scheduler.lwd_edge_publisher = self._edge_mesh
        self.scheduler.lwd_edge_link = config.my_links[0]
        logger.info(
            "[Lwd] edge engine assembled: %d DEALER link(s) registered, "
            "identity=%r",
            len(config.my_links), config.dealer_identity,
        )

    # ------------------------------------------------------------------ #
    # 通信面                                                              #
    # ------------------------------------------------------------------ #
    def _lwd_build_mesh(self, config: LwdConfig) -> LwdControlLoop:
        """边侧 DEALER mesh:per-link 一条 DEALER connect(identity 稳定,
        断线重连后云侧按 identity 认回),单线程 IO 循环驱动收发。"""
        dealers = {
            link: LwdControlCommunicator(
                endpoint, zmq.DEALER, bind=False, identity=config.dealer_identity
            )
            for link, endpoint in config.dealer_endpoints.items()
        }
        return LwdControlLoop(
            dealers,
            decoder=lwd_decode_cloud_notify,
            encoder=lwd_encode_notify,
            on_msg=self._lwd_on_cloud_msg,
        )

    def _lwd_await_registers(self, config: LwdConfig) -> None:
        """发 register 并等齐全部连接的 ack:互校失败或超时 fail-fast,
        报出未确认的连接(三元组)而非笼统超时。"""
        registers = {
            link: LwdRegisterNotify(
                edge_id=link[0],
                cloud_id=link[1],
                dp_idx=link[2],
                wire_version=LWD_WIRE_VERSION,
                edge_npu_count=config.edge_npu_count,
                cloud_npu_count=config.cloud_npu_count,
                topology_digest=config.topology_digest,
            )
            for link in config.my_links
        }
        deadline = time.monotonic() + config.hello_timeout_s
        next_retry = 0.0
        while True:
            if self._lwd_ack_error is not None:
                self._lwd_shutdown_planes()
                raise RuntimeError(self._lwd_ack_error)
            unacked = [link for link, event in self._lwd_ack.items()
                       if not event.is_set()]
            if not unacked:
                return
            now = time.monotonic()
            if now >= deadline:
                self._lwd_shutdown_planes()
                raise RuntimeError(
                    f"[Lwd] edge engine init failed: no register-ack within "
                    f"{config.hello_timeout_s}s for links {unacked} "
                    f"(identity={config.dealer_identity!r}; check cloud "
                    "ctrl_port reachability and topology identity on both "
                    "sides)"
                )
            if now >= next_retry:
                for link in unacked:
                    self._edge_mesh.send(link, registers[link])
                next_retry = now + _LWD_REGISTER_RETRY_S
            threading.Event().wait(0.05)

    def _lwd_on_cloud_msg(self, link, _identity, msg) -> None:
        """云->边入站分发(IO 线程回调)。

        RegisterAck -> 置对应连接的就绪事件(ack 在云引擎全量初始化
        完成后回,时序语义等价旧 HELLO 首拍)。
        LwdC2eNotify(云->边唯一载荷)-> 载荷队列(阻塞 put,不可丢)
        + WAKEUP 唤醒主循环:prefill 全部完成后请求转入 awaiting,
        引擎无排程工作、阻塞在 input_queue.get(),载荷只进队列不会
        唤醒任何线程,必须向 input_queue 敲门;WAKEUP 原生语义即丢弃
        消息体,数据与唤醒分离,多投无害(空 drain 一步即返回)。
        其余帧(坏帧已被 IO 循环丢弃后仍不认识的类型)告警丢弃。"""
        if isinstance(msg, LwdRegisterAckNotify):
            if msg.wire_version != LWD_WIRE_VERSION:
                # ack 版本不符:置错唤醒构造线程 fail-fast(回调内 raise
                # 只杀 IO 线程,构不成快败);计数/digest 校验在云侧
                # register 入口已做,ack 只核版本
                error = (
                    f"[Lwd] edge engine init failed: register-ack wire "
                    f"version mismatch (cloud={msg.wire_version}, "
                    f"edge={LWD_WIRE_VERSION})"
                )
                logger.error("[Lwd] %s", error)
                self._lwd_ack_error = error
                for event in self._lwd_ack.values():
                    event.set()
                return
            expected = self._lwd_ack.get(link)
            if expected is not None:
                expected.set()
            return
        if isinstance(msg, LwdC2eNotify):
            logger.info(
                "[Lwd][edge-ctrl] C2eNotify reqs=%d down_seqno=%s edge=%s dp=%s",
                len(getattr(msg, "req_ids", []) or []),
                getattr(msg, "down_seqno", None),
                getattr(msg, "edge_id", None),
                getattr(msg, "dp_idx", None),
            )
            # 队列元素带到达时间戳:消费滞后(c2e_wait=到达→派发)
            # 是"边消费不动 → 云出站队满小睡"闭环的前置指标
            self.lwd_c2e_meta_queue.put((msg, time.monotonic()))
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
            return
        logger.warning("[Lwd] drop unexpected cloud frame %r", type(msg))

    def _lwd_shutdown_planes(self) -> None:
        """通信面关停(幂等):IO 循环收尾(其内统一关闭 DEALER/唤醒管道)。"""
        mesh = getattr(self, "_edge_mesh", None)
        if mesh is not None:
            mesh.stop()

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
        return (super().has_work() or not self.lwd_c2e_meta_queue.empty()
                or self._lwd_batch_queue)

    def _lwd_edge_step(self) -> tuple[dict[int, object] | None, bool]:
        """单步编排(批队列异步化+多批在飞,embed 优先):收割队首直至
        清空 -> prefill 连续派发(优先占深度配额) -> 消费云载荷(剩余
        配额派发 UNEMBED)-> 返回({0: 云侧结果输出} | None,prefill
        是否有派发)。

        embed/unembed 均派发入队不收割,队首全局 FIFO 收割(序与
        executor 排水序同构,不可交错);突发时一步连派至深度上限
        喂饱 worker。派发序 embed 优先:196 并发下每云步都产 c2e,
        若 unembed 先吃配额会把 embed 挤出本步(EMBED 被 decode 回程
        挤住的饿死形态),TTFT 劣化——故 embed 先占位,unembed 用剩余
        配额,代价是 decode token 交付最多延后 1-2 步(顶部全量收割
        + c2e 队列背压兜底,不会饿死)。

        [Lwd][sched] 三类日志(饿死分析锚点):每批 harvest(含队列滞留
        wait)、每批 dispatch(含 ahead 队列构成/c2e 水位)、每步汇总。"""
        outputs: list = []
        finished_reqs: set = set()
        prefill_work = False
        h_emb = h_unemb = 0
        while self._lwd_batch_queue:
            kind, payload, t_dispatch, future = self._lwd_batch_queue.popleft()
            if kind == "unembed":
                self._lwd_deliver_unembed(
                    payload, future, outputs, finished_reqs
                )
                h_unemb += 1
            else:
                self.scheduler.lwd_edge_update_progress(
                    dict(payload.num_scheduled_tokens)
                )
                prefill_work = True
                h_emb += 1
            logger.info(
                "[Lwd][sched] edge harvest kind=%s seqno=%s wait=%.2fms",
                kind, self._lwd_batch_seqno(kind, payload),
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
        d_unemb = self._lwd_edge_consume_c2e()
        n_emb, n_unemb = self._lwd_queue_mix()
        logger.info(
            "[Lwd][sched] edge step harvest_emb=%d harvest_unemb=%d "
            "disp_emb=%d disp_unemb=%d queue=%demb/%dunemb c2e_pending=%d",
            h_emb, h_unemb, d_emb, d_unemb, n_emb, n_unemb,
            self.lwd_c2e_meta_queue.qsize(),
        )
        if outputs:
            step_outputs = EngineCoreOutputs(
                outputs=outputs,
                finished_requests=finished_reqs or None,
            )
            return {0: step_outputs}, prefill_work
        return None, prefill_work

    def _lwd_queue_mix(self) -> tuple[int, int]:
        """批队列构成 (embed数, unembed数):dispatch 日志的 ahead_unemb
        = EMBED 批前面压着的 UNEMBED 批数,是"EMBED 被 decode 回程挤住"
        (饿死假说)的直接读数。"""
        n_emb = sum(1 for e in self._lwd_batch_queue if e[0] == "embed")
        return n_emb, len(self._lwd_batch_queue) - n_emb

    @staticmethod
    def _lwd_batch_seqno(kind: str, payload) -> str:
        """harvest 日志的配对号:unembed 用 down_seqno(与云 DOWN 同号),
        embed 用批 seqno(与 RangeNotify/UP 同号)。"""
        if kind == "unembed":
            return str(getattr(payload, "down_seqno", "?"))
        batch = getattr(payload, "lwd_batch", None)
        return str(getattr(batch, "seqno", "?")) if batch else "?"

    def _lwd_edge_consume_c2e(self) -> int:
        """消费 c2e 通告:派发 UNEMBED 批入批队列(异步收割)。
        云侧有活请求才产 meta(build_hidden_payload 空则返回 None),
        每条 entry 行数>=1——通告必有行,无需无行分流。
        返回本步派发条数。"""
        quota = LWD_EDGE_BATCH_QUEUE_DEPTH - len(self._lwd_batch_queue)
        notifies: list[tuple[LwdC2eNotify, float]] = []
        while len(notifies) < quota:
            try:
                notifies.append(self.lwd_c2e_meta_queue.get_nowait())
            except queue.Empty:
                break
        for notify, t_arrive in notifies:
            future = self._lwd_dispatch_unembed(notify, t_arrive)
            self._lwd_batch_queue.append(
                ("unembed", notify, time.monotonic(), future)
            )
        return len(notifies)

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
        n_emb, n_unemb = self._lwd_queue_mix()
        logger.info(
            "[Lwd][sched] edge dispatch-embed seqno=%d req=%s tokens=%d "
            "ahead_unemb=%d ahead_emb=%d c2e_pending=%d",
            batch.seqno, batch.batch_meta.req_ids[0],
            scheduler_output.total_num_scheduled_tokens,
            n_unemb, n_emb, self.lwd_c2e_meta_queue.qsize(),
        )
        return self.model_executor.execute_model(
            scheduler_output, non_block=True
        )

    def _lwd_dispatch_unembed(self, notify: LwdC2eNotify, t_arrive: float):
        """UNEMBED 批提交侧:组批并以 non_block 提交 worker,立即返回
        future(不等执行)。同步路径提交后立即收割;异步路径
        (batch_queue)将 (notify, future) 入队,收割阶段再等 future——
        提交与等待的边界即本函数返回处。

        c2e_wait(到达→派发)是消费滞后读数:持续偏大说明边引擎步
        循环被收割/派发占住,c2e 在积压,云侧 publisher 队满小睡在即。"""
        unembed_batch = lwd_build_unembed_batch(notify)
        n_emb, n_unemb = self._lwd_queue_mix()
        logger.info(
            "[Lwd][sched] edge dispatch-unembed seqno=%s reqs=%d rows=%d "
            "ahead_unemb=%d ahead_emb=%d c2e_wait=%.2fms c2e_pending=%d",
            notify.down_seqno, len(notify.req_ids),
            sum(notify.num_accepted_tokens), n_unemb, n_emb,
            (time.monotonic() - t_arrive) * 1000,
            self.lwd_c2e_meta_queue.qsize(),
        )
        return self.model_executor.execute_model(
            unembed_batch, non_block=True
        )

    def _lwd_deliver_unembed(
        self, notify: LwdC2eNotify, future, outputs: list, finished_reqs: set,
    ) -> None:
        """UNEMBED 批收割侧:等 worker 应答取采样 token 表,逐请求交付。
        不感知提交时机,只消费 (notify, future)。应答契约:
        token ids 按 ModelRunnerOutput 的 req_ids x sampled_token_ids
        按位对齐还原(批的 req_ids 原样下发,worker 逐位回填);请求
        缺席或行无 token = unembed 失败,ERROR 优先于云侧完成码;
        迟到载荷幂等丢弃。"""
        _t = time.monotonic()
        result = future.result()
        # [Lwd][perf] 临时探针:收割时长 = RPC 往返 + worker 执行全长
        # (与 worker 侧 [Lwd][perf] unembed 分段对账,差值即进程往返开销)
        logger.info(
            "[Lwd][perf] harvest dur=%.2fms", (time.monotonic() - _t) * 1000
        )
        # req_ids x sampled_token_ids 按位对齐:worker lm_head 恢复的采样
        # token,即该请求本步的生成内容;后续仅两处流向——
        # lwd_edge_deliver_tokens(调度器只对账 awaiting 生命周期,
        # 不消费内容)与 EngineCoreOutput 的 new_token_ids(outputs ->
        # EngineCoreOutputs -> 主循环按 frontend 消费 -> 前端解流交付
        # 客户端,即最终输出的生成 token)。
        sampled_token_map: dict[str, list[int]] = (
            {} if result is None
            else dict(zip(result.req_ids, result.sampled_token_ids))
        )
        for index, request_id in enumerate(notify.req_ids):
            finish_reason = self._lwd_finish_code(notify, index)
            finished = finish_reason is not None
            sampled_token_ids = (
                list(sampled_token_map.get(request_id, []))
                if sampled_token_map else []
            )
            LwdDebug.edge_tokens_delivered(  # [lwd-debug]
                request_id, sampled_token_ids, finish_reason,
                self.vllm_config,
            )
            if not sampled_token_ids:
                # 行在批里但无 token = unembed 失败,ERROR 优先于云侧码
                finish_reason = FinishReason.ERROR
                finished = True
            if not self.scheduler.lwd_edge_deliver_tokens(
                request_id, sampled_token_ids, finished=finished
            ):
                logger.warning(
                    "[Lwd] drop stale cloud payload for %s (not awaiting)",
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
        """两面关停后走原生;初始化失败路径两面可能未建,容忍缺省。"""
        self._lwd_shutdown_planes()
        super().shutdown()
