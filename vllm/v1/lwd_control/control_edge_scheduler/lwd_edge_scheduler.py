"""边侧调度器:纯 prefill 调度 + 控制面发布(notify/abort/seqno)。

原生 AsyncScheduler 的两个前提在边侧不成立:prompt 算完会转 decode、
请求只能由模型输出终结。边侧只做 embedding(执行层无 decode),故需
专用调度器接管请求的边侧生命周期:

  入队 -> EMBEDDING(单请求组批/chunked 决策/范围预告)
       -> 嵌入完结:走原生 finish_requests 清出调度器(释放边侧 KV
          簿记、通知 worker 释放缓存),登记 awaiting
       -> AWAITING(等待云结果;前端未收到输出继续等待)
       -> 云结果终结(lwd_edge_deliver_tokens);迟到结果幂等丢弃;
          awaiting 超时僵尸兜底(lwd_edge_zombie_check)。

纯 prefill 的实现依据:schedule() 全量复用原生——边侧请求从不产生
输出 token(num_tokens_with_spec 恒等于 num_prompt_tokens),且嵌入
完结当步即被清出调度器,原生 RUNNING 段每步只会调度剩余 prefill,
decode 分支不可达。

单请求组批约束:prefill 批最多含一个请求。目的:数据面 chunk 流按
请求连续(全局 seqno 链上单请求的 chunk 相邻),消除跨请求交错带来
的张量配对/重组复杂度。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import (
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    LwdUnembedBatch,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LwdAbortNotify,
    LwdRangeNotify,
    LwdRequestNotify,
)
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
        LwdControlPublisher,
    )
    from vllm.v1.request import Request

logger = init_logger(__name__)

# add 预告发布重试:次数 x 递增间隔(共约 3s),耗尽即请求级报错
_LWD_ADD_RETRY_STEPS = 5
_LWD_ADD_RETRY_INTERVAL_S = 0.2
# awaiting 僵尸上限:超时本地 abort + 通知云停止(云崩溃/结果丢失兜底)
_LWD_AWAITING_TIMEOUT_S = 300.0


class LwdEdgeScheduler(AsyncScheduler):
    """纯 prefill 调度语义 + 控制面出口(notify/abort/seqno)。"""

    def __init__(
        self,
        *args,
        publisher: LwdControlPublisher | None = None,
        **kwargs,
    ) -> None:
        """publisher 经构造注入(与调度器同生命周期)。"""
        super().__init__(*args, **kwargs)
        self.lwd_edge_publisher = publisher
        self._lwd_seqno = 0
        self._lwd_last_scheduled: dict[str, int] = {}
        # 前缀缓存:manager 级关命中,配置级保留使能。两级拆分的原因:
        # - 必须关命中:命中会跳过 token 排程,首条 RangeNotify 的
        #   offset != 0,云侧按 offset==0 识别首块的约定失效;且被
        #   "命中"的块在 EMBED 批下从未写入 KV(边侧不落 KV),
        #   缓存登记残留会持续占用块池。关闭后命中恒空、free 直接
        #   归还块池。
        # - 不能在配置级关:request_block_hasher 只在配置级使能时
        #   创建,关掉则 Request.block_hashes 恒空,LwdRequestNotify
        #   带不出哈希链——云侧 prompt 是占位零值 token,前缀缓存
        #   只能靠边侧按真实内容算出的哈希链命中。
        self.kv_cache_manager.enable_caching = False
        # awaiting:嵌入完待云结果的 request_id -> 登记时刻(单调钟)。
        # 请求本体已清出调度器,此表是结果路径的唯一生命周期台账。
        self._lwd_awaiting: dict[str, float] = {}

    def schedule(self) -> SchedulerOutput:
        """单请求组批 + 原生分块决策;记录本步调度量供步末对账。

        EMBED 批的 LwdBatch(seqno/token 片段)由 lwd_edge_notify 在
        发布成功后挂批——seqno 必须与发布成功绑定,发布失败整步
        回退时批随 SchedulerOutput 一并废弃。"""
        scheduler_output = self._lwd_schedule_single()
        self._lwd_last_scheduled = dict(scheduler_output.num_scheduled_tokens)
        return scheduler_output

    def _lwd_schedule_single(self) -> SchedulerOutput:
        """单请求组批:容器交换让原生 schedule() 只见到一个工作单元。

        可见集只放一个 prefill 工作单元——running 尾巴优先(藏其余
        尾巴与全部 waiting),否则只放行 waiting 队首;原生 schedule()
        结构上见不到第二个请求,批无法跨请求,单请求内的 chunked
        决策(预算截断/KV 抢占)照旧。藏起的 waiting 走队首回插
        (含被抢占回插者),藏起的尾巴接回 running 尾部,均保 FIFO。
        """
        tail = next(
            (r for r in self.running if r.num_computed_tokens < r.num_prompt_tokens),
            None,
        )
        hidden_waiting = self.waiting
        self.waiting = create_request_queue(self.policy)
        hidden_tails: list = []
        if tail is not None:
            hidden_tails = [
                r
                for r in self.running
                if r is not tail and r.num_computed_tokens < r.num_prompt_tokens
            ]
            if hidden_tails:
                self.running = [
                    r
                    for r in self.running
                    if r is tail or r.num_computed_tokens >= r.num_prompt_tokens
                ]
        elif hidden_waiting:
            self.waiting.add_request(hidden_waiting.peek_request())
        try:
            return super().schedule()
        finally:
            leftover = self.waiting
            self.waiting = hidden_waiting
            while leftover:
                self.waiting.prepend_request(leftover.pop_request())
            if hidden_tails:
                self.running = self.running + hidden_tails

    def lwd_edge_add_request(self, request: Request) -> None:
        """请求入口:边界校验 -> 云预告 -> 本地入队。

        预告先于本地登记:云侧视图领先本地工作,云只能准备不能提前
        计算(没有 chunk 预告就不会开算)。abort_immediately 请求走
        finish + abort 出口,与原生语义一致。"""
        self._lwd_validate_request(request)
        sampling_params = request.sampling_params
        self.lwd_edge_notify_request(
            request_id=request.request_id,
            num_prompt_tokens=len(request.prompt_token_ids),
            max_tokens=(
                sampling_params.max_tokens if sampling_params is not None else 16
            ),
            block_hashes=list(request.block_hashes),
        )
        super().add_request(request)
        if request.abort_immediately:
            self.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)
            self.lwd_edge_abort([request.request_id])

    def lwd_edge_notify(self, scheduler_output: SchedulerOutput) -> bool:
        """对新调度的 prefill 块发 LwdRangeNotify(seqno 先行)。

        seqno 无空洞契约:UP 数据通道按连续号序配对(通道层对超前号
        扣留等待,一个空洞即永久挂死整条链),因此号只能分配给真正
        上 wire 的块:
        - peek-then-advance:发布成功才进位计数器,队满回退时号未
          消耗,重试复用同一号;
        - 单 notify 前提:本方法每步至多发一条,依赖单请求组批
          约束。若放开多请求组批,部分成功的 notify 已上 wire 而
          整步回退不发张量,会同时产生号空洞与张量失配,届时必须
          改为按已成功子集执行。

        发布队满返回 False:调用方本步不派发,下一步原生调度自然
        复现同一范围重试;重复预告在云侧按 (request_id, offset)
        幂等登记。"""
        publisher = self.lwd_edge_publisher
        if publisher is None:
            logger.error("[Lwd] edge scheduler assembled without publisher")
            return False
        scheduled = scheduler_output.num_scheduled_tokens
        assert len(scheduled) <= 1, (
            "[Lwd] single-request batch invariant violated: seqno hole risk"
        )
        for request_id, num_tokens in scheduled.items():
            request = self.requests.get(request_id)
            if request is None:
                continue
            # _update_after_schedule 已乐观推进 num_computed,起点需回退本步量
            offset = request.num_computed_tokens - num_tokens
            seqno = self._lwd_seqno
            if not publisher.publish(
                LwdRangeNotify(
                    request_id=request_id,
                    offset=offset,
                    num_tokens=num_tokens,
                    seqno=seqno,
                )
            ):
                return False
            self._lwd_seqno = seqno + 1
            # 发布成功即组 EMBED 批挂 SO:seqno 是数据面发云张量的
            # 配对键(与云侧 RangeNotify 登记同值),embed 载荷为本
            # chunk 的 token 片段
            scheduler_output.lwd_batch = LwdBatch(
                batch_type=LwdBatchType.LWD_EMBED,
                seqno=seqno,
                batch_meta=LwdEmbedBatch(
                    req_ids=[request_id],
                    token_ids=[
                        list(
                            request.prompt_token_ids[offset : offset + num_tokens]
                        )
                    ],
                ),
            )
        return True

    def lwd_edge_notify_request(
        self,
        request_id: str,
        num_prompt_tokens: int,
        max_tokens: int = 16,
        block_hashes: list[bytes] | None = None,
    ) -> None:
        """发 LwdRequestNotify(请求元数据预告)。

        block_hashes = prompt 全量满块哈希链(自位置 0 起)。云侧
        prompt token 是占位零值,本地算不出真实内容哈希,前缀缓存
        命中只能靠这条链;缺省空链 = 不提供,云侧回退占位链。

        失败语义 fail-fast:发布队满时短退避重试(瞬态背压几乎必在
        秒级窗口内腾出),耗尽即抛 RuntimeError——异常沿 add_request
        调用链回前端 error 通道,用户立即得到失败;不做无限阻塞
        重试(发布点在引擎主线程,云宕机会把整个引擎卡死在 add)。
        此时请求未入队、云侧零残留,无需补发 abort;已入队请求的
        云宕机由 awaiting 僵尸超时兜底。"""
        publisher = self.lwd_edge_publisher
        if publisher is None:
            return
        message = LwdRequestNotify(
            request_id=request_id,
            num_prompt_tokens=num_prompt_tokens,
            max_tokens=max_tokens,
            block_hashes=block_hashes if block_hashes is not None else [],
        )
        for attempt in range(_LWD_ADD_RETRY_STEPS):
            if publisher.publish(message):
                return
            time.sleep(_LWD_ADD_RETRY_INTERVAL_S * (attempt + 1))
        raise RuntimeError(
            f"[LWD] add-request notify for {request_id} dropped: publish "
            f"queue full after {_LWD_ADD_RETRY_STEPS} retries "
            f"(cloud PRE_OUT consumption stalled?)"
        )

    def lwd_edge_abort(self, request_ids: list[str]) -> None:
        """发 LwdAbortNotify + 摘除 awaiting;调度器内清理走原生路径。

        awaiting 请求已不在调度器视野(嵌入完结时清出),原生
        finish_requests 触不到它,须在此显式摘除,否则僵尸检查误报。"""
        publisher = self.lwd_edge_publisher
        for request_id in request_ids:
            self._lwd_awaiting.pop(request_id, None)
            if publisher is None:
                continue
            if not publisher.publish(LwdAbortNotify(request_id=request_id)):
                logger.warning(
                    "[Lwd] drop abort signal for %s: publish queue full", request_id
                )

    def lwd_edge_update_progress(self, executed: dict[str, int]) -> None:
        """步末对账实际执行量;嵌入完结即转入 awaiting。

        对账基准是本步排程登记而非 executed:原生 _update_after_schedule
        在调度时已乐观推进 num_computed,步末必须回退未派发的部分——
        预告失败时 executed 缺项即全量回退,步末后 num_computed == 本步
        实际派发水位,下一步原生调度自然复现同一范围。executed 必须是
        本步排程集的子集(同步执行保证)。

        嵌入完结 = 边侧工作结束而非请求结束:走原生 finish_requests
        做全套簿记(移出 running/requests、释放边侧 KV、进
        finished_req_ids 通知 worker 释放缓存——该通知随下一个排程步
        下发,引擎纯睡眠期滞后),同时登记 awaiting;前端未收到任何
        输出会继续等待,终结由云结果驱动(lwd_edge_deliver_tokens)。"""
        finished_ids: list[str] = []
        for request_id in self._lwd_last_scheduled:
            request = self.requests.get(request_id)
            if request is None:
                # 本步内已终结(abort/更早完成):迟到的对账无对象
                continue
            num_executed = executed.get(request_id, 0)
            self._lwd_reconcile_progress(request, request_id, num_executed)
            if request.num_computed_tokens >= request.num_prompt_tokens:
                finished_ids.append(request_id)
        if finished_ids:
            self.finish_requests(finished_ids, RequestStatus.FINISHED_STOPPED)
            now = time.monotonic()
            for request_id in finished_ids:
                self._lwd_awaiting[request_id] = now
        self._lwd_last_scheduled = {}

    def lwd_edge_deliver_tokens(
        self, request_id: str, token_ids: list[int], finished: bool
    ) -> bool:
        """云结果投递(awaiting 消费点,引擎步内调用)。

        token_ids 由边侧 worker unembedding 产生,本层不消费内容,
        仅做生命周期对账:
        - 请求在 awaiting:认领;finished=True 时出 awaiting(请求本体
          已在嵌入完结时清出调度器,无需再 finish);
        - 请求不在(已 abort/更早完结/未知):迟到结果,返回 False,
          调用方丢弃告警(幂等,不复活)。

        token_ids/finished 的输出组包归引擎层(EngineCoreOutputs)。"""
        if request_id not in self._lwd_awaiting:
            return False
        if finished:
            del self._lwd_awaiting[request_id]
        return True

    def lwd_edge_zombie_check(self) -> list[str]:
        """awaiting 僵尸检查(引擎步末调用),返回超时请求列表。

        云崩溃/结果丢失导致结果永不到达时,awaiting 登记会泄漏且
        前端永久等待。超时请求由引擎层生成 ABORT 输出终结前端;
        出口复用 lwd_edge_abort:向云发 LwdAbortNotify(云停止该请求
        后续 c2e 与 decode,不白算)+ 摘除 awaiting(幂等)。"""
        now = time.monotonic()
        zombie_ids = [
            request_id
            for request_id, since in self._lwd_awaiting.items()
            if now - since > _LWD_AWAITING_TIMEOUT_S
        ]
        if zombie_ids:
            self.lwd_edge_abort(zombie_ids)
            for request_id in zombie_ids:
                logger.warning(
                    "[Lwd] awaiting request %s timed out after %.0fs, "
                    "abort locally + notify cloud",
                    request_id,
                    _LWD_AWAITING_TIMEOUT_S,
                )
        return zombie_ids

    def _lwd_reconcile_progress(
        self, request, request_id: str, num_executed: int
    ) -> None:
        """回退本步未执行量;未派发重试(num_executed=0)即全量回退。"""
        num_undone = self._lwd_last_scheduled.get(request_id, 0) - num_executed
        if num_undone > 0:
            request.num_computed_tokens -= num_undone
            request.is_prefill_chunk = True
            # 原生调度最后一段排程时会为"预期采样输出帧"+1 占位符;
            # 边侧无采样,回退未完成态时须一并归零,否则重试排程再次
            # +1,原生 running 循环按 num_tokens_with_spec - num_computed
            # 会算出超出 prompt 的幻影 token。边侧从无输出帧在途,
            # 归零即陈述事实。
            request.num_output_placeholders = 0

    @staticmethod
    def _lwd_validate_request(request: Request) -> None:
        """模式边界校验;违规抛 ValueError,在入队之前拒绝,错误经
        add_request 调用链回到客户端 error 路径。

        边界 = 边侧能力面:边是 embedding 属主(拒绝客户端自带
        prompt_embeds),只处理纯文本补全(拒 pooling/多模态/结构化
        输出),prompt 非空。"""
        if request.prompt_embeds is not None:
            raise ValueError(
                f"[LWD] prefill-only mode does not accept client-provided "
                f"prompt_embeds (request {request.request_id}); the edge "
                "is the embedding owner"
            )
        if request.pooling_params is not None:
            raise ValueError(
                "[LWD] prefill-only mode does not support pooling requests "
                f"(request {request.request_id})"
            )
        if request.use_structured_output:
            raise ValueError(
                "[LWD] prefill-only mode does not support structured output "
                f"(request {request.request_id})"
            )

def lwd_build_unembed_batch(notifies: list, seqno: int) -> SchedulerOutput:
    """组 UNEMBED 批(引擎步内调用,云载荷派发给边 worker 做 lm_head)。

    云结果不经过原生 schedule,无原生排程产物可用——以 make_empty
    为骨架:
    - lwd_batch 携带 LwdUnembedBatch:req_ids(隐藏行序)/
      num_accept_tokens/top_id_ths 逐请求透传自 c2e;recv_num_elements
      (DOWN 通道每请求接收元素数)与 out_token_idxs(生成序号)控制面
      不可知,留空由数据面按 DOWN 张量实收推导;
    - 请求集合同步镜像到 num_scheduled_tokens(值 1 = 单 token 位,
      不作 token 预算解释,保持原生管道字段完整性);
    - 触发本批的 LwdC2eNotify 全量挂 lwd_c2e_notifies 动态属性:
      数据面据 hidden_num_elements 对齐 DOWN 张量(元数据先于张量
      到达的预挂契约)并读 finished 完结标志。
    """
    scheduler_output = SchedulerOutput.make_empty()
    req_ids = [rid for notify in notifies for rid in notify.req_ids]
    scheduler_output.num_scheduled_tokens = {rid: 1 for rid in req_ids}
    scheduler_output.total_num_scheduled_tokens = len(req_ids)
    scheduler_output.lwd_batch = LwdBatch(
        batch_type=LwdBatchType.LWD_UNEMBED,
        seqno=seqno,
        batch_meta=LwdUnembedBatch(
            req_ids=req_ids,
            num_accept_tokens=[
                n for notify in notifies for n in notify.num_accepted_tokens
            ],
            recv_num_elements=[],
            out_token_idxs=[],
            top_id_ths=[t for notify in notifies for t in notify.top_id_ths],
        ),
    )
    scheduler_output.lwd_c2e_notifies = list(notifies)
    return scheduler_output
