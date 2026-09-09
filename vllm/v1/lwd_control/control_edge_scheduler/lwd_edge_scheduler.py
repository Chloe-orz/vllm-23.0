"""边侧调度器:纯 prefill 调度 + 控制面发布(notify/abort/seqno,§9.10/§9.12)。

原生 AsyncScheduler 的两个前提在边侧不成立:prompt 算完会转 decode、
请求只能由模型输出终结;边侧无本地解码,故需专用调度器。

生命周期(嵌入完 → awaiting → 云结果终结):
  prompt 嵌入完成的当步,update_progress 走原生 finish_requests 清出
  调度器(释放边侧 KV/通知 worker)并登记 _lwd_awaiting;前端未收到
  输出继续等待;终结由云结果(lwd_edge_deliver_tokens,token ids 由
  边侧 worker unembedding 产生)驱动,迟到结果幂等丢弃,awaiting
  超时僵尸兜底(lwd_edge_zombie_check)。

纯 prefill 的实现依据:schedule() 全量复用原生——边侧请求从不产生
输出 token(num_tokens_with_spec 恒等于 num_prompt_tokens),且 prompt
嵌入完成的当步即被 lwd_edge_update_progress 清出调度器,原生 RUNNING 段
每步只会调度剩余 prefill,decode 分支不可达。

单请求组批约束(§9.9 修订):prefill 批最多含一个请求——容器交换
只放行一个 prefill 工作单元(running 尾巴优先,否则 waiting 队首),
原生 schedule() 结构上见不到第二个请求。单请求内的 chunked 决策
(预算截断/KV 抢占)照旧。目的:数据面 chunk 流按请求连续(全局
seqno 链上单请求 chunk 相邻),消除跨请求交错带来的配对/重组复杂度。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
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

_LWD_ADD_RETRY_STEPS = 3
_LWD_ADD_RETRY_INTERVAL_S = 0.1
# awaiting(嵌入完待云结果)僵尸上限:超时本地 abort 兜底(云崩溃/结果丢失)
_LWD_AWAITING_TIMEOUT_S = 300.0


class LwdEdgeScheduler(AsyncScheduler):
    """纯 prefill 语义 + notify/abort/seqno(边侧唯一控制面出口,§9.12)。"""

    def __init__(
        self,
        *args,
        publisher: LwdControlPublisher | None = None,
        **kwargs,
    ) -> None:
        """构造注入 publisher(scheduler_cls 以 partial 携带通道)。"""
        super().__init__(*args, **kwargs)
        self.lwd_edge_publisher = publisher
        self._lwd_seqno = 0
        self._lwd_last_scheduled: dict[str, int] = {}
        # awaiting:嵌入完待云结果的 request_id -> 登记时刻(单调钟);
        # 请求本体已走原生 finish_requests 清出调度器(释放边侧 KV),
        # 前端 OutputProcessor 未收到输出会继续等待 —— 正是 awaiting 语义
        self._lwd_awaiting: dict[str, float] = {}

    def schedule(self) -> SchedulerOutput:
        """单请求组批 + 原生分块决策;记录本步调度量供进度对账(§2.4)。"""
        scheduler_output = self._lwd_schedule_single()
        self._lwd_last_scheduled = dict(scheduler_output.num_scheduled_tokens)
        return scheduler_output

    def _lwd_schedule_single(self) -> SchedulerOutput:
        """单请求组批约束(§9.9 修订):本步 prefill 批最多一个请求。

        容器交换:visible 集只放一个 prefill 工作单元——running 尾巴
        优先(藏其余尾巴与全部 waiting),否则只放行 waiting 队首。
        原生 schedule() 结构上见不到第二个请求,批无法跨请求;单请求
        内 chunked 决策(预算截断/KV 抢占)照旧。藏起的 waiting 走
        队首回插(未调度的 head 与抢占回插者),藏起的尾巴接回
        running 尾部,均保 FIFO。
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
        """边侧请求入口(core.py add_request 守卫的委托点,§9.12)。

        边界校验先于入队:非法请求不得进入调度器;请求预告先行于本地
        登记(源仓 §14.4 同款:云侧视图领先本地工作,只能准备不能计算);
        abort_immediately 走 finish + abort 出口(原生 core.py 路径已被
        守卫旁路,语义等价迁移)。
        """
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
        """对新调度的 prefill 发 LwdRangeNotify(seqno 先行)。

        publish 队满返回 False,调用方本步视为未派发、下一步重试
        (原生 SO 由调度器自然复现,无需回滚;重复预告在云侧按
        (request_id, offset) 幂等登记)。
        """
        publisher = self.lwd_edge_publisher
        if publisher is None:
            logger.error("[Lwd] edge scheduler assembled without publisher")
            return False
        for request_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            request = self.requests.get(request_id)
            if request is None:
                continue
            # _update_after_schedule 已乐观推进 num_computed,起点需回退本步量
            notify = LwdRangeNotify(
                request_id=request_id,
                offset=request.num_computed_tokens - num_tokens,
                num_tokens=num_tokens,
                seqno=self._lwd_edge_next_seqno(),
            )
            if not publisher.publish(notify):
                return False
        return True

    def lwd_edge_notify_request(
        self,
        request_id: str,
        num_prompt_tokens: int,
        max_tokens: int = 16,
        block_hashes: list[bytes] | None = None,
    ) -> None:
        """发 LwdRequestNotify(EngineCore.add_request 守卫的出口,§9.12)。

        block_hashes = 边侧本地请求的 prompt 全量满块链(Request.block_hashes,
        自位置 0 起)—— 云侧 prompt token 是占位零值,靠这条链按真实内容
        命中前缀缓存(§10.13);缺省空 = 不提供,云侧回退本地占位链。
        add 语义不可丢也不可挡本地调度:队满时短退避重试,超限告警放行,
        云侧 zombie 检测兜底(§8.3-2 无自动回压)。
        """
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
        logger.warning(
            "[Lwd] drop add-request notify for %s: publish queue full", request_id
        )

    def lwd_edge_abort(self, request_ids: list[str]) -> None:
        """发 LwdAbortNotify + awaiting 摘除;调度器内清理走原生路径。

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
        """步末对账实际执行量;prompt 全部嵌入完成即转入 awaiting(§9.10)。

        对账基准是本步排程登记(_lwd_last_scheduled)而非 executed:原生
        _update_after_schedule 在调度时已乐观推进 num_computed,步末必须
        回退未派发的部分——notify 队满时 executed 缺项即全量回退(与
        update_from_output 的拒绝回退同款语义),步末后 num_computed ==
        本步实际派发水位,下一步原生调度自然复现同一范围。executed 必须
        是本步排程集的子集(同步执行接缝保证;数据面落位时按 §9.12
        重定义此接缝)。

        嵌入完结 = 边侧工作结束而非请求结束:走原生 finish_requests 做
        全套簿记(移出 running/requests、释放边侧 KV、进 finished_req_ids
        通知 worker 释放缓存——引擎睡眠期该通知滞后到下一个排程步,
        可接受),同时登记 _lwd_awaiting;前端未收到任何输出会继续等待,
        终结由云结果的 deliver 语义驱动(lwd_edge_deliver_tokens)。
        """
        finished_ids: list[str] = []
        for request_id in self._lwd_last_scheduled:
            request = self.requests.get(request_id)
            if request is None:
                # 本步内已终结(abort/更早完成):迟到的对账无对象。
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
        """云结果投递(awaiting 消费点,引擎步内调用;token_ids 由边侧
        worker unembedding 产生,内容本层不消费,仅作生命周期对账)。

        - 请求在 _lwd_awaiting:登记即认领;finished=True 时出 awaiting
          (请求本体已在嵌入完结时清出调度器,无需再 finish);
        - 请求不在(已 abort/更早完结/未知):迟到结果,返回 False 由
          调用方丢弃告警(幂等,不复活)。

        返回是否成功投递到活请求;token_ids/finished 的输出组包归引擎层
        (EngineCoreOutputs 构造,后续 Step)。
        """
        if request_id not in self._lwd_awaiting:
            return False
        if finished:
            del self._lwd_awaiting[request_id]
        return True

    def lwd_edge_zombie_check(self) -> list[str]:
        """awaiting 僵尸检查(引擎步末调用):超时请求本地 abort 兜底。

        云崩溃/add notify 丢失等导致结果永不到达时,awaiting 登记会
        泄漏;超时摘除并告警,由引擎层生成 abort 语义的输出终结前端
        等待(云侧 zombie 检测的同款兜底,方向相反)。
        """
        now = time.monotonic()
        zombie_ids = [
            request_id
            for request_id, since in self._lwd_awaiting.items()
            if now - since > _LWD_AWAITING_TIMEOUT_S
        ]
        for request_id in zombie_ids:
            del self._lwd_awaiting[request_id]
            logger.warning(
                "[Lwd] awaiting request %s timed out after %.0fs, abort locally",
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
            # 最后一段排程时 _update_after_schedule 已为"预期采样输出帧"
            # +1 占位符;边侧无采样,回退到未完成态时一并归零——否则重试
            # 排程再次 +1,原生 running 循环按 num_tokens_with_spec + 占位
            # 符 - num_computed 算出超出 prompt 的幻影 token。边侧从无
            # 输出帧在途,归零即陈述事实。
            request.num_output_placeholders = 0

    @staticmethod
    def _lwd_validate_request(request: Request) -> None:
        """首版边界(源仓 dispatcher._validate 迁移,语义不变,§14.4/§14.10)。

        违规即抛 ValueError:在请求进入调度器之前拒绝(core.py 守卫先
        校验后入队),错误经 add_request 调用链回到客户端的 error 路径。
        """
        if request.prompt_embeds is not None:
            raise ValueError(
                f"[LWD] prefill-only mode does not accept client-provided "
                f"prompt_embeds (request {request.request_id}); the edge "
                "is the embedding owner"
            )
        if not request.prompt_token_ids:
            raise ValueError(
                f"[LWD] prefill-only mode requires a non-empty prompt "
                f"(request {request.request_id})"
            )
        if request.pooling_params is not None:
            raise ValueError(
                "[LWD] prefill-only mode does not support pooling requests "
                f"(request {request.request_id})"
            )
        if request.mm_features:
            raise ValueError(
                "[LWD] prefill-only mode does not support multimodal inputs "
                f"(request {request.request_id})"
            )
        if request.use_structured_output:
            raise ValueError(
                "[LWD] prefill-only mode does not support structured output "
                f"(request {request.request_id})"
            )

    def _lwd_edge_next_seqno(self) -> int:
        """seqno 单调分配;控制面登记与数据面 tag 都由它派生。"""
        current = self._lwd_seqno
        self._lwd_seqno += 1
        return current
