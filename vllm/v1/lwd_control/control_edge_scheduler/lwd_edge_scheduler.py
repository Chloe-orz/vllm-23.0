"""边侧调度器:纯 prefill 调度 + 控制面发布(notify/abort/seqno,§9.10/§9.12)。

原生 AsyncScheduler 的两个前提在边侧不成立:prompt 算完会转 decode、
请求只能由模型输出终结;边侧无本地解码,故需专用调度器。

纯 prefill 的实现依据:schedule() 全量复用原生——边侧请求从不产生
输出 token(num_tokens_with_spec 恒等于 num_prompt_tokens),且 prompt
嵌入完成的当步即被 lwd_edge_update_progress 本地终结,原生 RUNNING 段
每步只会调度剩余 prefill,decode 分支不可达。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
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

logger = init_logger(__name__)

_LWD_ADD_RETRY_STEPS = 3
_LWD_ADD_RETRY_INTERVAL_S = 0.1


class LwdEdgeScheduler(AsyncScheduler):
    """纯 prefill 语义 + notify/abort/seqno(边侧唯一控制面出口,§9.12)。"""

    def __init__(
        self,
        *args,
        publisher: LwdControlPublisher | None = None,
        **kwargs,
    ) -> None:
        """publisher 经装配期注入(scheduler_cls 以 partial 携带通道)。"""
        super().__init__(*args, **kwargs)
        self.lwd_edge_publisher = publisher
        self._lwd_seqno = 0
        self._lwd_last_scheduled: dict[str, int] = {}

    def schedule(self) -> SchedulerOutput:
        """全量复用原生分块决策;记录本步调度量供进度对账(§2.4 未派发重试)。"""
        scheduler_output = super().schedule()
        self._lwd_last_scheduled = dict(scheduler_output.num_scheduled_tokens)
        return scheduler_output

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
        """发 LwdAbortNotify;本地终结走原生 abort 路径,此处只管出口。"""
        publisher = self.lwd_edge_publisher
        if publisher is None:
            return
        for request_id in request_ids:
            if not publisher.publish(LwdAbortNotify(request_id=request_id)):
                logger.warning(
                    "[Lwd] drop abort signal for %s: publish queue full", request_id
                )

    def lwd_edge_update_progress(self, executed: dict[str, int]) -> None:
        """步末对账实际执行量;prompt 全部嵌入完成即本地终结(§9.10)。

        对账基准是本步排程登记(_lwd_last_scheduled)而非 executed:原生
        _update_after_schedule 在调度时已乐观推进 num_computed,步末必须
        回退未派发的部分——notify 队满时 executed 缺项即全量回退(与
        update_from_output 的拒绝回退同款语义),步末后 num_computed ==
        本步实际派发水位,下一步原生调度自然复现同一范围。executed 必须
        是本步排程集的子集(同步执行接缝保证;数据面落位时按 §9.12
        重定义此接缝)。终结不依赖任何模型输出,这是与原生路径的唯一
        语义差。
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
        self._lwd_last_scheduled = {}

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

    def _lwd_edge_next_seqno(self) -> int:
        """seqno 单调分配;控制面登记与数据面 tag 都由它派生。"""
        current = self._lwd_seqno
        self._lwd_seqno += 1
        return current
