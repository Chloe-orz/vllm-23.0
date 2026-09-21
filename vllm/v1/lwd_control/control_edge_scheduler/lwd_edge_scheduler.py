"""边侧调度器:prefill 相位发云(embedding),decode 相位收云(unembed)。

请求全生命周期留在原生记账内:嵌入完结后保持 decode 相位留在 running
等云侧通告;unembed 行数以占位欠条表达,真实 token 经原生
update_from_output 入账、判停、终结。prefill 批恒单请求(数据面 chunk
流按请求连续)。
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.sched.output import (
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    LwdUnembedBatch,
    SchedulerOutput,
)
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LWD_NOT_FINISHED,
    LwdC2eNotify,
    LwdRangeNotify,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
    LwdReqPhase,
)
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

# 云侧完成码 -> 边侧内部终态(前端拿到的 reason 走输出通道)
_LWD_FINISH_STATUS = {
    FinishReason.LENGTH: RequestStatus.FINISHED_LENGTH_CAPPED,
    FinishReason.ABORT: RequestStatus.FINISHED_ABORTED,
    FinishReason.ERROR: RequestStatus.FINISHED_ABORTED,
}


class LwdEdgeScheduler(LwdBaseScheduler):
    """prompt 发云 / 输出收云的纯相位调度 + 控制面出口(notify/abort/seqno)。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 发布面由引擎装配后回填(RangeNotify 出口)
        self.lwd_publisher = None
        self._lwd_seqno = 0
        # 命中必须关(命中会跳过 token 排程,首条 RangeNotify 的
        # offset != 0,云侧首块识别失效);配置级保留使能以产出哈希链
        # (云侧占位 prompt 的前缀缓存只能靠边侧哈希链命中)
        self.kv_cache_manager.enable_caching = False
        # unembed 通告队列:接收线程 append / 主步 popleft,一步一条
        self.unembed_notify_queue: deque[LwdC2eNotify] = deque()

    def _lwd_select_phase(self) -> LwdReqPhase | None:
        """embed 优先:有 prefill 活走 prefill,否则看 unembed 通告,
        皆无返回 None。不做禁连续 prefill——边侧 decode 活要等全部
        chunk 发完云侧才产,不变量会拦死 chunk 发送(单请求死锁)。"""
        # prefill 活 = running 续传(不受水位约束,先收尾)或水位有余且
        # waiting 非空(decode 相位请求占 running 名额,原生语义计数)
        has_prefill_work = any(
            self._lwd_the_phase_of_req(req) is LwdReqPhase.PREFILL
            for req in self.running
        ) or (len(self.running) < self.max_num_running_reqs
              and bool(self.waiting))
        if has_prefill_work:
            return LwdReqPhase.PREFILL
        if self.unembed_notify_queue:
            return LwdReqPhase.DECODE
        return None

    def _lwd_pick_prefill_req_id(self) -> str | None:
        """选 prefill 工作单元:running 续传优先,否则 waiting 队首。"""
        running_prefill = next(
            (r for r in self.running
             if self._lwd_the_phase_of_req(r) is LwdReqPhase.PREFILL),
            None,
        )
        if running_prefill is not None:
            return running_prefill.request_id
        return self.waiting.peek_request().request_id if self.waiting else None

    def schedule_prefill(self) -> SchedulerOutput:
        """单请求组批:picker 选一个,可见集调度;发布 RangeNotify 成功
        才挂 EMBED 批(seqno 与发布成功绑定)。发布失败返回空步弃批:
        进度已乐观推进,chunk 丢失由发布通道不丢消息保证不发生。"""
        req_id = self._lwd_pick_prefill_req_id()
        if req_id is not None:
            logger.info("[Lwd][edge-sched] pick req=%s", req_id)
        out = self._lwd_schedule_for_visible_reqs([req_id] if req_id else [])
        scheduled = out.num_scheduled_tokens
        if not scheduled:
            return out
        publisher = self.lwd_publisher
        for request_id, num_tokens in scheduled.items():
            if not self._lwd_publish_embed_chunk(
                publisher, out, request_id, num_tokens
            ):
                return SchedulerOutput.make_empty()
        return out

    def _lwd_publish_embed_chunk(
        self, publisher, out: SchedulerOutput, request_id: str, num_tokens: int
    ) -> bool:
        """发布 chunk 的 RangeNotify 并挂 EMBED 批;seqno 与发布成功绑定
        (peek-then-advance),失败返回 False(整步弃批)。"""
        request = self.requests.get(request_id)
        if request is None:
            return True
        # num_computed 已被乐观推进,起点回退本步量
        offset = request.num_computed_tokens - num_tokens
        seqno = self._lwd_seqno
        if publisher is None or not publisher.publish(
            LwdRangeNotify(
                request_id=request_id,
                offset=offset,
                num_tokens=num_tokens,
                seqno=seqno,
            )
        ):
            return False
        self._lwd_seqno = seqno + 1
        logger.info(
            "[Lwd][edge-notify] req=%s offset=%d num=%d seqno=%d",
            request_id, offset, num_tokens, seqno,
        )
        # seqno 是数据面发云张量的配对键,载荷为本 chunk 片段
        out.lwd_batch = LwdBatch(
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

    def schedule_decode(self) -> SchedulerOutput:
        """弹一条 unembed 通告派一步 decode(不经原生排程,账本自管):
        ① 无可调度的行 → 丢弃;② 纯终结通告(hidden=0,无 token 无
        数据)→ 本地终结,不派 worker;③ 带数据 → 排程量恒等于通告
        accept——本就是云侧决定的接收量,边侧无准入/预算/抢占决策
        自由度,欠条(收割冲销)与 computed 乐观推进在此手动登记,
        挂批走全量行集保 worker 切行对齐。"""
        notify = q.popleft() if (q := self.unembed_notify_queue) else None
        if notify is None:
            return SchedulerOutput.make_empty()
        # 通告里仍可调度的请求(存在、未终结、decode 相位);其余行
        # 由收割期原生跳过,批仍带全量通告行集保 worker 切行对齐
        targets = [
            rid for rid in notify.req_ids
            if (req := self.requests.get(rid)) is not None
            and self._lwd_the_phase_of_req(req) is LwdReqPhase.DECODE
        ]
        if not targets:
            logger.warning(
                "[Lwd][edge-sched] decode notify dropped (no schedulable "
                "targets): reqs=%s down_seqno=%s finish=%s",
                notify.req_ids, notify.down_seqno, notify.finish_reasons,
            )
            return SchedulerOutput.make_empty()
        # 纯终结通告:零 token 零数据(hidden=0 ⟹ accept 全零),无物
        # 可收、无批可派——通告挂空步交给 update_from_output 的兜底
        # 本地终结(空步过 super 无账可入,兜底照常消费完成码)
        if notify.hidden_num_elements == 0:
            logger.info(
                "[Lwd][edge-sched] pure-finish notify: local finish, no "
                "unembed: reqs=%s down_seqno=%s finish=%s",
                notify.req_ids, notify.down_seqno, notify.finish_reasons,
            )
            out = SchedulerOutput.make_empty()
            out.lwd_c2e_notify = [notify]
            return out
        # 手工构造 UNEMBED 步:不经原生排程(守卫/预算/准入均不涉及,
        # _update_after_schedule 亦不跑);收割期原生入账按欠条冲销归平
        accept = dict(zip(notify.req_ids, notify.num_accepted_tokens))
        for rid in targets:
            request = self.requests[rid]
            request.num_output_placeholders = (
                accept[rid] + request.num_computed_tokens
                - request.num_tokens_with_spec
            )
            request.num_computed_tokens += accept[rid]
        out = SchedulerOutput.make_empty()
        out.num_scheduled_tokens = {rid: accept[rid] for rid in targets}
        out.total_num_scheduled_tokens = sum(accept[rid] for rid in targets)
        self._lwd_attach_unembed_batch(out, notify)
        return out

    @staticmethod
    def _lwd_attach_unembed_batch(
        out: SchedulerOutput, notify: LwdC2eNotify
    ) -> None:
        """挂 UNEMBED 批:行集取全量通告,recv 尺寸以整条通告为准。"""
        out.lwd_batch = LwdBatch(
            batch_type=LwdBatchType.LWD_UNEMBED,
            seqno=notify.down_seqno,
            batch_meta=LwdUnembedBatch(
                req_ids=list(notify.req_ids),
                num_accept_tokens=list(notify.num_accepted_tokens),
                recv_num_elements=notify.hidden_num_elements,
                out_token_idxs=[],
                top_id_ths=list(notify.top_id_ths),
            ),
        )
        out.lwd_c2e_notify = [notify]

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """抑制 decode 行的自动 +1 占位:边侧不自产 token,行数由
        下一条通告决定,原生"账面领先 1"的预支不成立。"""
        super()._update_after_schedule(scheduler_output)
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            if request is not None and not request.is_prefill_chunk:
                request.num_output_placeholders -= 1

    def update_from_output(
        self, scheduler_output: SchedulerOutput, model_output
    ) -> dict[int, EngineCoreOutputs]:
        """原生入账后消费云侧完成码(兜底终结):云侧判停而原生
        check_stop 未触发的请求在此终结并补带完成码的空输出。终结必须
        在 super() 后(本批行含最后 token,批不能跳过执行);原生已停/
        迟到的请求已出 requests,自然跳过。

        EMBED 批无本地出账(采样在云侧,进度已于调度期乐观推进),整体
        跳过原生入账:原生循环以 num_scheduled_tokens 为驱动、要求输出
        含全部已调度请求,空输出必 KeyError。"""
        batch = scheduler_output.lwd_batch
        if batch is not None and batch.batch_type is LwdBatchType.LWD_EMBED:
            engine_core_outputs: dict[int, EngineCoreOutputs] = {}
        else:
            engine_core_outputs = super().update_from_output(
                scheduler_output, model_output
            )
        for notify in getattr(scheduler_output, "lwd_c2e_notify", None) or ():
            for rid, code in zip(notify.req_ids, notify.finish_reasons):
                if code == LWD_NOT_FINISHED:
                    continue
                request = self.requests.get(rid)
                if request is None or request.is_finished():
                    continue
                reason = FinishReason(code)
                self.finish_requests(
                    [rid], _LWD_FINISH_STATUS.get(
                        reason, RequestStatus.FINISHED_STOPPED
                    )
                )
                client_outputs = engine_core_outputs.setdefault(
                    request.client_index, EngineCoreOutputs()
                )
                client_outputs.outputs.append(
                    EngineCoreOutput(rid, [], finish_reason=reason)
                )
        return engine_core_outputs

