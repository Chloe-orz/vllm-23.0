"""边侧调度器:prompt 位置阶段发云 + 输出位置阶段收云(纯相位,与云侧镜像)。

边侧请求的完整生命周期留在原生记账内:prefill(embedding)完结后请求
保持 decode 相位留在 running,等待云侧 c2e 通告;decode 步弹一条通告
点名,arm 欠账后经可见集调度,worker 收 DOWN hidden 做 unembed,真实
token 经原生 update_from_output 入账销欠、判停、finish 全走原生路径。
迟到通告行由原生 update_from_output 跳过(requests.get 为 None 即
continue),无需手工幂等台账。

单请求组批约束:prefill 批最多含一个请求。目的:数据面 chunk 流按
请求连续(全局 seqno 链上单请求的 chunk 相邻),消除跨请求交错带来
的张量配对/重组复杂度。
"""

from __future__ import annotations

import time
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
    LwdAbortNotify,
    LwdC2eNotify,
    LwdRangeNotify,
    LwdRequestNotify,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
    LwdReqPhase,
)
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
        LwdControlPublisher,
    )
    from vllm.v1.request import Request

logger = init_logger(__name__)

# add 预告发布重试:次数 x 递增间隔(共约 3s),耗尽即请求级报错
_LWD_ADD_RETRY_STEPS = 5
_LWD_ADD_RETRY_INTERVAL_S = 0.2

# 云侧完成码 -> 边侧内部终态(前端拿到的 reason 走输出通道,不用此映射)
_LWD_FINISH_STATUS = {
    FinishReason.LENGTH: RequestStatus.FINISHED_LENGTH_CAPPED,
    FinishReason.ABORT: RequestStatus.FINISHED_ABORTED,
    FinishReason.ERROR: RequestStatus.FINISHED_ABORTED,
}


class LwdEdgeScheduler(LwdBaseScheduler):
    """prompt 发云 / 输出收云的纯相位调度 + 控制面出口(notify/abort/seqno)。"""

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
        # decode 通告队列:云侧步元数据(C2eNotify)逐条入队,decode 步
        # 弹队首点名其请求(跨线程单操作原子:接收线程 append/主步
        # popleft);一步一条,行集与通告严格对齐
        self.unembed_notify_queue: deque[LwdC2eNotify] = deque()

    def _lwd_select_phase(self) -> LwdReqPhase | None:
        """embed 优先直判(与旧引擎步序同语义):有 prefill 活走 prefill,
        否则看 unembed 通告,皆无返回 None(空排)。

        不做云侧的禁连续 prefill/强制翻转,原因见云侧对照:云侧 decode
        活本地自产、两相位随时有活,不变量防 prefill 独占;边侧 decode
        活要等本请求全部 chunk 发完云侧才产,不变量会把 chunk 发送本身
        拦死(单请求多 chunk 死锁)。"""
        has_prefill_work = any(
            self._lwd_the_phase_of_req(req) is LwdReqPhase.PREFILL
            for req in self.running
        ) or (self.lwd_edge_max_num_seqs_check() and bool(self.waiting))
        if has_prefill_work:
            return LwdReqPhase.PREFILL
        if self.unembed_notify_queue:
            return LwdReqPhase.DECODE
        return None

    def _lwd_pick_prefill_req_id(self) -> str | None:
        """选下一步 embed 工作单元:running 中第一个未发完的 prefill
        优先(断点续传,先收尾再开新),否则 waiting 队首(FCFS 到达序);
        两处皆无返回 None。"""
        running_prefill = next(
            (r for r in self.running
             if self._lwd_the_phase_of_req(r) is LwdReqPhase.PREFILL),
            None,
        )
        if running_prefill is not None:
            return running_prefill.request_id
        return self.waiting.peek_request().request_id if self.waiting else None

    def schedule_prefill(self) -> SchedulerOutput:
        """单请求组批:picker 选一个工作单元,经基类可见集机制单独调度,
        发布 RangeNotify 成功后挂 EMBED 批(与云侧 schedule_prefill 尾部
        挂批对称——发布点随调度,seqno 与发布成功绑定)。

        选择规则见 _lwd_pick_prefill_req_id;队列剔除/隔离/拼回复用基类
        _lwd_schedule_for_visible_reqs(waiting/skipped 来源走原生准入
        窗口,running 来源走续跑)。发布失败返回空步:进度已按排程乐观
        推进,该 chunk 随 SO 废弃即永久丢失,由发布通道不丢消息保证
        不发生(弃批语义与 dispatch 期失败一致)。"""
        req_id = self._lwd_pick_prefill_req_id()
        if req_id is not None:
            logger.info("[Lwd][edge-sched] pick req=%s", req_id)
        out = self._lwd_schedule_for_visible_reqs([req_id] if req_id else [])
        scheduled = out.num_scheduled_tokens
        if not scheduled:
            return out
        publisher = self.lwd_edge_publisher
        for request_id, num_tokens in scheduled.items():
            request = self.requests.get(request_id)
            if request is None:
                continue
            # _update_after_schedule 已乐观推进 num_computed,起点回退本步量
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
                return SchedulerOutput.make_empty()
            self._lwd_seqno = seqno + 1
            logger.info(
                "[Lwd][edge-notify] req=%s offset=%d num=%d seqno=%d",
                request_id, offset, num_tokens, seqno,
            )
            # 发布成功即组 EMBED 批挂 SO:seqno 是数据面发云张量的配对键
            # (与云侧 RangeNotify 登记同值),embed 载荷为本 chunk 片段
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
        return out

    def schedule_decode(self) -> SchedulerOutput:
        """纯 decode 步(输出位置阶段):弹一条 unembed 通告点名请求,
        校准欠账后经可见集调度,行数对齐挂 UNEMBED 批。

        欠账(占位数)是"这请求有 n 行活要算"的账面表达,由通告收据
        设定(边侧不自产 token);原生 RUNNING 循环按欠账排 n 行,worker
        收 DOWN hidden 做 unembed,真实 token 由原生 update_from_output
        入账销欠。全部迟到则丢弃通告;全部未准入则还原欠账、通告回塞
        重试;部分排程 = DOWN 行无法对齐,fail-loud。"""
        notify = self._lwd_pop_unembed_notify()
        if notify is None:
            return SchedulerOutput.make_empty()
        targets = self._lwd_alive_decode_reqs(notify)
        if not targets:
            return SchedulerOutput.make_empty()
        arm_deltas = self._lwd_arm_row_debt(notify, targets)
        out = self._lwd_schedule_for_visible_reqs(targets)
        if not out.num_scheduled_tokens:
            self._lwd_refund_row_debt(arm_deltas)
            self.unembed_notify_queue.appendleft(notify)
            return out
        self._lwd_assert_rows_match_notify(out, notify, targets)
        self._lwd_attach_unembed_batch(out, notify)
        return out

    def _lwd_pop_unembed_notify(self) -> LwdC2eNotify | None:
        """弹队首通告;空队返回 None。"""
        return q.popleft() if (q := self.unembed_notify_queue) else None

    def _lwd_alive_decode_reqs(self, notify: LwdC2eNotify) -> list[str]:
        """通告里仍可调度的请求:存在、未终结、处于 decode 相位。
        其余(已 abort/已原生终结的)行交给收割期原生跳过,不影响
        worker 的 DOWN 行切分(批仍带全量通告行集)。"""
        return [
            rid for rid in notify.req_ids
            if (req := self.requests.get(rid)) is not None
            and self._lwd_the_phase_of_req(req) is LwdReqPhase.DECODE
        ]

    def _lwd_arm_row_debt(
        self, notify: LwdC2eNotify, targets: list[str]
    ) -> dict[str, int]:
        """把每个目标请求的欠账校准到通告行数(占位欠条加减),返回
        {request_id: 占位增量} 供未准入时全额退还。"""
        rows_by_req = dict(zip(notify.req_ids, notify.num_accepted_tokens))
        arm_deltas: dict[str, int] = {}
        for rid in targets:
            request = self.requests[rid]
            owed = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            delta = rows_by_req[rid] - owed
            request.num_output_placeholders += delta
            arm_deltas[rid] = delta
        return arm_deltas

    def _lwd_refund_row_debt(self, arm_deltas: dict[str, int]) -> None:
        """退还欠账(未准入回退路径,防下次弹同一条通告双重欠账)。"""
        for rid, delta in arm_deltas.items():
            req = self.requests.get(rid)
            if req is not None:
                req.num_output_placeholders -= delta

    @staticmethod
    def _lwd_assert_rows_match_notify(
        out: SchedulerOutput, notify: LwdC2eNotify, targets: list[str]
    ) -> None:
        """排程行集/行数必须与通告的目标集逐请求相等(迟到行不参与
        排程):部分排程会让 worker 的 DOWN 行切分错位,当场报错。"""
        rows_by_req = dict(zip(notify.req_ids, notify.num_accepted_tokens))
        scheduled = out.num_scheduled_tokens
        if set(scheduled) != set(targets) or any(
            scheduled[rid] != rows_by_req[rid] for rid in targets
        ):
            raise RuntimeError(
                f"[LWD] edge decode batch mismatch vs c2e: "
                f"scheduled={dict(scheduled)} notify={rows_by_req}"
            )

    @staticmethod
    def _lwd_attach_unembed_batch(
        out: SchedulerOutput, notify: LwdC2eNotify
    ) -> None:
        """挂 UNEMBED 批:行集取全量通告(worker 按 DOWN 张量实际到达
        的行切分,recv 尺寸以整条通告为准),迟到行收割期原生跳过。"""
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
        """原生排程后抑制 decode 行的自动 +1 占位:边侧不自产 token,
        下一步行数由下一条 c2e 决定,原生"账面领先 1"的预支在此无法
        兑现(decode 行欠账由 schedule_decode 的 arm 精确设定)。"""
        super()._update_after_schedule(scheduler_output)
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            if request is not None and not request.is_prefill_chunk:
                request.num_output_placeholders -= 1

    def update_from_output(
        self, scheduler_output: SchedulerOutput, model_output
    ) -> dict[int, EngineCoreOutputs]:
        """原生入账后消费云侧完成码(云侧权威的兜底终结)。

        本批 token 已交付(含云侧判停请求的最后几行——所以终结必须在
        super() 之后、且批仍需执行,跳过执行 = 丢最后 token)。云侧判停
        而原生 check_stop 未触发的请求(如惩罚类停止)在此强制终结并
        补一条带完成码的空输出,防云侧停发后请求挂死占名额;原生已停/
        迟到的请求 requests 里已移除,自然跳过。"""
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
                outputs = engine_core_outputs.setdefault(
                    request.client_index, EngineCoreOutputs()
                )
                outputs.outputs.append(
                    EngineCoreOutput(rid, [], finish_reason=reason)
                )
        return engine_core_outputs

    def lwd_edge_max_num_seqs_check(self) -> bool:
        """max_num_seqs 水位:running 是否还有名额,True=可开新请求。

        decode 相位请求(等云结果)留在 running 占名额,计数即原生
        语义。续传豁免不在本判断(相位选择按 running 直判放行收尾);
        请求到达时的 announce 亦不受约束(云只登记不计算)。"""
        return len(self.running) < self.max_num_running_reqs

    def lwd_edge_add_request(self, request: Request) -> None:
        """请求入口:边界校验 -> 云预告 -> 本地入队。

        预告先于本地登记:云侧视图领先本地工作,云只能准备不能提前
        计算(没有 chunk 预告就不会开算)。abort_immediately 请求走
        finish + abort 出口,与原生语义一致。"""
        self._lwd_validate_request(request)
        self.lwd_edge_notify_request(
            request_id=request.request_id,
            num_prompt_tokens=len(request.prompt_token_ids),
            sampling_params=request.sampling_params,
            block_hashes=list(request.block_hashes),
        )
        super().add_request(request)
        if request.abort_immediately:
            self.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)
            self.lwd_edge_abort([request.request_id])

    def lwd_edge_notify_request(
        self,
        request_id: str,
        num_prompt_tokens: int,
        sampling_params: SamplingParams | None = None,
        block_hashes: list[bytes] | None = None,
    ) -> None:
        """发 LwdRequestNotify(请求元数据预告)。

        block_hashes = prompt 全量满块哈希链(自位置 0 起)。云侧
        prompt token 是占位零值,本地算不出真实内容哈希,前缀缓存
        命中只能靠这条链;缺省空链 = 不提供,云侧回退占位链。

        sampling_params 只透传影响云侧 token 选择的字段(采样核/惩罚/
        EOS 策略/min_tokens);stop 字符串等 detokenizer 层参数留在
        边侧前端原生处理,不上 wire。

        失败语义 fail-fast:发布队满时短退避重试(瞬态背压几乎必在
        秒级窗口内腾出),耗尽即抛 RuntimeError——异常沿 add_request
        调用链回前端 error 通道,用户立即得到失败;不做无限阻塞
        重试(发布点在引擎主线程,云宕机会把整个引擎卡死在 add)。
        此时请求未入队、云侧零残留,无需补发 abort。"""
        publisher = self.lwd_edge_publisher
        if publisher is None:
            return
        sp = sampling_params
        message = LwdRequestNotify(
            request_id=request_id,
            num_prompt_tokens=num_prompt_tokens,
            max_tokens=(
                sp.max_tokens if sp is not None and sp.max_tokens is not None else 16
            ),
            block_hashes=block_hashes if block_hashes is not None else [],
            temperature=sp.temperature if sp is not None else 1.0,
            top_p=sp.top_p if sp is not None else 1.0,
            top_k=sp.top_k if sp is not None else 0,
            min_p=sp.min_p if sp is not None else 0.0,
            seed=sp.seed if sp is not None else None,
            repetition_penalty=sp.repetition_penalty if sp is not None else 1.0,
            presence_penalty=sp.presence_penalty if sp is not None else 0.0,
            frequency_penalty=sp.frequency_penalty if sp is not None else 0.0,
            ignore_eos=sp.ignore_eos if sp is not None else False,
            stop_token_ids=(
                list(sp.stop_token_ids) if sp is not None and sp.stop_token_ids else []
            ),
            min_tokens=sp.min_tokens if sp is not None else 0,
            eos_token_id=sp.eos_token_id if sp is not None else None,
        )
        for attempt in range(_LWD_ADD_RETRY_STEPS):
            if publisher.publish(message):
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

    def lwd_edge_abort(self, request_ids: list[str]) -> None:
        """发 LwdAbortNotify;调度器内清理走原生路径(请求在三队列内,
        原生 abort/finish_requests 直接可见)。"""
        publisher = self.lwd_edge_publisher
        for request_id in request_ids:
            if publisher is None:
                continue
            if not publisher.publish(LwdAbortNotify(request_id=request_id)):
                logger.warning(
                    "[Lwd] drop abort signal for %s: publish queue full", request_id
                )
            else:
                logger.info(
                    "[Lwd][edge-notify] AbortNotify req=%s", request_id
                )

    @staticmethod
    def _lwd_validate_request(request: Request) -> None:
        """模式边界校验(入队前拒绝,错误回客户端):拒客户端自带
        prompt_embeds(边是 embedding 属主)、拒结构化输出、拒不上
        wire 的采样参数——后两者不拒会被云侧静默忽略,输出悄悄错。"""
        if request.prompt_embeds is not None:
            raise ValueError(
                f"[LWD] prefill-only mode does not accept client-provided "
                f"prompt_embeds (request {request.request_id}); the edge "
                "is the embedding owner"
            )
        if request.use_structured_output:
            raise ValueError(
                "[LWD] prefill-only mode does not support structured output "
                f"(request {request.request_id})"
            )
        sp = request.sampling_params
        unsupported = [
            name
            for name, value in (
                ("logit_bias", sp.logit_bias),
                ("allowed_token_ids", sp.allowed_token_ids),
                ("logprobs", sp.logprobs),
            )
            if value is not None
        ]
        if unsupported:
            raise ValueError(
                "[LWD] prefill-only mode does not support sampling "
                f"param(s) {', '.join(unsupported)} "
                f"(request {request.request_id}): not carried on the "
                "edge->cloud wire, cloud would silently sample without them"
            )
