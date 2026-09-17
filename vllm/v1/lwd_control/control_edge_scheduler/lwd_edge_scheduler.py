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
from vllm.v1.lwd_control.control_communication.lwd_notify import (
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
        self.decode_notify_queue: deque[LwdC2eNotify] = deque()
        # One-shot 翻转与禁连续 prefill 不变量(与云侧同款镜像簿记)
        self._force_prefill_once: bool = False
        self._force_decode_once: bool = False
        self._last_step_was_prefill: bool = False

    def _lwd_select_phase(self) -> LwdReqPhase | None:
        """embed 优先的相位选择(镜像云侧簿记)。

        prefill 活 = running 有未发完的续传(不受闸门约束,先收尾再开新,
        亦防上限=1 时自锁)或水位有余且 waiting 非空;decode 活 = 云侧
        c2e 通告在队。无活返回 None(空排)。"""
        has_prefill_work = any(
            self._lwd_the_phase_of_req(req) is LwdReqPhase.PREFILL
            for req in self.running
        ) or (self.lwd_edge_max_num_seqs_check() and bool(self.waiting))
        prefer_prefill = has_prefill_work
        if self._force_prefill_once:
            self._force_prefill_once = False
            prefer_prefill = True
        elif self._force_decode_once:
            self._force_decode_once = False
            prefer_prefill = False
        # 不变量(与云侧同款):prefill 不连续执行两步,防连续 chunk
        # 挤占 decode 交付;空 decode 步经 _force_prefill_once 翻回。
        if prefer_prefill and self._last_step_was_prefill and self.running:
            prefer_prefill = False
        if prefer_prefill:
            return LwdReqPhase.PREFILL
        if self.decode_notify_queue:
            return LwdReqPhase.DECODE
        return None

    def _lwd_after_phase(
        self, phase: LwdReqPhase, out: SchedulerOutput
    ) -> None:
        """空步翻转与禁连续 prefill 簿记(镜像云侧)。"""
        if phase is LwdReqPhase.PREFILL:
            if not out.total_num_scheduled_tokens and self.running:
                self._force_decode_once = True
            else:
                self._last_step_was_prefill = True
            return
        self._last_step_was_prefill = False
        if not out.total_num_scheduled_tokens and (
            bool(self.waiting)
            or any(
                self._lwd_the_phase_of_req(req) is LwdReqPhase.PREFILL
                for req in self.running
            )
        ):
            self._force_prefill_once = True

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
        """纯 decode 步(输出位置阶段):弹一条 c2e 点名,arm 欠账后经
        可见集调度,行数与通告逐请求对齐后挂 UNEMBED 批。

        arm = 把每请求的待算行数(占位欠账)校准到 c2e 的行数:原生
        RUNNING 循环按欠账排 n 行,worker 收 DOWN hidden 做 unembed,
        真实 token 由原生 update_from_output 入账销欠。行数欠账由
        c2e 收据决定(边侧不自产 token、不预支未来);全部未准入时
        还原欠账、通告回塞队首重试;部分排程 = DOWN 行无法对齐,
        fail-loud。迟到行交给原生跳过,通告批保留全量行集(worker
        按 DOWN 张量实际到达的行切分,recv 尺寸以整条通告为准)。"""
        notify = q.popleft() if (q := self.decode_notify_queue) else None
        if notify is None:
            return SchedulerOutput.make_empty()
        rows = dict(zip(notify.req_ids, notify.num_accepted_tokens))
        live = [
            rid for rid in notify.req_ids
            if (req := self.requests.get(rid)) is not None
            and self._lwd_the_phase_of_req(req) is LwdReqPhase.DECODE
        ]
        if not live:
            return SchedulerOutput.make_empty()
        armed: dict[str, int] = {}
        for rid in live:
            request = self.requests[rid]
            cur = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            delta = rows[rid] - cur
            request.num_output_placeholders += delta
            armed[rid] = delta
        out = self._lwd_schedule_for_visible_reqs(live)
        scheduled = out.num_scheduled_tokens
        if not scheduled:
            for rid, delta in armed.items():
                req = self.requests.get(rid)
                if req is not None:
                    req.num_output_placeholders -= delta
            self.decode_notify_queue.appendleft(notify)
            return out
        if set(scheduled) != set(live) or any(
            scheduled[rid] != rows[rid] for rid in scheduled
        ):
            raise RuntimeError(
                f"[LWD] edge decode batch mismatch vs c2e: "
                f"scheduled={dict(scheduled)} notify={rows}"
            )
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
        return out

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """原生排程后抑制 decode 行的自动 +1 占位:边侧不自产 token,
        下一步行数由下一条 c2e 决定,原生"账面领先 1"的预支在此无法
        兑现(decode 行欠账由 schedule_decode 的 arm 精确设定)。"""
        super()._update_after_schedule(scheduler_output)
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            if request is not None and not request.is_prefill_chunk:
                request.num_output_placeholders -= 1

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
        """模式边界校验;违规抛 ValueError,在入队之前拒绝,错误经
        add_request 调用链回到客户端 error 路径。

        边界 = 边侧能力面:边是 embedding 属主(拒绝客户端自带
        prompt_embeds),只处理纯文本补全(拒 pooling/结构化
        输出),prompt 非空;不上 wire 的采样参数(logit_bias/
        allowed_token_ids/logprobs)缺省即拒,不静默丢约束。"""
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
