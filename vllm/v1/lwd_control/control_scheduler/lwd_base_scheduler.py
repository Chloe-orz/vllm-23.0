"""Lwd 边/云调度器公共基类:收敛两侧共享的调度语义。

仍继承 AsyncScheduler,原生调度行为与 isinstance 判定不变;共享逻辑
随重构逐步上提至此类。
"""

from __future__ import annotations

import enum

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.request import Request

logger = init_logger(__name__)


class LwdReqPhase(enum.Enum):
    """请求相位:只由请求本体决定,与所在队列无关。"""

    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"


class LwdBaseScheduler(AsyncScheduler):
    """边/云调度器公共基类(AsyncScheduler 子类)。"""

    @staticmethod
    def _lwd_the_phase_of_req(request: Request) -> LwdReqPhase:
        """判定请求当前相位,边云共用:

        - FINISHED 优先判定(已终结的请求账面可能同时满足 DECODE 判据);
        - DECODE = prompt 已算完未终结;PREFILL = prompt 未算完
          (waiting 新请求 computed=0 天然落在 PREFILL,被抢占/续跑的
          decode 请求住回 waiting 也不影响判定——相位不读队列);
        - 勿与原生 Request.is_prefill_chunk 混淆:那是排程记账标志
          (公式含 spec/占位项、每步重算、新请求初值 False),非相位谓词。"""
        if request.is_finished():
            return LwdReqPhase.FINISHED
        if request.num_computed_tokens >= request.num_prompt_tokens:
            return LwdReqPhase.DECODE
        return LwdReqPhase.PREFILL

    def schedule(self) -> SchedulerOutput:
        """模板:选相位 → 纯相位批 → 步后簿记;None = 本步无活(空排)。

        步相位只取 LwdReqPhase 的 PREFILL/DECODE 两态(FINISHED 属请求
        相位,不会成为步相位);选择策略与步后簿记由边云子类各自实现,
        prefill/decode 是调度层的位置阶段词汇,worker 的执行风味
        (embed/unembed)由两侧实现各自挂批,不在本层出现。"""
        phase = self._lwd_select_phase()
        if phase is None:
            # [诊断] 空步不进原生 schedule,其尾部的 finished_req_ids
            # 交接在此不可达;打出 has_requests 的两个判定项以定位
            # 空转驱动源(unfinished= 谁还在调度器, finished_ids= 花名册)
            logger.info(
                "[Lwd][diag] idle step: unfinished=%s finished_ids=%s",
                self.get_num_unfinished_requests(), self.finished_req_ids,
            )
            return SchedulerOutput.make_empty()
        out = (
            self.schedule_prefill()
            if phase is LwdReqPhase.PREFILL
            else self.schedule_decode()
        )
        self._lwd_after_phase(phase, out)
        return out

    # ------------------------------------------------------------------ #
    # 边云钩子                                                            #
    # ------------------------------------------------------------------ #
    def _lwd_select_phase(self) -> LwdReqPhase | None:
        """选择策略:返回本步步相位;None = 本步无活。"""
        raise NotImplementedError

    def schedule_prefill(self) -> SchedulerOutput:
        """纯 prefill 步(prompt 位置阶段)。"""
        raise NotImplementedError

    def schedule_decode(self) -> SchedulerOutput:
        """纯 decode 步(输出位置阶段)。"""
        raise NotImplementedError

    def _lwd_after_phase(self, phase: LwdReqPhase, out: SchedulerOutput) -> None:
        """步后簿记钩子;缺省空实现。"""

    def _lwd_new_queue(self, reqs: list[Request]) -> RequestQueue:
        """新建调度策略队列并装入 reqs。"""
        queue = create_request_queue(self.policy)
        for req in reqs:
            queue.add_request(req)
        return queue

    def _lwd_schedule_for_visible_reqs(
        self, req_ids: list[str], token_budget_cap: int | None = None
    ) -> SchedulerOutput:
        """把 req_ids 指定的请求从三队列剔除、单独调度,步后按原队列拼回。

        waiting/skipped 来源的请求走原生准入窗口(allocate/状态迁移/
        记账一样不少),running 来源的走续跑。拼回:仍被调度的接在隐藏
        running 之后,被抢占的排 waiting 尾部,被跳过的排 skipped 队首。

        参数 ``token_budget_cap``:本步 prefill 预算上限,用于把"本步执行
        多少token"的决定权从云侧原生预算移交给边侧公告量。仅用于收紧,不放宽原生上限。"""
        picked = {self.requests[req_id] for req_id in req_ids}
        from_running = [req for req in self.running if req in picked]
        from_waiting = [req for req in self.waiting if req in picked]
        from_skipped = [req for req in self.skipped_waiting if req in picked]
        # 剔除后保存队列状态(隐藏集)
        self.running = [req for req in self.running if req not in picked]
        self.waiting.remove_requests(from_waiting)
        self.skipped_waiting.remove_requests(from_skipped)
        saved = (self.running, self.waiting, self.skipped_waiting)
        # 隐藏的 running 对原生并发准入不可见,名额按隐藏数扣减,防止
        # 隔离调用期间超发(超发崩溃记录 §6:prefill 步闸门失明准入第
        # N+1 个,decode 全员可见时撞原生断言)
        saved_cap = self.max_num_running_reqs
        self.max_num_running_reqs = max(0, saved_cap - len(saved[0]))
        # 被剔除的请求按原队列归位成可见集,单独调度
        self.running = from_running
        self.waiting = self._lwd_new_queue(from_waiting)
        self.skipped_waiting = self._lwd_new_queue(from_skipped)
        saved_token_budget = self.max_num_scheduled_tokens
        if token_budget_cap is not None:
            self.max_num_scheduled_tokens = min(
                saved_token_budget, max(0, token_budget_cap)
            )
        try:
            out = super().schedule()
        finally:
            # 先恢复名额与预算再拼回:running 存活者接尾,waiting 被抢占者排队尾,
            # skipped 被跳过者排队首
            self.max_num_running_reqs = saved_cap
            self.max_num_scheduled_tokens = saved_token_budget
            post = (self.running, self.waiting, self.skipped_waiting)
            self.running, self.waiting, self.skipped_waiting = saved
            self.running += post[0]
            self.waiting.extend(post[1])
            self.skipped_waiting.prepend_requests(post[2])
        return out
