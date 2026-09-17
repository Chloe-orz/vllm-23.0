"""Lwd 边/云调度器公共基类:收敛两侧共享的调度语义。

仍继承 AsyncScheduler,原生调度行为与 isinstance 判定不变;共享逻辑
随重构逐步上提至此类。
"""

from __future__ import annotations

import enum

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.request import Request


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

    def _lwd_new_queue(self, reqs: list[Request]) -> RequestQueue:
        """新建调度策略队列并装入 reqs。"""
        queue = create_request_queue(self.policy)
        for req in reqs:
            queue.add_request(req)
        return queue

    def _lwd_schedule_for_visible_reqs(self, req_ids: list[str]) -> SchedulerOutput:
        """把 req_ids 指定的请求从三队列剔除、单独调度,步后按原队列拼回。

        waiting/skipped 来源的请求走原生准入窗口(allocate/状态迁移/
        记账一样不少),running 来源的走续跑。拼回:仍被调度的接在隐藏
        running 之后,被抢占的排 waiting 尾部,被跳过的排 skipped 队首。"""
        picked = {self.requests[req_id] for req_id in req_ids}
        from_running = [req for req in self.running if req in picked]
        from_waiting = [req for req in self.waiting if req in picked]
        from_skipped = [req for req in self.skipped_waiting if req in picked]
        # 剔除后保存队列状态(隐藏集)
        self.running = [req for req in self.running if req not in picked]
        self.waiting.remove_requests(from_waiting)
        self.skipped_waiting.remove_requests(from_skipped)
        saved = (self.running, self.waiting, self.skipped_waiting)
        # 被剔除的请求按原队列归位成可见集,单独调度
        self.running = from_running
        self.waiting = self._lwd_new_queue(from_waiting)
        self.skipped_waiting = self._lwd_new_queue(from_skipped)
        try:
            out = super().schedule()
        finally:
            # 按原队列拼回:running 存活者接尾,waiting 被抢占者排队尾,
            # skipped 被跳过者排队首
            post = (self.running, self.waiting, self.skipped_waiting)
            self.running, self.waiting, self.skipped_waiting = saved
            self.running += post[0]
            self.waiting.extend(post[1])
            self.skipped_waiting.prepend_requests(post[2])
        return out
