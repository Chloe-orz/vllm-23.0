"""云侧相位调度器:工作纯相位批次(源 pure_phase_scheduler.py 基础上的裁定修订)。

相对源实现的差异(台账 §10.8/§10.15):
  1. 工作纯相位取代"按人口分伙":prefill 步算一切 prompt 工作(WAITING 首块
     + RUNNING 中的 prefill 尾巴),decode 步只算 prompt 已完结请求的 1-token
     采样 —— 长序列被 chunked prefill 截断的尾巴不再混进 decode 批;
     单请求组批约束(§9.9 修订):prefill 批最多含一个请求(容器交换,
     尾巴优先/队首放行),数据面 chunk 流按请求连续;
  2. 子类合并:LwdCloudPrefillFirst/DecodeFirst 收敛为本类,相位在构造期经
     LwdConfig 自解析(scheduler_cls 注入只带类身份,相位改走 vllm_config,
     §10.15);注册表/工厂删除,未知相位名告警回退 prefill_first;
  3. 准入固定 immediate 直进,separate_phases 准入族已按裁定删除(§10.14);
  4. 类名映射:PurePhaseSchedulerBase -> LwdCloudPhaseScheduler。

容器交换手法与源同款:改写 self.running/self.waiting 的可见集跑一次
super().schedule(),finally 复原。前置约束:云侧不启用 spec decode
(eagle 会 shift num_computed,工作纯度判据失真)。
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue

logger = init_logger(__name__)

_LWD_PHASE_PREFILL_FIRST = "prefill_first"
_LWD_PHASE_DECODE_FIRST = "decode_first"


class LwdCloudPhaseScheduler(AsyncScheduler):
    """工作纯相位批次策略;相位(prefill_first/decode_first)构造期自解析。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lwd_prefill_first = self._lwd_resolve_phase()
        # One-shot flag: the last chosen phase produced an empty step and
        # the other population has work — try that phase next step.
        self._force_other_phase: bool = False
        logger.info(
            "[Lwd] cloud phase scheduler: single-request prefill batches "
            "enforced (edge/cloud chunk stream stays per-request contiguous)"
        )

    # ------------------------------------------------------------------ #
    # Phase resolution(§10.15:scheduler_cls 单类,相位经 LwdConfig)      #
    # ------------------------------------------------------------------ #
    def _lwd_resolve_phase(self) -> bool:
        """返回 True=prefill_first;缺省/未知相位告警回退 prefill_first。"""
        from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
            LwdConfig,
        )

        phase = LwdConfig.from_env_and_config(self.vllm_config).scheduler_name
        if phase == _LWD_PHASE_DECODE_FIRST:
            return False
        if phase != _LWD_PHASE_PREFILL_FIRST:
            logger.warning(
                "[Lwd] unknown cloud scheduler phase %r "
                "(known: %s); falling back to %s",
                phase,
                [_LWD_PHASE_DECODE_FIRST, _LWD_PHASE_PREFILL_FIRST],
                _LWD_PHASE_PREFILL_FIRST,
            )
        return True

    # ------------------------------------------------------------------ #
    # Work-purity predicates(工作纯度唯一判据:prompt 是否算完)           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _lwd_is_prefill_tail(request) -> bool:
        """RUNNING 且 prompt 未算完 = prefill 尾巴(尾巴只允许进 prefill 步)。"""
        return request.num_computed_tokens < request.num_prompt_tokens

    def _lwd_split_running(self) -> tuple[list, list]:
        """running 二分为 (decode_ready, tails),各自保持 FIFO 序。"""
        decode_ready: list = []
        tails: list = []
        for request in self.running:
            (tails if self._lwd_is_prefill_tail(request) else decode_ready).append(
                request
            )
        return decode_ready, tails

    def _lwd_has_prefill_tails(self) -> bool:
        return any(self._lwd_is_prefill_tail(r) for r in self.running)

    def _lwd_has_decode_ready(self) -> bool:
        return any(not self._lwd_is_prefill_tail(r) for r in self.running)

    # ------------------------------------------------------------------ #
    # Phase primitives(容器交换;原生 schedule() 零改动)                  #
    # ------------------------------------------------------------------ #
    def _schedule_pure_prefill(self) -> SchedulerOutput:
        """一个纯 prefill 步,批内最多一个请求(§9.9 单请求组批约束)。

        visible 集只放一个 prefill 工作单元:running 尾巴优先(藏其余
        尾巴与全部 waiting),否则只放行 waiting 队首;decode-ready 照旧
        藏起(其 1-token 采样不得混入 prefill 批)。单请求内 chunked
        决策(预算截断/KV 抢占)照旧;藏起的 waiting 走队首回插,藏起
        的尾巴接回 running 尾部,均保 FIFO。"""
        decode_ready, tails = self._lwd_split_running()
        hidden_waiting = self.waiting
        self.waiting = create_request_queue(self.policy)
        hidden_tails: list = []
        if tails:
            hidden_tails = tails[1:]
            self.running = tails[:1]
        else:
            # decode-ready 藏起(与原版同语义);无尾巴时 running 清空
            self.running = []
            if hidden_waiting:
                self.waiting.add_request(hidden_waiting.peek_request())
        try:
            out = super().schedule()
        finally:
            # schedule 期间该列表 = 幸存尾巴(被抢占的已弹出)
            # + 本步 waiting->running 的新请求,接在 decode-ready 之后。
            leftover = self.waiting
            self.waiting = hidden_waiting
            while leftover:
                self.waiting.prepend_request(leftover.pop_request())
            self.running = decode_ready + self.running + hidden_tails
        return out

    def _schedule_pure_decode(self) -> SchedulerOutput:
        """一个纯 decode 步:WAITING 与 prefill 尾巴都藏起,
        批内只剩 prompt 已完结请求的采样。"""
        hidden_waiting = self.waiting
        self.waiting = create_request_queue(self.policy)
        decode_ready, tails = self._lwd_split_running()
        self.running = decode_ready
        try:
            out = super().schedule()
        finally:
            newly_waiting = self.waiting
            self.waiting = hidden_waiting
            # The temp queue should be empty or hold nothing scheduled;
            # drain defensively back in front (FIFO order preserved).
            while newly_waiting:
                self.waiting.append(newly_waiting.popleft())
            # 解抢占的请求经 _preempt_request 进了临时 waiting,已随上两行
            # 回插队首;这里复原被藏的尾巴(保持其 FIFO 序)。
            self.running = tails + self.running
        return out

    @staticmethod
    def _is_empty(out: SchedulerOutput) -> bool:
        return out.total_num_scheduled_tokens == 0

    # ------------------------------------------------------------------ #
    # Phase choice                                                        #
    # ------------------------------------------------------------------ #
    def _prefer_prefill(self) -> bool:
        """相位选择:prefill_first 有等待/尾巴即 prefill;
        decode_first 只要存在纯 decode 活就优先 decode。"""
        if self._lwd_prefill_first:
            return bool(self.waiting) or self._lwd_has_prefill_tails()
        return not self._lwd_has_decode_ready()

    # ------------------------------------------------------------------ #
    # Entry point                                                         #
    # ------------------------------------------------------------------ #
    def schedule(self) -> SchedulerOutput:
        prefer_prefill = self._prefer_prefill()
        if self._force_other_phase:
            prefer_prefill = not prefer_prefill
            self._force_other_phase = False

        if prefer_prefill:
            out = self._schedule_pure_prefill()
            if self._is_empty(out) and self.running:
                # Prefill blocked (KV pressure). Yield the empty cleanup step
                # and let the next step run decode so KV pressure can drain.
                self._force_other_phase = True
            return out
        out = self._schedule_pure_decode()
        if self._is_empty(out) and (self.waiting or self._lwd_has_prefill_tails()):
            # Decode 无活但有 waiting/尾巴:翻回 prefill 让尾巴推进。
            self._force_other_phase = True
        return out
