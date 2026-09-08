"""边侧执行类:唯一接口 step_with_batch_queue,由 core.py 守卫委托(§9.8/§9.12)。

step 语义 = 原生步进的控制面替代:调度器出分块决策并发预告 ->
原生路径提交执行 -> 步末推进进度。数据面动作不在本层(§9.12)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.lwd_control.control_edge_scheduler.lwd_step_core import (
    LwdLog,
    LwdStepCore,
    LwdStepSettings,
)

if TYPE_CHECKING:
    from vllm.v1.lwd_control.control_communication.lwd_control_communicator import (
        EngineCoreOutputs,
    )
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_step_core import LwdEnginePort


class LwdEdgeCore(LwdStepCore):
    """边侧 prefill_only core:只发不收(§9.1),输出恒为 (None, 是否有工作)。"""

    def __init__(self, engine_port: LwdEnginePort, settings: LwdStepSettings) -> None:
        self._engine_port = engine_port
        self._log = LwdLog(settings.debug)

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """编排:调度器 schedule() -> 发预告 -> 提交执行 -> 步末推进进度。"""
        executed: dict[str, int] = {}
        if self._engine_port.lwd_scheduler().has_requests():
            scheduler_output = self._lwd_edge_step_schedule()
            executed = self._lwd_edge_step_dispatch(scheduler_output)
            self._log.phase("edge step: %d reqs executed", len(executed))
        self._lwd_edge_step_update_progress(executed)
        return None, bool(executed)

    def _lwd_edge_step_schedule(self):
        """取分块决策;调度器为装配期注入的 LwdEdgeScheduler(纯 prefill,§9.10)。"""
        return self._engine_port.lwd_scheduler().schedule()

    def _lwd_edge_step_dispatch(self, scheduler_output) -> dict[str, int]:
        """调度器 lwd_edge_notify 发预告 + 经 engine_port.lwd_execute_model 提交。

        预告失败(队满)本步不派发,返回空执行量 -> 进度回退、下一步重试(§2.4)。
        """
        scheduler = self._engine_port.lwd_scheduler()
        if not scheduler.lwd_edge_notify(scheduler_output):
            return {}
        self._engine_port.lwd_execute_model(scheduler_output)
        # 同步执行:调度量即执行量;数据面落位后由此接缝改报实际量(§9.12)
        return dict(scheduler_output.num_scheduled_tokens)

    def _lwd_edge_step_update_progress(self, executed: dict[str, int]) -> None:
        """步末推进调度器进度并终结已完请求(执行量来源由数据面对接,§9.12)。"""
        self._engine_port.lwd_scheduler().lwd_edge_update_progress(executed)
