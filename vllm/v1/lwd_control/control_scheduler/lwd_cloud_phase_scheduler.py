"""云侧相位调度器:纯 prefill/纯 decode 交替(内核唯一上游继承点之一,§7.5-1)。

相位实现 = 原生 schedule() 的队列手术复用(§9.9 不自造分割):
  - 纯 prefill 相位:decode-ready 请求自 running 暂存摘出,本步只调度
    prefill 续段与新准入;
  - 纯 decode 相位:waiting 队列整体暂存(无新准入),本步只调度 running
    (含未完 prefill 续段,避免已占块请求停滞)。
代价:纯 prefill 相位下 max_num_running_reqs 上限按摘出后的 running 计,
准入策略(SeparatePhases)是流量的实际闸门。
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue

logger = init_logger(__name__)


class LwdCloudPhaseScheduler(AsyncScheduler):
    """模板方法:schedule 编排固定,子类只翻转 _lwd_prefer_prefill 原语钩子。"""

    def schedule(self) -> SchedulerOutput:
        """上游签名(不改名);两相皆空时返回空输出保持步进节奏。"""
        if self._lwd_prefer_prefill() and self._lwd_has_prefill_work():
            return self._lwd_schedule_pure_prefill()
        return self._lwd_schedule_pure_decode()

    def _lwd_has_prefill_work(self) -> bool:
        """新准入或续段存在即有 prefill 工作。"""
        return bool(self.waiting or self.skipped_waiting) or any(
            request.is_prefill_chunk for request in self.running
        )

    def _lwd_has_decode_work(self) -> bool:
        return any(not request.is_prefill_chunk for request in self.running)

    def _lwd_schedule_pure_prefill(self) -> SchedulerOutput:
        """纯 prefill 相位:摘出 decode-ready 后全量复用原生调度。"""
        parked = [r for r in self.running if not r.is_prefill_chunk]
        if parked:
            self.running = [r for r in self.running if r.is_prefill_chunk]
        try:
            return super().schedule()
        finally:
            # 归还队首:decode-ready 到得早,恢复 FCFS 优先级与抢占序
            self.running[:0] = parked

    def _lwd_schedule_pure_decode(self) -> SchedulerOutput:
        """纯 decode 相位:暂存 waiting(无新准入)后复用原生调度。"""
        stashed_waiting = self.waiting
        stashed_skipped = self.skipped_waiting
        self.waiting = create_request_queue(self.policy)
        self.skipped_waiting = create_request_queue(self.policy)
        try:
            return super().schedule()
        finally:
            self._lwd_restore_waiting(stashed_waiting, stashed_skipped)

    def _lwd_restore_waiting(self, stashed_waiting, stashed_skipped) -> None:
        """归还暂存队列;本步被抢占回来的请求保持在最前(优先恢复)。"""
        for live, stashed in (
            (self.waiting, stashed_waiting),
            (self.skipped_waiting, stashed_skipped),
        ):
            for request in reversed(list(live)):
                stashed.prepend_request(request)
        self.waiting = stashed_waiting
        self.skipped_waiting = stashed_skipped

    def _lwd_prefer_prefill(self) -> bool:
        """相位偏好原语钩子(基类默认 prefill 优先)。"""
        return True


class LwdCloudPrefillFirstScheduler(LwdCloudPhaseScheduler):
    """prefill 相位优先(默认策略)。"""


class LwdCloudDecodeFirstScheduler(LwdCloudPhaseScheduler):
    """decode 相位优先(延迟敏感策略)。"""

    def _lwd_prefer_prefill(self) -> bool:
        return not self._lwd_has_decode_work()


_LWD_PHASE_SCHEDULERS = {
    "prefill_first": LwdCloudPrefillFirstScheduler,
    "decode_first": LwdCloudDecodeFirstScheduler,
}


def lwd_cloud_scheduler_cls(name: str | None = None) -> type[LwdCloudPhaseScheduler]:
    """工厂:按配置名取相位策略类;未知名回退默认并告警(不抛)。"""
    scheduler_cls = _LWD_PHASE_SCHEDULERS.get(name or "prefill_first")
    if scheduler_cls is None:
        logger.warning(
            "[Lwd] unknown phase scheduler %r, fallback to prefill_first", name
        )
        scheduler_cls = LwdCloudPrefillFirstScheduler
    return scheduler_cls
