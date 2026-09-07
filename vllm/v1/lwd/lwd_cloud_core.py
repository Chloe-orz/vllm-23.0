"""云侧执行类:唯一接口 step_with_batch_queue,由 core.py 守卫委托(§9.8/§9.9)。

不再继承 EngineCore:真实 EngineCore 实例照常装配(worker/scheduler/batch_queue),
本类只替换其 step 语义;对 EngineCore 的触达一律经 LwdEnginePort(§7.3-C1)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.lwd.lwd_step_core import LwdStepCore

if TYPE_CHECKING:
    from vllm.v1.lwd.lwd_cloud_admission import LwdCloudAdmissionPolicy
    from vllm.v1.lwd.lwd_cloud_channel import LwdCloudControlSubscriber
    from vllm.v1.lwd.lwd_cloud_embeds import LwdCloudEmbedStore
    from vllm.v1.lwd.lwd_config import LwdConfig
    from vllm.v1.lwd.lwd_message import EngineCoreOutputs, LwdEmbedNotify
    from vllm.v1.lwd.lwd_ports import LwdEnginePort


class LwdCloudCore(LwdStepCore):
    """云侧 prefill_only core:控制面 drain -> 相位调度执行 -> 步后释放。

    调度器是装配期注入的 LwdCloudPhaseScheduler 实例(scheduler_cls 经
    lwd_cloud_scheduler_cls() 选取):分块决策走其原生 AsyncScheduler 语义
    (§9.9),Lwd 增量只有相位偏好;经 engine_port.lwd_scheduler() 触达。
    请求只经 PRE_OUT 到达(add/abort 全在 drain 内处理),公共接口仅
    step_with_batch_queue(§9.8)。
    """

    def __init__(
        self,
        subscriber: LwdCloudControlSubscriber,
        embed_store: LwdCloudEmbedStore,
        admission_policy: LwdCloudAdmissionPolicy,
        engine_port: LwdEnginePort,
        config: LwdConfig,
    ) -> None:
        ...

    def step_with_batch_queue(
        self,
    ) -> "tuple[dict[int, EngineCoreOutputs] | None, bool]":
        """编排:drain 控制面 -> 相位调度 + 执行(经 engine_port)-> 步后记账。"""
        ...

    # ---- 控制面处理(单向:仅订阅 drain,无快路径) ----

    def _lwd_drain_control_plane(self) -> None:
        ...

    def _lwd_handle_embed_notify(self, notify: LwdEmbedNotify) -> None:
        """登记 seqno->request 并转投 embeds 仓预登记。"""
        ...

    def _lwd_handle_add_request(self, request) -> None:
        ...

    def _lwd_handle_abort(self, request_id: str) -> None:
        """abort:embeds 仓丢弃 + 登记清理,在途 recv 的 tag 无人认领作废。"""
        ...

    # ---- 步进内部 ----

    def _lwd_apply_scheduling_policy(self) -> None:
        """准入策略(经 scheduler view 快照)决定本步可调度的 waiting 集合。"""
        ...

    def _lwd_update_retain_requests(self) -> None:
        """由调度器被抢占集合刷新 embeds 仓 retain(重计算候选,§9.2)。"""
        ...

    def _lwd_release_consumed(self) -> None:
        """释放已消费 embeds(retain 集合跳过);fill 重放依赖保留缓存。"""
        ...

    def _lwd_collect_finished(self) -> None:
        """finished 请求本地记账(结果不外发,§9.1)。"""
        ...

    def _lwd_release_request(self, request_id: str) -> None:
        """请求登记表清理唯一实现(W11):abort/finished/finish_reason 三路径共调。"""
        ...
