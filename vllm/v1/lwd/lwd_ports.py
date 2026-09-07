"""端口协议(端口-适配器/六边形):内核只依赖协议,不触达 EngineCore/executor 内部(§7.3/§9.8)。

分块决策复用原生 schedule()(§9.9),端口只承载步进编排所需的最小触达面。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Protocol

if TYPE_CHECKING:
    from vllm.v1.lwd.lwd_message import LwdEdgeEmbedAck


class LwdFuture(Protocol):
    """嵌入提交结果的 future 抽象。"""

    def lwd_is_done(self) -> bool:
        ...

    def lwd_result(self):
        ...


class LwdEdgeExecutorPort(Protocol):
    """边侧执行器端口:原生 SchedulerOutput 原样过本地环 MQ 送 worker。

    不构造自定义 BatchType/SO 字段;worker_base 按角色守卫路由(§9.3/§9.9)。
    """

    def lwd_submit_embeds(self, scheduler_output) -> LwdFuture:
        ...

    def lwd_drain_acks(self) -> "list[LwdEdgeEmbedAck]":
        ...


class LwdEnginePort(Protocol):
    """边/云 step 编排对 EngineCore 的最小触达面(§7.3-C1/§9.8,两侧共用)。

    step_wrapper 模式下专用 core 不继承 EngineCore,scheduler/executor
    一律经本端口方法触达;方法集以 S2 step 实现实际所需为准收敛定稿。
    """

    def lwd_vllm_config(self):
        ...

    def lwd_scheduler(self):
        ...

    def lwd_execute_model(self, scheduler_output):
        ...


class LwdEnginePortAdapter:
    """LwdEnginePort 的通用适配器:包住真实 EngineCore(仅触达公共属性)。

    内核其余文件零 engine_core 属性触达;本类是台账登记的容忍点(§7.3-C5)。
    """

    def __init__(self, engine_core) -> None:
        ...

    def lwd_vllm_config(self):
        ...

    def lwd_scheduler(self):
        ...

    def lwd_execute_model(self, scheduler_output):
        ...


class LwdCloudSchedulerView(Protocol):
    """调度器只读快照(3 方法,C1,准入策略输入)。"""

    def lwd_unfinished_count(self) -> int:
        ...

    def lwd_waiting_count(self) -> int:
        ...

    def lwd_request_progress(self) -> "Iterable[tuple[str, int, int]]":
        ...
