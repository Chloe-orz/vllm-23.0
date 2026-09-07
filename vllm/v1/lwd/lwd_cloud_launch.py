"""云侧 L3 装配:进程入口 + 调度器视图适配器(违禁 import 只允许本文件,§7.2/§9.8)。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from vllm.v1.lwd.lwd_cloud_core import LwdCloudCore
    from vllm.v1.lwd.lwd_config import LwdConfig
    from vllm.v1.lwd.lwd_ports import LwdEnginePort


class LwdCloudSchedulerViewAdapter:
    """包住 engine_core.scheduler 的只读快照(§7.3-C1,准入策略输入)。"""

    def __init__(self, scheduler) -> None:
        ...

    def lwd_unfinished_count(self) -> int:
        ...

    def lwd_waiting_count(self) -> int:
        ...

    def lwd_request_progress(self) -> "Iterable[tuple[str, int, int]]":
        ...


def _lwd_cloud_admit_request(
    port: LwdEnginePort, request, config: LwdConfig
) -> None:
    """唯一准入交互点:Request 构建/block_hasher/远程 embed 视图挂载全收于此(§7.3-C1)。

    挂载的 prompt_embeds 即 LwdCloudRemoteEmbeds 惰性视图(对齐上游
    fill 访问面),上游填充循环零改动(§9.5)。
    """
    ...


def lwd_cloud_main(args) -> None:
    """云进程入口(serve.py run_headless 守卫分支调用);装配三段见下(§2.6)。

    装配完成前最后一件事:engine_core.step_wrapper = cloud_core(§9.8 委托点)。
    """
    ...


def _lwd_cloud_init_process(args) -> None:
    """进程级初始化(角色标记/信号/日志)。"""
    ...


def _lwd_cloud_connect_planes(config: LwdConfig):
    """建控制面订阅通道(单向,仅 PRE_OUT;无结果面,§9.1)。"""
    ...


def _lwd_cloud_build_core(*plane_deps) -> LwdCloudCore:
    """装配云侧执行类。

    调度接线:scheduler_cls 经 lwd_cloud_scheduler_cls() 按配置选取
    (PrefillFirst/DecodeFirst),注入真实 EngineCore 的构造——此后
    engine_port.lwd_scheduler() 拿到的即相位调度器实例;EngineCore 侧经
    LwdEnginePortAdapter 包装,准入策略经 lwd_cloud_admission_policy() 注入。
    """
    ...
