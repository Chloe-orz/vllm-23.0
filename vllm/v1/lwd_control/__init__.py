"""Lwd prefill-only 控制面内核(仅 vllm 仓):控制面调度与控制面通信,数据面
经既有接缝对接。设计文档与 import 白名单见 docs/refactor/prefill_only_migration.md。"""

from __future__ import annotations


def lwd_resolve_engine_cls(vllm_config):
    """core.py 类选择点的分流点:云角色返回 LwdCloudEngineCore,边角色返回
    LwdEdgeEngineCore,其余返回 None(调用方用原生 EngineCoreProc)。"""
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
        LwdConfig,
        is_lwd_prefill_only,
    )

    if not is_lwd_prefill_only(vllm_config):
        return None
    config = LwdConfig.from_vllm_config(vllm_config)
    from vllm.logger import init_logger

    if config.is_edge_node:
        from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
            LwdEdgeEngineCore,
        )

        init_logger(__name__).info(
            "[Lwd] prefill_only edge: engine class selected (LwdEdgeEngineCore)"
        )
        return LwdEdgeEngineCore
    from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
        LwdCloudEngineCore,
    )

    init_logger(__name__).info(
        "[Lwd] prefill_only cloud: engine class selected (LwdCloudEngineCore)"
    )
    return LwdCloudEngineCore


def lwd_serve_guard(vllm_config) -> None:
    """serve 入口守卫:注入云相位调度器;云引擎类由子进程内
    lwd_resolve_engine_cls 解析,边调度器由引擎自注入。

    部署校验已收敛到配置层(lwd_topology 加载期校验含 addr 合法性/
    ctrl_port 错开/连接关系),云侧 ROUTER 端点由拓扑唯一决定,本守卫
    不再重复校验端点字段。"""
    from vllm.logger import init_logger
    from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_phase_scheduler import (
        LwdCloudPhaseScheduler,
    )
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
        is_lwd_prefill_only,
    )

    if not is_lwd_prefill_only(vllm_config):
        return
    if vllm_config.lwd_config.is_edge:
        return
    vllm_config.scheduler_config.scheduler_cls = LwdCloudPhaseScheduler
    init_logger(__name__).info(
        "[Lwd] prefill_only cloud: phase scheduler injected (construction-time)"
    )
