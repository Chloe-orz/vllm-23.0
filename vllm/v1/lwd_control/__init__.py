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
    config = LwdConfig.from_env_and_config(vllm_config)
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
    """serve 入口守卫:云角色经 _lwd_cloud_deploy_guard 校验后注入相位调度器;
    云引擎类由子进程内 lwd_resolve_engine_cls 解析,边调度器由引擎自注入。"""
    from vllm.logger import init_logger
    from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_phase_scheduler import (
        LwdCloudPhaseScheduler,
    )
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
        LwdConfig,
        is_lwd_prefill_only,
    )

    if not is_lwd_prefill_only(vllm_config):
        return
    config = LwdConfig.from_env_and_config(vllm_config)
    if config.is_edge_node:
        return
    _lwd_cloud_deploy_guard(vllm_config, config)
    vllm_config.scheduler_config.scheduler_cls = LwdCloudPhaseScheduler
    init_logger(__name__).info(
        "[Lwd] prefill_only cloud: phase scheduler injected (construction-time)"
    )


def _lwd_cloud_deploy_guard(vllm_config, config) -> None:
    """云角色部署校验(fail-fast,serve 入口即拦):post_out_host 非空
    (POST_OUT 连边必需,master_addr 仅兼容回退);pre_out_host 非
    0.0.0.0(通告值须可路由)。"""
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    master_addr = vllm_config.parallel_config.master_addr
    if not config.post_out_host and not master_addr:
        raise ValueError(
            "[Lwd] prefill_only cloud requires lwd_config.post_out_host "
            "(POST_OUT connect target = edge IP; --master-addr is only a "
            "deprecated fallback)"
        )
    if not config.post_out_host and master_addr:
        logger.warning(
            "[Lwd] post_out_host unset, falling back to --master-addr=%s; "
            "prefer lwd_config.post_out_host",
            master_addr,
        )
    connect_host = config.post_out_host or master_addr
    if config.pre_out_host == "0.0.0.0":
        raise ValueError(
            "[Lwd] prefill_only cloud pre_out_host=0.0.0.0 is not announceable; "
            "set a routable IP (VLLM_ASCEND_LWD_PRE_OUT_HOST)"
        )
    if config.pre_out_host == "127.0.0.1" and connect_host not in (
        "127.0.0.1",
        "localhost",
    ):
        logger.warning(
            "[Lwd] cloud announces pre_out_host=127.0.0.1 but edge is remote "
            "(%s); edge will fail to reach PRE_OUT unless same host",
            connect_host,
        )
