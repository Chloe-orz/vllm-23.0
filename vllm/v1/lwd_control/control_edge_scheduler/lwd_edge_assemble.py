"""边侧 L3 装配:EngineCore.__init__ 守卫调用的单点开关(§8.2/§9.8)。

单向语义:无云结果/水位 drain,步进逻辑全部收编进
LwdEdgeCore.step_with_batch_queue;本文件只负责装配与生命周期,
原 6 个 monkey-patch 由 core.py in-tree 守卫分支替代。

本文件同时承载两侧共享的配置支撑(§10.3 折入):LwdConfig
(env/additional_config 唯一解析入口)与 is_lwd_prefill_only
(模式判定唯一实现);云侧装配经 import 复用,内核模块收 plain 值。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
    LWD_PUBLISH_QUEUE_MAX,
    LwdControlPublisher,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_core import LwdEdgeCore
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_step_core import LwdStepSettings

logger = init_logger(__name__)

LWD_PRE_OUT_PORT_DEFAULT = 5558

_LWD_CONFIG_SECTION = "edge_cloud_config"
_LWD_ENV_PREFIX = "VLLM_ASCEND_LWD_"


@dataclass(frozen=True)
class LwdConfig:
    """plain 值配置对象;装配期一次成型,内核模块只收值不再解析。"""

    is_edge_node: bool = True
    pre_out_host: str = "127.0.0.1"
    pre_out_port: int = LWD_PRE_OUT_PORT_DEFAULT
    scheduler_name: str = "prefill_first"
    publish_queue_max: int = LWD_PUBLISH_QUEUE_MAX
    zombie_log_interval_s: float = 30.0
    debug: bool = False

    def lwd_pre_out_endpoint(self) -> str:
        """PRE_OUT 通道端点(边 connect / 云 bind 同一地址)。"""
        return f"tcp://{self.pre_out_host}:{self.pre_out_port}"

    def lwd_step_settings(self) -> LwdStepSettings:
        """派生内核 plain 值(§10.3:内核不解析配置)。"""
        return LwdStepSettings(
            debug=self.debug, zombie_log_interval_s=self.zombie_log_interval_s
        )

    @classmethod
    def from_env_and_config(cls, vllm_config) -> LwdConfig:
        """解析 edge_cloud_config 段;env(VLLM_ASCEND_LWD_*)只覆盖地址与开关。"""
        section = _lwd_read_section(vllm_config)
        config = cls(
            is_edge_node=str(section.get("role", "edge")) == "edge",
            pre_out_host=str(section.get("pre_out_host", "127.0.0.1")),
            pre_out_port=int(section.get("pre_out_port", LWD_PRE_OUT_PORT_DEFAULT)),
            scheduler_name=str(section.get("scheduler", "prefill_first")),
            publish_queue_max=int(
                section.get("publish_queue_max", LWD_PUBLISH_QUEUE_MAX)
            ),
            zombie_log_interval_s=float(section.get("zombie_log_interval_s", 30.0)),
            debug=bool(section.get("debug", False)),
        )
        return _lwd_apply_env_overrides(config)


def is_lwd_prefill_only(vllm_config) -> bool:
    """模式判定唯一实现(全仓 1 处,§2.3/W9);edge/cloud 角色由 LwdConfig 区分。

    主仓既有文件不 import 本函数:上游守卫经装配函数内部分流。
    """
    section = _lwd_read_section(vllm_config)
    return section.get("mode") == "prefill_only"


def _lwd_read_section(vllm_config) -> dict:
    """取 additional_config 下的 edge_cloud_config 段;缺省/非 dict 均按空段处理。"""
    additional = vllm_config.additional_config or {}
    section = additional.get(_LWD_CONFIG_SECTION)
    return section if isinstance(section, dict) else {}


def _lwd_apply_env_overrides(config: LwdConfig) -> LwdConfig:
    """env 只覆盖部署相关项(地址/端口/调试开关);坏值告警并保留默认,不抛。"""
    host = os.getenv(_LWD_ENV_PREFIX + "PRE_OUT_HOST")
    port = _lwd_read_env_int("PRE_OUT_PORT")
    debug = os.getenv(_LWD_ENV_PREFIX + "DEBUG")
    return LwdConfig(
        is_edge_node=config.is_edge_node,
        pre_out_host=host if host else config.pre_out_host,
        pre_out_port=port if port is not None else config.pre_out_port,
        scheduler_name=config.scheduler_name,
        publish_queue_max=config.publish_queue_max,
        zombie_log_interval_s=config.zombie_log_interval_s,
        debug=config.debug or (debug is not None and debug.lower() == "1"),
    )


def _lwd_read_env_int(name: str) -> int | None:
    """读整型 env;缺省返回 None,坏值告警返回 None(调用方保留默认)。"""
    raw = os.getenv(_LWD_ENV_PREFIX + name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("[Lwd] ignore invalid env %s%s=%r", _LWD_ENV_PREFIX, name, raw)
        return None


class LwdEdgeEnginePortAdapter:
    """边侧端口适配器(§10.3 落位装配文件):仅承载边侧编排触达的两个方法。

    台账登记的 engine_core 属性容忍点(§7.3-C5);内核零属性触达。
    """

    def __init__(self, engine_core) -> None:
        self._engine_core = engine_core

    def lwd_scheduler(self):
        return self._engine_core.scheduler

    def lwd_execute_model(self, scheduler_output):
        executor = self._engine_core.model_executor
        return executor.execute_model(scheduler_output).result()


def lwd_edge_try_assemble(engine_core) -> bool:
    """装配点(core.py __init__ 尾守卫调用):非 PO 立即返回 False,零副作用。

    PO 时:建 PRE_OUT 通道,scheduler_cls 以 partial(LwdEdgeScheduler,
    publisher=...)注入(调度器即控制面出口,§9.12),建 LwdEdgeCore 并赋给
    engine_core.step_wrapper(core.py step 守卫的唯一委托对象,§9.8)。
    """
    vllm_config = engine_core.vllm_config
    if not is_lwd_prefill_only(vllm_config):
        return False
    config = LwdConfig.from_env_and_config(vllm_config)
    if not config.is_edge_node:
        return False
    if engine_core.scheduler.get_kv_connector() is not None:
        # 调度器重建会丢失 connector 握手态:PO 不支持 kv_connector,降级原生
        logger.warning(
            "[Lwd] kv_connector enabled on edge: skip Lwd assembly, degrade to native"
        )
        return False
    publisher = _lwd_edge_build_planes(config)
    _lwd_edge_install_scheduler(engine_core, publisher)
    engine_core.step_wrapper = LwdEdgeCore(
        LwdEdgeEnginePortAdapter(engine_core), config.lwd_step_settings()
    )
    return True


def _lwd_edge_build_planes(config: LwdConfig) -> LwdControlPublisher:
    """建控制面发布通道(单向,仅 PRE_OUT;无结果面,§9.1)。

    bind/connect 是装配期 wiring(传输层 side-agnostic):边侧 connect,
    云侧 bind 于同一端点。
    """
    return LwdControlPublisher(
        config.lwd_pre_out_endpoint(), bind=False, queue_max=config.publish_queue_max
    )


def _lwd_edge_install_scheduler(engine_core, publisher) -> None:
    """以 LwdEdgeScheduler 重建调度器(publisher 经构造注入,§9.10)。

    __init__ 尾装配晚于原生调度器构建,只能整实例替换:装配点无在途
    请求,重建仅多一次前缀缓存管理器构建;入口期注入 scheduler_cls
    可免此重建(上游接线/数据面落位时一并处理,记入设计文档 backlog)。
    """
    vllm_config = engine_core.vllm_config
    native_scheduler = engine_core.scheduler
    block_size, hash_block_size = resolve_kv_cache_block_sizes(
        native_scheduler.kv_cache_config, vllm_config
    )
    engine_core.scheduler = LwdEdgeScheduler(
        vllm_config=vllm_config,
        kv_cache_config=native_scheduler.kv_cache_config,
        structured_output_manager=engine_core.structured_output_manager,
        log_stats=engine_core.log_stats,
        block_size=block_size,
        hash_block_size=hash_block_size,
        publisher=publisher,
    )


def lwd_edge_shutdown(engine_core) -> None:
    """通道关停(core.py shutdown 守卫分支调用;装配层掌生命周期)。"""
    scheduler = engine_core.scheduler
    if (
        isinstance(scheduler, LwdEdgeScheduler)
        and scheduler.lwd_edge_publisher is not None
    ):
        scheduler.lwd_edge_publisher.shutdown()
