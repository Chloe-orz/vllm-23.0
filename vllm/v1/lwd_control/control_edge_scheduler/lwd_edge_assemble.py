"""两侧共享的配置支撑(§10.3 折入):LwdConfig(env/additional_config 唯一
解析入口)与 is_lwd_prefill_only(模式判定唯一实现)。

边云装配均已收进引擎子类(§10.14 对齐:边 LwdEdgeEngineCore / 云
LwdCloudEngineCore,类选择点出生即子类),本文件不再承载装配/端口
适配/生命周期,仅保留配置解析;云侧装配经 import 复用,内核模块收
plain 值。

配置源(生效配置类优先):vllm_config.lwd_config(vllm/config/lwd.py,
additional_config["lwd_config"] 于 VllmConfig.__post_init__ 解析)提供
enabled/role/mode;传输层字段(pre_out_host 等)取 lwd_config 段,旧
edge_cloud_config 段兼容回退;env(VLLM_ASCEND_LWD_*)只覆盖地址与开关。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
    LWD_PUBLISH_QUEUE_MAX,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_step_core import LwdStepSettings

logger = init_logger(__name__)

LWD_PRE_OUT_PORT_DEFAULT = 5558
LWD_POST_OUT_PORT_DEFAULT = LWD_PRE_OUT_PORT_DEFAULT + 1
# 云结果队列深度(接收线程 -> 引擎主线程;lwd-post-in 写 / 引擎步读)
LWD_RESULT_QUEUE_MAX = 1024

# 云->边步元数据接缝队列容量(§9.12):生产端 lwd-post-in,消费端随数据面落位
LWD_C2E_META_QUEUE_MAX = 1000

_LWD_CONFIG_SECTION = "lwd_config"
# 传输层字段(pre_out_host 等)的历史段名;仅作兼容回退,新增部署用 lwd_config
_LWD_LEGACY_SECTION = "edge_cloud_config"
_LWD_ENV_PREFIX = "VLLM_ASCEND_LWD_"


@dataclass(frozen=True)
class LwdConfig:
    """plain 值配置对象;装配期一次成型,内核模块只收值不再解析。"""

    is_edge_node: bool = True
    pre_out_host: str = "127.0.0.1"
    pre_out_port: int = LWD_PRE_OUT_PORT_DEFAULT
    post_out_port: int = LWD_POST_OUT_PORT_DEFAULT
    post_out_bind: str = "*"
    hello_timeout_s: float = 30.0
    scheduler_name: str = "prefill_first"
    publish_queue_max: int = LWD_PUBLISH_QUEUE_MAX
    zombie_log_interval_s: float = 30.0
    debug: bool = False

    def lwd_pre_out_endpoint(self) -> str:
        """PRE_OUT 端点:云侧 bind 地址,随 HELLO 通告给边(§9.1 数据面方向)。"""
        return f"tcp://{self.pre_out_host}:{self.pre_out_port}"

    def lwd_post_out_bind_endpoint(self) -> str:
        """POST_OUT 端点:边侧 bind(通告面,云经 master_addr 来连)。"""
        return f"tcp://{self.post_out_bind}:{self.post_out_port}"

    def lwd_step_settings(self) -> LwdStepSettings:
        """派生内核 plain 值(§10.3:内核不解析配置)。"""
        return LwdStepSettings(
            debug=self.debug, zombie_log_interval_s=self.zombie_log_interval_s
        )

    @classmethod
    def from_env_and_config(cls, vllm_config) -> LwdConfig:
        """解析 lwd_config 段(兼容回退旧 edge_cloud_config 段);角色取自
        生效配置类 vllm_config.lwd_config(vllm/config/lwd.py)。

        env(VLLM_ASCEND_LWD_*)只覆盖地址与开关。
        """
        section = _lwd_read_section(vllm_config)
        effective = getattr(vllm_config, "lwd_config", None)
        config = cls(
            is_edge_node=effective.is_edge
            if effective is not None
            else str(section.get("role", "edge")) == "edge",
            pre_out_host=str(section.get("pre_out_host", "127.0.0.1")),
            pre_out_port=int(section.get("pre_out_port", LWD_PRE_OUT_PORT_DEFAULT)),
            post_out_port=int(section.get("post_out_port", LWD_POST_OUT_PORT_DEFAULT)),
            post_out_bind=str(section.get("post_out_bind", "*")),
            hello_timeout_s=float(section.get("hello_timeout_s", 30.0)),
            scheduler_name=str(section.get("scheduler", "prefill_first")),
            publish_queue_max=int(
                section.get("publish_queue_max", LWD_PUBLISH_QUEUE_MAX)
            ),
            zombie_log_interval_s=float(section.get("zombie_log_interval_s", 30.0)),
            debug=bool(section.get("debug", False)),
        )
        return _lwd_apply_env_overrides(config)


def is_lwd_prefill_only(vllm_config) -> bool:
    """模式判定唯一实现(全仓 1 处,§2.3/W9);基于生效配置类
    vllm_config.lwd_config(vllm/config/lwd.py,additional_config
    ["lwd_config"] 在 VllmConfig.__post_init__ 解析):enabled 且
    mode == "prefill_only" 才生效;配置类缺位时回退旧
    edge_cloud_config 段。edge/cloud 角色由 LwdConfig 区分。

    主仓既有文件不 import 本函数:上游守卫经装配函数内部分流。
    """
    effective = getattr(vllm_config, "lwd_config", None)
    if effective is not None:
        return effective.enabled and effective.mode == "prefill_only"
    section = _lwd_read_section(vllm_config)
    return section.get("mode") == "prefill_only"


def _lwd_read_section(vllm_config) -> dict:
    """取 additional_config 下传输层字段所在段:lwd_config 优先,
    旧 edge_cloud_config 段回退;缺省/非 dict 均按空段处理。"""
    additional = vllm_config.additional_config or {}
    section = additional.get(_LWD_CONFIG_SECTION)
    if isinstance(section, dict):
        return section
    legacy = additional.get(_LWD_LEGACY_SECTION)
    return legacy if isinstance(legacy, dict) else {}


def _lwd_apply_env_overrides(config: LwdConfig) -> LwdConfig:
    """env 只覆盖部署相关项(地址/端口/调试开关);坏值告警并保留默认,不抛。"""
    host = os.getenv(_LWD_ENV_PREFIX + "PRE_OUT_HOST")
    port = _lwd_read_env_int("PRE_OUT_PORT")
    post_port = _lwd_read_env_int("POST_OUT_PORT")
    post_bind = os.getenv(_LWD_ENV_PREFIX + "POST_OUT_BIND")
    hello_timeout = _lwd_read_env_float("HELLO_TIMEOUT_S")
    debug = os.getenv(_LWD_ENV_PREFIX + "DEBUG")
    return LwdConfig(
        is_edge_node=config.is_edge_node,
        pre_out_host=host if host else config.pre_out_host,
        pre_out_port=port if port is not None else config.pre_out_port,
        post_out_port=post_port if post_port is not None else config.post_out_port,
        post_out_bind=post_bind if post_bind else config.post_out_bind,
        hello_timeout_s=(
            hello_timeout if hello_timeout is not None else config.hello_timeout_s
        ),
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


def _lwd_read_env_float(name: str) -> float | None:
    """读浮点 env;语义与 _lwd_read_env_int 一致(坏值告警保留默认)。"""
    raw = os.getenv(_LWD_ENV_PREFIX + name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("[Lwd] ignore invalid env %s%s=%r", _LWD_ENV_PREFIX, name, raw)
        return None
