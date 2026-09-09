"""边侧 L3 装配:EngineCore.__init__ 守卫调用的单点开关(§8.2/§9.8)。

单向语义:无云结果/水位 drain,步进逻辑全部收编进
LwdEdgeCore.step_with_batch_queue;本文件只负责装配与生命周期,
原 6 个 monkey-patch 由 core.py in-tree 守卫分支替代。

双面拓扑(§9.1):边 bind POST_OUT(云经 master_addr 来连,承载
周期 HELLO),PRE_OUT 延迟连接、目标端点由 HELLO 通告唯一决定
(决策 B:边不读 pre_out_host 做连接);装配阻塞等首条 HELLO,
超时 fail-fast。

本文件同时承载两侧共享的配置支撑(§10.3 折入):LwdConfig
(env/additional_config 唯一解析入口)与 is_lwd_prefill_only
(模式判定唯一实现);云侧装配经 import 复用,内核模块收 plain 值。
"""

from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
    LWD_PUBLISH_QUEUE_MAX,
    LwdControlPublisher,
)
from vllm.v1.lwd_control.control_communication.lwd_control_subscriber import (
    LwdControlSubscriber,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LwdHelloNotify,
    lwd_decode_cloud_notify,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_core import LwdEdgeCore
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_step_core import LwdStepSettings

logger = init_logger(__name__)

LWD_PRE_OUT_PORT_DEFAULT = 5558
LWD_POST_OUT_PORT_DEFAULT = LWD_PRE_OUT_PORT_DEFAULT + 1
# 云结果队列深度(接收线程 -> 引擎主线程;lwd-post-in 写 / LwdEdgeCore 步读)
LWD_RESULT_QUEUE_MAX = 1024

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

    PO 时:bind POST_OUT 订阅面,建延迟连接的 PRE_OUT 发布面,起
    lwd-post-in 发现线程并阻塞等首条 HELLO(超时 fail-fast,§9.1);
    scheduler_cls 以 partial(LwdEdgeScheduler, publisher=...)注入
    (调度器即控制面出口,§9.12),建 LwdEdgeCore 并赋给
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
    receiver = _lwd_edge_build_post_out(config)
    publisher = LwdControlPublisher(
        None, bind=False, queue_max=config.publish_queue_max
    )
    hello_event = threading.Event()
    discovery = threading.Thread(
        target=_lwd_edge_discovery_loop,
        args=(receiver, publisher, hello_event),
        name="lwd-post-in",
        daemon=True,
    )
    discovery.start()
    if not hello_event.wait(config.hello_timeout_s):
        _lwd_edge_shutdown_planes(receiver, publisher)
        raise RuntimeError(
            f"[Lwd] edge assembly failed: no cloud HELLO within "
            f"{config.hello_timeout_s}s on POST_OUT "
            f"(bind {config.lwd_post_out_bind_endpoint()}; check cloud "
            f"master_addr connectivity and POST_OUT port)"
        )
    _lwd_edge_install_scheduler(engine_core, publisher)
    engine_core.lwd_edge_post_out_receiver = receiver
    # 云结果队列(Step 2 仅建队列):接收线程(lwd-post-in)按类型分发写入
    # (LwdResultNotify 入队,实现待后续 Step),LwdEdgeCore 步首 drain 读取
    # (待后续 Step);队满语义为结果不可丢(写入侧自旋重试,随写入实现落位)。
    engine_core.lwd_edge_result_queue = queue.Queue(maxsize=LWD_RESULT_QUEUE_MAX)
    engine_core.step_wrapper = LwdEdgeCore(
        LwdEdgeEnginePortAdapter(engine_core), config.lwd_step_settings()
    )
    return True


def _lwd_edge_build_post_out(config: LwdConfig) -> LwdControlSubscriber:
    """bind POST_OUT 订阅面(云经 master_addr 主动来连;边不预知云地址)。"""
    return LwdControlSubscriber(
        config.lwd_post_out_bind_endpoint(),
        bind=True,
        decoder=lwd_decode_cloud_notify,
    )


def _lwd_edge_discovery_loop(
    receiver: LwdControlSubscriber,
    publisher: LwdControlPublisher,
    hello_event: threading.Event,
) -> None:
    """发现线程(lwd-post-in):消费 POST_OUT,HELLO -> retarget PRE_OUT。

    常驻运行(不只首发):云换址重启后周期 HELLO 仍能驱动 retarget
    先连新断旧(§9.1);retarget 队满失败靠周期重发自愈。
    """
    while not receiver.closed:
        msg = receiver.recv(timeout_ms=5000)
        if msg is None:
            continue
        if not isinstance(msg, LwdHelloNotify):
            # 协议分面:POST_OUT 目前只承载 HELLO,坏帧已被订阅层丢弃
            logger.warning("[Lwd] drop unexpected POST_OUT frame %r", type(msg))
            continue
        endpoint = f"tcp://{msg.pre_out_host}:{msg.pre_out_port}"
        if not hello_event.is_set():
            logger.info("[Lwd] cloud discovered via HELLO: PRE_OUT -> %s", endpoint)
        if not publisher.retarget(endpoint):
            # 队满丢令:周期重发(5s)会再来,下条 HELLO 重试
            logger.warning("[Lwd] PRE_OUT retarget deferred: publish queue full")
        hello_event.set()


def _lwd_edge_shutdown_planes(
    receiver: LwdControlSubscriber | None, publisher: LwdControlPublisher | None
) -> None:
    """两面关停(幂等):receiver 先关(断输入),publisher 收尾。"""
    if receiver is not None:
        receiver.shutdown()
    if publisher is not None:
        publisher.shutdown()


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
    """通道关停(core.py shutdown 守卫分支调用;装配层掌生命周期)。

    超时 fail-fast 路径已在装配点自清理;此处覆盖正常关停:
    PRE_OUT(publisher 挂在调度器上)与 POST_OUT(receiver 挂在
    engine_core 属性上)两面都关。
    """
    scheduler = engine_core.scheduler
    publisher = None
    if isinstance(scheduler, LwdEdgeScheduler):
        publisher = scheduler.lwd_edge_publisher
    receiver = getattr(engine_core, "lwd_edge_post_out_receiver", None)
    _lwd_edge_shutdown_planes(receiver, publisher)
