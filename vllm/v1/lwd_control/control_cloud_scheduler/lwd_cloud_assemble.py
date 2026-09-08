"""云侧 L3 装配:进程入口 + 请求准入装配(违禁 import 只允许本文件,§7.2/§9.8)。

EngineCore 的构建仍走原生 headless 路径(serve.py 守卫负责),本文件
在真实 EngineCore 建成后完成 Lwd 装配;云侧端口适配器与调度器视图
适配器落位本文件(§10.3),LwdConfig/模式判定复用 lwd_edge_assemble。
"""

from __future__ import annotations

from collections.abc import Callable

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils.system_utils import set_process_title
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_core import LwdCloudCore
from vllm.v1.lwd_control.control_communication.lwd_control_subscriber import (
    LwdControlSubscriber,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LwdRequestNotify,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
    LwdConfig,
    is_lwd_prefill_only,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_step_core import (
    LwdCloudSchedulerView,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


class LwdCloudEnginePortAdapter:
    """云侧端口适配器(§10.3 落位装配文件):engine_core 触达 + abort + 视图。

    增强步体(lwd_cloud_core.lwd_native_step_bq_prefill_only)直接以
    engine_core 为入参,不回调原生步体、无 wrapper 翻转(§10.8)。
    台账登记的 engine_core 属性容忍点(§7.3-C5)。
    """

    def __init__(self, engine_core) -> None:
        self._engine_core = engine_core

    def lwd_engine_core(self):
        """增强步体的 EngineCore 触达(LwdCloudCore._process_engine_step 用)。"""
        return self._engine_core

    def lwd_abort_requests(self, request_ids: list[str]) -> None:
        self._engine_core.abort_requests(request_ids)

    def lwd_scheduler_view(self) -> LwdCloudSchedulerView:
        return LwdCloudSchedulerViewAdapter(self._engine_core.scheduler)


class LwdCloudSchedulerViewAdapter:
    """包住 engine_core.scheduler 的只读快照(§7.3-C1,准入策略输入)。

    台账登记的调度器内部容忍点:只读 waiting/skipped_waiting/requests
    三个公共容器,不触碰调度器私有状态。
    """

    def __init__(self, scheduler) -> None:
        self._scheduler = scheduler

    def lwd_unfinished_count(self) -> int:
        return self._scheduler.get_num_unfinished_requests()

    def lwd_waiting_count(self) -> int:
        return len(self._scheduler.waiting) + len(self._scheduler.skipped_waiting)

    def lwd_request_progress(self):
        """产出 (request_id, num_computed_tokens, num_prompt_tokens)。"""
        for request in self._scheduler.requests.values():
            if not request.is_finished():
                yield (
                    request.request_id,
                    request.num_computed_tokens,
                    request.num_prompt_tokens,
                )


def lwd_cloud_main(args, engine_core) -> bool:
    """云进程入口(serve.py run_headless 守卫分支调用);装配三段见下(§2.6)。

    守卫先按 headless 原生路径构建 EngineCore,再以本入口完成 Lwd 装配;
    与 core.py __init__ 尾守卫(lwd_try_assemble 分流)幂等共存。
    """
    _lwd_cloud_init_process(args)
    return lwd_cloud_try_assemble(engine_core)


def lwd_cloud_try_assemble(engine_core) -> bool:
    """云侧装配点(幂等):非 PO / 边角色立即返回 False,零副作用。

    PO 云角色:建 PRE_OUT 订阅通道 -> 换装相位调度器 -> 建 LwdCloudCore
    并赋给 engine_core.step_wrapper(§9.8 委托点)。
    """
    if engine_core.step_wrapper is not None:
        return True
    if not is_lwd_prefill_only(engine_core.vllm_config):
        return False
    config = LwdConfig.from_env_and_config(engine_core.vllm_config)
    if config.is_edge_node:
        return False
    if engine_core.batch_queue is None:
        # 无 batch queue 时 step_fn 绑定同步 step(core.py:221),step_wrapper
        # 守卫(core.py:509)永不触发,云侧 drain 静默失效 —— 装配期 fail-fast
        # (源装配断言同款:requires the batch queue, max_concurrent_batches > 1)
        raise RuntimeError(
            "[Lwd] cloud prefill_only requires the batch queue "
            "(async_scheduling / max_concurrent_batches > 1)"
        )
    subscriber = _lwd_cloud_connect_planes(config)
    # 请求构建唯一交互点绑定给调度器(§10.11):门/暂存/准入均在调度器,
    # 工厂只负责 Request 构建(违禁 import 容忍点,§7.2)。
    engine_core.scheduler.lwd_cloud_bind_request_factory(
        _lwd_cloud_build_request(engine_core)
    )
    engine_core.step_wrapper = _lwd_cloud_build_core(engine_core, subscriber, config)
    return True


def lwd_cloud_shutdown(cloud_core) -> None:
    """关停云侧订阅通道(core.py shutdown 守卫经 lwd_shutdown 路由)。

    包内生命周期辅助:cores 不暴露额外公共接口(§9.8),通道生命周期
    由装配层掌管。
    """
    cloud_core._lwd_subscriber.shutdown()


def _lwd_cloud_init_process(args) -> None:
    """进程级初始化(角色标记/信号/日志)。"""
    del args  # 信号注册由原生 headless 入径负责,此处只做角色标记
    set_process_title("vllm::EngineCore::LwdCloud")
    logger.info("[Lwd] cloud process assembling (prefill_only subscriber)")


def _lwd_cloud_connect_planes(config: LwdConfig) -> LwdControlSubscriber:
    """建控制面订阅通道(单向,仅 PRE_OUT;无结果面,§9.1)。

    bind/connect 是装配期 wiring(传输层 side-agnostic):云侧 bind,
    边侧 connect 于同一端点。
    """
    return LwdControlSubscriber(config.lwd_pre_out_endpoint(), bind=True)


def _lwd_cloud_build_core(
    engine_core, subscriber: LwdControlSubscriber, config: LwdConfig
) -> LwdCloudCore:
    """装配云侧执行类。

    调度接线:scheduler 已在构造期注入(serve 守卫按 (scheduler_name,
    admission_name) 二维写 scheduler_config.scheduler_cls,相位排批 +
    相位准入 + 首预告门同体,§10.10/§10.11;request factory 由
    try_assemble 绑定);本函数只装 step 委托。
    """
    port = LwdCloudEnginePortAdapter(engine_core)
    return LwdCloudCore(
        subscriber=subscriber,
        engine_port=port,
        settings=config.lwd_step_settings(),
    )


def _lwd_cloud_build_request(engine_core) -> Callable[[LwdRequestNotify], Request]:
    """唯一请求构建点:Request 构建/block_hasher 全收于此(§7.3-C1)。

    数据面挂载(prompt_embeds 视图)由数据面落位侧在此对接(§9.12);
    产物经调度器 add_request 进首预告门后的暂存/准入(§10.10/§10.11)。
    """
    block_hasher = engine_core.request_block_hasher

    def _build(wire: LwdRequestNotify) -> Request:
        return Request(
            request_id=wire.request_id,
            # 占位 token:云侧调度只看长度,真值由边侧 embeds 经数据面提供(§9.5)
            prompt_token_ids=[0] * wire.num_prompt_tokens,
            sampling_params=SamplingParams(max_tokens=wire.max_tokens),
            pooling_params=None,
            block_hasher=block_hasher,
        )

    return _build
