"""Lwd prefill-only 控制面内核(仅 vllm 仓)。

设计文档:docs/refactor/prefill_only_migration.md。

范围(§9.12/§10):本目录收编两仓全部控制面与控制面通信;数据面
(嵌入前向、tag 直传、接收落位、消费释放)不在本目录,由其落位侧经
既有接缝对接:
  - 边侧:LwdEdgeScheduler.lwd_edge_update_progress(执行量来源)
  - 云侧:LwdCloudCore._lwd_handle_range_notify(接收登记)/
    lwd_cloud_assemble._lwd_cloud_build_admit_request(请求侧挂载点)

通信模型(§9):单向 边 -> 云,仅控制面 PRE_OUT
  (notify/add_request/abort,ZMQ PUSH/PULL);无结果面、无水位、无快路径。

扩展模型(§9.8):对 EngineCore 的唯一扩展点是 step 接口。
  LwdStepCore <- LwdEdgeCore / LwdCloudCore;装配期把专用 core 赋给
  EngineCore.step_wrapper(默认 None),core.py step 守卫委托。
  上游接线共 5 处 additive 守卫(§10.1):core.py 4 处
  (step_wrapper 字段 / __init__ 尾装配点 / step 顶部守卫 / shutdown 守卫,
  经本模块 lwd_try_assemble / lwd_shutdown 分流)+ serve.py 入口守卫
  (lwd_serve_guard)。

分块模型(§9.9):不自造分割,复用原生 Scheduler.schedule() 的 chunked
  prefill 决策;边侧纯 prefill = 原生调度 + 完结即本地终结
  (lwd_edge_scheduler);云侧相位 = 原生调度的队列手术复用
  (lwd_cloud_phase_scheduler,空步不重复调 schedule)。

目录布局(§10.3 支撑折入既有文件,框架不增文件):
  control_communication/(传输层,side-agnostic,只认方向不认边/云;
             侧别与 bind/connect 是装配期 wiring):
             lwd_message(线上消息 + 编解码,纯协议,§10.7)/
             lwd_control_communicator(公共基类)
             + lwd_control_publisher(OUTBOUND)/ lwd_control_subscriber(INBOUND)
  control_scheduler/:     lwd_step_core(step 抽象 + 端口/视图协议 +
             LwdStepSettings/LwdLog) <- lwd_edge_core / lwd_cloud_core;
             lwd_edge_scheduler / lwd_cloud_phase_scheduler /
             lwd_cloud_admission;lwd_edge_assemble(装配 + LwdConfig +
             模式判定唯一实现 + 边侧适配器)/ lwd_cloud_assemble(装配 +
             云侧/视图适配器)为 L3(违禁 import 容忍点)

import 白名单与交互预算(唯一事实源为设计文档 §7/§8.4/§9/§10,
变更先改台账再改代码;检查脚本 tools/lwd_check_budget.py):
  - 内核文件禁止 import: pd_separation / ascend 系符号 / vllm.envs /
    vllm.v1.request / vllm.v1.outputs / vllm.v1.engine(.core) /
    vllm.v1.executor / vllm.tracing / torch.distributed(零数据面);
    内核 getattr = 0
  - os.environ/os.getenv 仅 lwd_edge_assemble(LwdConfig 定义处是唯一
    env 入口,§7.3-C3;内核收 LwdStepSettings plain 值)
  - 例外(台账登记):两个调度器文件可 import vllm.v1.request 的
    RequestStatus——AsyncScheduler 继承面的既有传递依赖,不新增依赖边;
    云装配文件 import Request/SamplingParams(请求构建唯一交互点)
  - vllm.v1.engine 仅 lwd_message(TYPE_CHECKING
    re-export:EngineCoreOutputs)
  - vllm.v1.core.sched.*(output/async_scheduler/request_queue)仅两个
    调度器文件;kv_cache_utils 仅两个装配文件(L3)
  - 继承例外共 2 个:lwd_cloud_phase_scheduler 与 lwd_edge_scheduler
    均继承 AsyncScheduler(§7.5/§9.10);内核不继承 EngineCore,
    触达一律经两侧装配文件中的端口适配器(台账登记的 engine_core
    属性容忍点,§7.3-C5)
"""

from __future__ import annotations


def lwd_try_assemble(engine_core) -> bool:
    """core.py __init__ 尾装配守卫的分流点(§10.1):边角色/云角色各装各的。"""
    from vllm.v1.lwd_control.control_scheduler.lwd_cloud_assemble import (
        lwd_cloud_try_assemble,
    )
    from vllm.v1.lwd_control.control_scheduler.lwd_edge_assemble import (
        lwd_edge_try_assemble,
    )

    return lwd_edge_try_assemble(engine_core) or lwd_cloud_try_assemble(engine_core)


def lwd_shutdown(engine_core) -> None:
    """core.py shutdown 守卫的路由点:按装配形态关停通道(边 publisher/云 subscriber)。"""
    from vllm.v1.lwd_control.control_scheduler.lwd_cloud_assemble import (
        LwdCloudCore,
        lwd_cloud_shutdown,
    )
    from vllm.v1.lwd_control.control_scheduler.lwd_edge_assemble import (
        lwd_edge_shutdown,
    )

    lwd_edge_shutdown(engine_core)
    if isinstance(engine_core.step_wrapper, LwdCloudCore):
        lwd_cloud_shutdown(engine_core.step_wrapper)


def lwd_serve_guard(vllm_config) -> None:
    """serve.py run_headless 入口守卫(§10.1):PO 模式角色标记。

    控制面装配发生在 EngineCore 进程内(__init__ 尾守卫),云角色由
    headless 原生路径拉起;运行时底座(parallel_state 角色/executor
    拓扑)延后(§10.1),本守卫是后续云入口分叉的唯一挂点。
    """
    from vllm.logger import init_logger
    from vllm.v1.lwd_control.control_scheduler.lwd_edge_assemble import (
        is_lwd_prefill_only,
    )

    if is_lwd_prefill_only(vllm_config):
        init_logger(__name__).info(
            "[Lwd] prefill_only mode enabled: engines self-assemble on startup"
        )
