"""Lwd prefill-only 控制面内核(仅 vllm 仓)。

设计文档:docs/refactor/prefill_only_migration.md。

范围(§9.12/§10):本目录收编两仓全部控制面与控制面通信;数据面
(嵌入前向、tag 直传、接收落位、消费释放)不在本目录,由其落位侧经
既有接缝对接:
  - 边侧:LwdEdgeScheduler.lwd_edge_update_progress(执行量来源)
  - 云侧:LwdCloudEngineCore 的 PRE_OUT 接收泵(首预告门 + 接收登记)/
    _lwd_build_request(请求侧挂载点)

通信模型(§9):单向 边 -> 云,仅控制面 PRE_OUT
  (notify/add_request/abort,ZMQ PUSH/PULL);无结果面、无水位、无快路径。

扩展模型(§9.8/§10.12/§10.14):边侧对 EngineCore 的扩展点是 step 接口
  (LwdStepCore <- LwdEdgeCore,装配期赋 EngineCore.step_wrapper,
  core.py step 守卫委托);云侧零 step 依赖 —— 引擎经 core.py
  run_engine_core 类选择点(lwd_resolve_engine_cls)出生即云形态;
  云侧收发接入点 = 覆写原生 socket IO 线程入口 process_input_sockets
  (super 引用父类原版照跑,core.py 零改动):本线程跑边侧 PRE_OUT
  循环,过门请求转 Request 投 input_queue 走原生 ADD/ABORT 分发
  (_handle_client_request 零改动);首预告门住引擎子类、归该 IO 线程
  独占,调度器只经 input_queue 被主循环碰,空闲唤醒由原生机制自然
  解决(IO 线程 put ADD 唤醒主循环的 input_queue.get()),无轮询。
  上游接线 additive 守卫(§10.1):core.py 3 处(step_wrapper 字段 /
  __init__ 尾装配点 / step 顶部守卫 / shutdown 守卫,经本模块
  lwd_try_assemble / lwd_shutdown 分流;run_engine_core 类选择点,
  经 lwd_resolve_engine_cls 分流)+ serve.py 入口守卫(lwd_serve_guard)。

分块模型(§9.9):不自造分割,复用原生 Scheduler.schedule() 的 chunked
  prefill 决策;边侧纯 prefill = 原生调度 + 完结即本地终结
  (lwd_edge_scheduler);云侧相位 = 原生调度的队列手术复用
  (lwd_cloud_phase_scheduler,空步不重复调 schedule)。

目录布局(§10.3 支撑折入既有文件,框架不增文件):
  control_communication/(传输层,side-agnostic,只认方向不认边/云;
             侧别与 bind/connect 是装配期 wiring):
             lwd_notify(通知定义 + 编解码,纯协议,§10.7)/
             lwd_control_communicator(纯收发句柄,无线程)
             + lwd_control_publisher(OUTBOUND)/ lwd_control_subscriber(INBOUND)
  control_scheduler/(分组入口,仅包说明)
  control_edge_scheduler/(边侧):lwd_step_core(step 抽象 + 端口/视图
             协议 + LwdStepSettings/LwdLog) <- lwd_edge_core;
             lwd_edge_scheduler;lwd_edge_assemble(装配 + LwdConfig +
             模式判定唯一实现 + 边侧适配器)为 L3(违禁 import 容忍点)
  control_cloud_scheduler/(云侧):lwd_cloud_phase_scheduler(纯相位
             排批;准入固定 immediate 直进,§10.10/§10.14);
             lwd_cloud_engine(类选择点注入的云 EngineCore 子类:ZMQ
             收发 + 首预告门 + 请求构建)为 L3;
             lwd_cloud_assemble/lwd_cloud_core 已删(§10.12/§10.14)

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
    云引擎文件 import Request/SamplingParams(请求构建唯一交互点)
  - vllm.v1.engine 仅 lwd_notify(TYPE_CHECKING re-export:
    EngineCoreOutputs)与 lwd_cloud_engine(运行时:EngineCoreProc
    继承 + ADD/ABORT 请求元组类型,§10.14)
  - vllm.v1.core.sched.*(output/async_scheduler/request_queue)仅两个
    调度器文件;kv_cache_utils 仅两个装配文件(L3)
  - 继承例外共 3 个:lwd_cloud_phase_scheduler 与 lwd_edge_scheduler
    均继承 AsyncScheduler(§7.5/§9.10);lwd_cloud_engine 继承
    EngineCoreProc(§10.14,类选择点注入,装配/触达收进引擎子类自身)
"""

from __future__ import annotations


def lwd_try_assemble(engine_core) -> bool:
    """core.py __init__ 尾装配守卫的分流点(§10.1):仅边角色。

    云侧不经此处(§10.14):run_engine_core 类选择点(lwd_resolve_engine_cls)
    使云引擎出生即 LwdCloudEngineCore,装配收进子类 __init__。
    """
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
        lwd_edge_try_assemble,
    )

    return lwd_edge_try_assemble(engine_core)


def lwd_resolve_engine_cls(vllm_config):
    """core.py run_engine_core 类选择点的分流点(§10.14):云 PO 引擎返回
    LwdCloudEngineCore(出生即云形态,装配收进子类 __init__),其余返回
    None(调用方用原生 EngineCoreProc)。边角色不换类,仍走 __init__ 尾
    lwd_edge_try_assemble。
    """
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
        LwdConfig,
        is_lwd_prefill_only,
    )

    if not is_lwd_prefill_only(vllm_config):
        return None
    config = LwdConfig.from_env_and_config(vllm_config)
    if config.is_edge_node:
        return None
    from vllm.logger import init_logger
    from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
        LwdCloudEngineCore,
    )

    init_logger(__name__).info(
        "[Lwd] prefill_only cloud: engine class selected (LwdCloudEngineCore)"
    )
    return LwdCloudEngineCore


def lwd_shutdown(engine_core) -> None:
    """core.py shutdown 守卫的路由点:边角色通道关停。

    云侧不经此处:step_wrapper 恒为 None(§10.12),关停经原生
    scheduler.shutdown 钩子转发停桥(相位调度器 shutdown 覆写)。
    """
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
        lwd_edge_shutdown,
    )

    lwd_edge_shutdown(engine_core)


def lwd_serve_guard(vllm_config) -> None:
    """serve.py run_headless 入口守卫(§10.1):云角色构造期注入相位调度器。

    位置在 vllm_config 建成之后、任何引擎构造之前 —— 写
    scheduler_config.scheduler_cls(单类 LwdCloudPhaseScheduler,相位由
    调度器构造期经 LwdConfig 自解析,§10.15;类对象跨进程按模块引用
    序列化,EngineCore.__init__ core.py:139 get_scheduler_cls 构造期
    解析),引擎即以相位调度器出生,无需事后整实例替换(§10.8)。
    云引擎类由 run_engine_core 的 lwd_resolve_engine_cls 子进程内
    解析(§10.14),不在此处。边角色不注入:边调度器需 publisher
    构造注入,仍走 __init__ 尾 lwd_edge_try_assemble。
    """
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
    vllm_config.scheduler_config.scheduler_cls = LwdCloudPhaseScheduler
    init_logger(__name__).info(
        "[Lwd] prefill_only cloud: phase scheduler injected (construction-time)"
    )
