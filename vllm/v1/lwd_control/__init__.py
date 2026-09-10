"""Lwd prefill-only 控制面内核(仅 vllm 仓)。

设计文档:docs/refactor/prefill_only_migration.md。

范围(§9.12/§10):本目录收编两仓全部控制面与控制面通信;数据面
(嵌入前向、tag 直传、接收落位、消费释放)不在本目录,由其落位侧经
既有接缝对接:
  - 边侧:LwdEdgeScheduler.lwd_edge_update_progress(执行量来源)
  - 云侧:LwdCloudEngineCore 的 PRE_OUT 接收泵(首预告门 + 接收登记)/
    _lwd_build_request(请求侧挂载点)

通信模型(§9,双面拓扑):数据传输仅 边 -> 云 的 PRE_OUT
  (notify/add_request/abort,ZMQ PUSH:云 bind,边经 HELLO 发现后 connect);
  云 -> 边 的 POST_OUT(ZMQ PUSH:边 bind wildcard,云经 master_addr
  connect)承载 HELLO 发现通告(云端点唯一事实源,决策 B)与
  LwdC2eNotify 步元数据(云->边唯一载荷,兼结果回传驱动);
  无水位、无快路径。

扩展模型(§10.14 边云同构):两侧引擎均经 core.py run_engine_core
  类选择点(lwd_resolve_engine_cls)出生即子类——边 LwdEdgeEngineCore
  (通信面装配 + 调度器注入 + step/add/abort/shutdown 覆写)、云
  LwdCloudEngineCore(ZMQ 收发 + 首预告门,覆写原生 socket IO 线程入口
  process_input_sockets,父线程照跑父类原版,core.py 零改动):
  过门请求转 Request 投 input_queue 走原生 ADD/ABORT 分发
  (_handle_client_request 零改动)。
  上游接线 additive 守卫(§10.1)收敛为 2 处:core.py run_engine_core
  类选择点(经 lwd_resolve_engine_cls 分流)+ serve.py 入口守卫
  (lwd_serve_guard,云角色注入相位调度器)。

分块模型(§9.9):不自造分割,复用原生 Scheduler.schedule() 的 chunked
  prefill 决策;单请求组批约束(§9.9 修订):两侧 prefill 批最多含一个
  请求(容器交换隐藏其余,单请求内 chunked/KV 决策照旧)——数据面
  chunk 流按请求连续,云侧 fill 无跨请求 head-of-line;边侧纯 prefill =
  原生调度 + 完结即本地终结(lwd_edge_scheduler);云侧相位 = 原生调度的
  队列手术复用(lwd_cloud_phase_scheduler,空步不重复调 schedule)。

目录布局(§10.3 支撑折入既有文件,框架不增文件):
  control_communication/(传输层,side-agnostic,只认方向不认边/云;
             侧别与 bind/connect 是装配期 wiring):
             lwd_notify(通知定义 + 编解码,纯协议,§10.7)/
             lwd_control_communicator(纯收发句柄,无线程)
             + lwd_control_publisher(OUTBOUND)/ lwd_control_subscriber(INBOUND)
  control_scheduler/(分组入口,仅包说明)
  control_edge_scheduler/(边侧):lwd_edge_scheduler;lwd_edge_assemble(LwdConfig +
             模式判定唯一实现,纯配置支撑);lwd_edge_engine(类选择点
             注入的边 EngineCore 子类:通信面装配 + 调度器注入 + 引擎
             接口覆写)为 L3
  control_cloud_scheduler/(云侧):lwd_cloud_phase_scheduler(纯相位
             排批;准入固定 immediate 直进,§10.10/§10.14);
             lwd_cloud_engine(类选择点注入的云 EngineCore 子类:ZMQ
             收发 + 首预告门 + 请求构建)为 L3

import 白名单与交互预算(唯一事实源为设计文档 §7/§8.4/§9/§10,
变更先改台账再改代码;检查脚本 tools/lwd_check_budget.py):
  - 内核文件禁止 import: pd_separation / ascend 系符号 / vllm.envs /
    vllm.v1.request / vllm.v1.outputs / vllm.v1.engine(.core) /
    vllm.v1.executor / vllm.tracing / torch.distributed(零数据面);
    内核 getattr = 0
  - os.environ/os.getenv 仅 lwd_edge_assemble(LwdConfig 定义处是唯一
    env 入口,内核不读 env)
  - 例外(台账登记):两个调度器文件可 import vllm.v1.request 的
    RequestStatus——AsyncScheduler 继承面的既有传递依赖,不新增依赖边;
    云引擎文件 import Request/SamplingParams(请求构建唯一交互点)
  - vllm.v1.engine(.core) 仅 lwd_notify(TYPE_CHECKING re-export:
    EngineCoreOutputs)与两个引擎子类(运行时:EngineCoreProc 继承,
    §10.14 边云同构)
  - vllm.v1.core.sched.*(output/async_scheduler/request_queue)仅两个
    调度器文件(引擎子类仅 TYPE_CHECKING 引 SchedulerOutput)
  - 继承例外共 3 个:lwd_cloud_phase_scheduler 与 lwd_edge_scheduler
    均继承 AsyncScheduler(§7.5/§9.10);lwd_cloud_engine 与
    lwd_edge_engine 继承 EngineCoreProc(§10.14,类选择点注入,装配/
    触达收进引擎子类自身)
"""

from __future__ import annotations


def lwd_resolve_engine_cls(vllm_config):
    """core.py run_engine_core 类选择点的分流点(§10.14 边云同构):

    - 云角色返回 LwdCloudEngineCore(出生即云形态,装配收进子类 __init__)
    - 边角色返回 LwdEdgeEngineCore(出生即边形态:通信面装配 + 调度器注入)
    - 其余返回 None(调用方用原生 EngineCoreProc)
    """
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
    """serve.py run_headless 入口守卫(§10.1):云角色构造期注入相位调度器。

    位置在 vllm_config 建成之后、任何引擎构造之前 —— 写
    scheduler_config.scheduler_cls(单类 LwdCloudPhaseScheduler,相位由
    调度器构造期经 LwdConfig 自解析,§10.15;类对象跨进程按模块引用
    序列化,EngineCore.__init__ core.py:139 get_scheduler_cls 构造期
    解析),引擎即以相位调度器出生,无需事后整实例替换(§10.8)。
    云引擎类由 run_engine_core 的 lwd_resolve_engine_cls 子进程内
    解析(§10.14),不在此处。边角色不经此处:边调度器由
    LwdEdgeEngineCore.__init__ 自注入(裸类 + publisher 构造后回填)。

    同处执行云角色部署校验(_lwd_cloud_deploy_guard):master_addr
    非空、pre_out_host 非 0.0.0.0(HELLO 通告值必须可路由,§9.1)。
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
    _lwd_cloud_deploy_guard(vllm_config, config)
    vllm_config.scheduler_config.scheduler_cls = LwdCloudPhaseScheduler
    init_logger(__name__).info(
        "[Lwd] prefill_only cloud: phase scheduler injected (construction-time)"
    )


def _lwd_cloud_deploy_guard(vllm_config, config) -> None:
    """云角色部署校验(fail-fast,serve 入口即拦,不等到运行期):

    - master_addr 非空:POST_OUT 经它连边,缺了通告面整条不成立;
    - pre_out_host 非 0.0.0.0:该值随 HELLO 通告给边作连接目标,
      通配 bind 地址不可路由(决策 B 的配套约束)。
    """
    from vllm.logger import init_logger

    master_addr = vllm_config.parallel_config.master_addr
    if not master_addr:
        raise ValueError(
            "[Lwd] prefill_only cloud requires --master-addr (POST_OUT connect)"
        )
    if config.pre_out_host == "0.0.0.0":
        raise ValueError(
            "[Lwd] prefill_only cloud pre_out_host=0.0.0.0 is not announceable; "
            "set a routable IP (VLLM_ASCEND_LWD_PRE_OUT_HOST)"
        )
    if config.pre_out_host == "127.0.0.1" and master_addr not in (
        "127.0.0.1",
        "localhost",
    ):
        init_logger(__name__).warning(
            "[Lwd] cloud announces pre_out_host=127.0.0.1 but master_addr=%s "
            "is remote; edge will fail to reach PRE_OUT unless same host",
            master_addr,
        )
