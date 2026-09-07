"""Lwd prefill-only 控制面内核(仅 vllm 仓,设计文档 docs/refactor/prefill_only_migration.md)。

范围(§9.12):本目录只做控制面。数据面(嵌入前向、tag 直传、接收落位、
消费释放)不在本目录,由其落位侧经既有接缝对接:
  - 边侧:LwdEdgeScheduler.lwd_edge_update_progress(执行量来源)
  - 云侧:LwdCloudCore._lwd_handle_embed_notify(接收登记)/
    _lwd_cloud_admit_request(请求侧挂载)

通信模型(§9 新标准):单向 边 -> 云,仅控制面 PRE_OUT
  (notify/add_request/abort, ZMQ);无结果面、无水位、无快路径。

扩展模型(§9.8):对 EngineCore 的唯一扩展点是 step 接口。
  LwdStepCore(执行类抽象,唯一接口 step_with_batch_queue)<- LwdEdgeCore /
  LwdCloudCore;装配期把专用 core 赋给 EngineCore.step_wrapper(默认 None),
  core.py 的 step_with_batch_queue 顶部守卫委托,step_fn 接线(core.py:218)不变。

分块模型(§9.9):不自造分割,复用原生 Scheduler.schedule() 的 chunked prefill
  决策;dispatcher 职责已并入 LwdEdgeScheduler(§9.12)。

分层(依赖严格单向,L3 -> L2 -> L1):
  L3 装配: lwd_edge_assemble / lwd_cloud_launch
  L2 内核: lwd_step_core <- lwd_edge_core / lwd_cloud_core;
           lwd_edge_scheduler / lwd_cloud_phase_scheduler / lwd_cloud_admission;
           lwd_message <- lwd_edge_channel / lwd_cloud_channel
  支撑:    lwd_mode / lwd_config / lwd_diagnostics / lwd_ports

import 白名单与交互预算(唯一事实源为设计文档 §7/§8.4/§9,变更先改台账再改代码):
  - 内核文件禁止 import: pd_separation / ascend 系符号 / vllm.envs /
    vllm.v1.request / vllm.v1.outputs / vllm.v1.engine.core / tracing /
    torch.distributed(本目录零数据面);getattr 与 os.environ 预算为 0
  - vllm.v1.engine 仅 lwd_message(线上类型 re-export:EngineCoreRequest/
    EngineCoreOutputs)
  - vllm.v1.core.sched.output / async_scheduler 仅两个调度器文件
    (lwd_cloud_phase_scheduler / lwd_edge_scheduler)
  - 继承例外共 2 个:lwd_cloud_phase_scheduler 与 lwd_edge_scheduler
    均继承 AsyncScheduler(§7.5/§9.10);内核不继承 EngineCore
    (边/云 core 经 LwdEnginePort 触达,适配器 LwdEnginePortAdapter
    是台账登记的属性容忍点)
"""
