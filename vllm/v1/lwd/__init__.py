"""Lwd prefill-only 边云协同内核(仅 vllm 仓,设计文档 docs/refactor/prefill_only_migration.md)。

通信模型(§9 新标准):单向 边 -> 云。
  控制面 PRE_OUT(notify/add_request/abort, ZMQ)+ 数据面 isend(tag=BASE+seqno);
  无结果面、无消费水位回传、无快路径直投。

扩展模型(§9.8):对 EngineCore 的唯一扩展点是 step 接口。
  LwdStepCore(执行类抽象,唯一接口 step_with_batch_queue)<- LwdEdgeCore / LwdCloudCore;
  装配期把专用 core 赋给 EngineCore.step_wrapper(默认 None),
  core.py 的 step_with_batch_queue 顶部守卫委托,step_fn 接线(core.py:218)不变。

分块模型(§9.9):不自造分割,复用原生 Scheduler.schedule() 的 chunked prefill
  决策(num_scheduled_tokens/max_num_batched_tokens);内核只做通知/传输/落位。

分层(依赖严格单向,L3 -> L2 -> L1):
  L3 装配: lwd_edge_assemble / lwd_cloud_launch
  L2 内核: lwd_step_core <- lwd_edge_core / lwd_cloud_core;
           lwd_message <- lwd_edge_channel / lwd_cloud_channel <- dispatcher;
           lwd_edge_scheduler / lwd_cloud_phase_scheduler / lwd_cloud_embeds;
           lwd_edge_embed(模块函数:worker 嵌入 + isend)
  支撑:    lwd_mode / lwd_config / lwd_diagnostics / lwd_ports

import 白名单与交互预算(唯一事实源为设计文档 §7/§8.4/§9,变更先改台账再改代码):
  - 内核文件禁止 import: pd_separation / ascend 系符号 / vllm.envs /
    vllm.v1.request / vllm.v1.outputs / vllm.v1.engine.core / tracing;
    getattr 与 os.environ 预算为 0
  - vllm.v1.engine 仅 lwd_message(线上类型 re-export:EngineCoreRequest/
    EngineCoreOutputs)
  - vllm.v1.core.sched.output / async_scheduler 仅两个调度器文件
    (lwd_cloud_phase_scheduler / lwd_edge_scheduler)
  - torch.distributed 仅 lwd_edge_embed / lwd_cloud_embeds
  - 继承例外共 2 个:lwd_cloud_phase_scheduler 与 lwd_edge_scheduler
    均继承 AsyncScheduler(§7.5/§9.10);§9.8 后内核不再继承 EngineCore
    (边/云 core 经 LwdEnginePort 触达,适配器 LwdEnginePortAdapter
    是台账登记的属性容忍点)
"""
