"""控制面调度/执行层:step 载体 + 边/云调度器 + 准入策略 + L3 装配。

依赖方向:本包单向依赖 control_communication;继承例外 2 个
(lwd_edge_scheduler / lwd_cloud_phase_scheduler 均继承 AsyncScheduler,
§7.5/§9.10);内核不继承 EngineCore,触达一律经两侧装配文件中的
端口适配器(协议住 lwd_step_core,§10.3 折入)。
"""
