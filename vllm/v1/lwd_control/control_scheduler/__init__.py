"""控制面调度/执行层:边/云调度器与配置支撑。

依赖方向:本包单向依赖 control_communication;继承例外 2 个
(lwd_edge_scheduler / lwd_cloud_phase_scheduler 均继承
AsyncScheduler);内核不继承 EngineCore,引擎触达收在两侧引擎
子类(lwd_edge_engine / lwd_cloud_engine)。
"""
