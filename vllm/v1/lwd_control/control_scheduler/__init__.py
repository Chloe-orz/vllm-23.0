"""控制面公共层:边/云调度器与引擎的基类。

依赖方向:本包单向依赖 control_communication。继承关系 4 条:
lwd_edge/cloud_scheduler 继承 LwdBaseScheduler(AsyncScheduler 子类),
lwd_edge/cloud_engine 继承 LwdBaseEngineCore(EngineCoreProc 子类)。
"""
