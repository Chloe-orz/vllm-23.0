"""步进共享支撑(§10.3 折入):LwdStepSettings plain 值载体与 LwdLog。

step 编排已随装配消亡收进引擎子类(LwdEdgeEngineCore._lwd_edge_step,
§10.14 对齐);LwdStepCore/LwdEnginePort 抽象随 step_wrapper 模式删除,
本文件仅保留两侧引擎子类共用的纯支撑件。
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class LwdStepSettings:
    """步进编排 plain 值;装配期由 LwdConfig 派生,内核不解析配置(§10.3)。"""

    debug: bool = False
    zombie_log_interval_s: float = 30.0


class LwdLog:
    """按 debug 开关分级的极简诊断面;error 留给真正的业务中断点。"""

    def __init__(self, debug: bool = False) -> None:
        self._debug = debug

    def phase(self, message: str, *args) -> None:
        """步进/相位/准入轨迹;默认关,生产路径零输出。"""
        if self._debug:
            logger.info("[Lwd] %s", message % args if args else message)

    def degrade(self, message: str, *args) -> None:
        """可恢复降级(warning):丢预告、降级原生等。"""
        logger.warning("[Lwd] %s", message % args if args else message)
