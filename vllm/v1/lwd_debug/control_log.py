# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""控制面层日志:边/云调度器、引擎编排(含 ZMQ 通信的编排侧事件)。"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.lwd_debug.log_base import LwdLogBase

logger = init_logger(__name__)

# flight 水位日志限频间隔(秒)
LWD_FLIGHT_LOG_INTERVAL_S = 5.0


class LwdControlLog(LwdLogBase):
    """控制面层:调度决策/准入/通告/生命周期对账事件。"""

    LAYER = "control"

    @classmethod
    def flight(cls, running: int, awaiting: int) -> None:
        """在飞请求水位(running + awaiting);常开,限频隔几秒一条。

        常开不走 debug 门:一条水位线是调度观测的基本面,限频后
        日志量可忽略。接口隔离:只收数值,不感知调度器结构;限频
        状态内聚于基类助手。"""
        if not cls._rate_pass("flight", LWD_FLIGHT_LOG_INTERVAL_S):
            return
        logger.info(
            "[Lwd][control-flight] running=%d awaiting=%d cloud_active=%d",
            running, awaiting, running + awaiting,
        )
