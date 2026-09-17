# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD 层日志基类:层前缀 + debug 门控,子类只定 LAYER 与层特有方法。

开关默认读 VLLM_ASCEND_LWD_DEBUG(与 LwdConfig 同一环境变量,worker/
model 等独立进程靠环境变量继承开关);引擎进程可再按 config 段经
set_debug 置位。调用侧零装配(全类方法),生产路径零输出。
"""

from __future__ import annotations

import time

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# 限频放行时刻表:(类名, key) -> 上次放行的单调钟时刻
_RATE_LAST: dict[tuple[str, str], float] = {}


class LwdLogBase:
    """按层打 ``[Lwd][<layer>-<tag>]`` 前缀日志的静态基类。"""

    LAYER = ""

    DEBUG = envs.VLLM_ASCEND_LWD_DEBUG

    @classmethod
    def set_debug(cls, enabled: bool) -> None:
        """引擎启动时按 config 段置位(只置开不置关,仅影响本进程)。"""
        if enabled:
            cls.DEBUG = True

    @classmethod
    def event(cls, tag: str, msg: str, *args) -> None:
        """站点事件:[Lwd][<layer>-<tag>] msg%args;默认关。"""
        if not cls.DEBUG:
            return
        logger.info(
            "[Lwd][%s-%s] %s", cls.LAYER, tag, msg % args if args else msg
        )

    @classmethod
    def phase(cls, message: str, *args) -> None:
        """步进/相位轨迹:[Lwd][<layer>] message;默认关。"""
        if not cls.DEBUG:
            return
        logger.info("[Lwd][%s] %s", cls.LAYER, message % args if args else message)

    @classmethod
    def _rate_pass(cls, key: str, interval_s: float) -> bool:
        """限频助手:(类, key) 级隔 interval_s 秒放行一次。"""
        now = time.monotonic()
        k = (cls.__name__, key)
        if now - _RATE_LAST.get(k, 0.0) < interval_s:
            return False
        _RATE_LAST[k] = now
        return True
