# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD 诊断日志包:层日志族(基类 + control/worker/model/comm/data)+ 静态 facade。

层日志统一前缀 ``[Lwd][<layer>-<tag>]``,开关默认读
VLLM_ASCEND_LWD_DEBUG(跨进程生效),引擎进程可按 config 段经
``LwdLogBase.set_debug`` 置位。整体下线时删除本目录,并移除带
``# [lwd-debug]`` 标记的调用行与层日志调用点即可,功能逻辑零残留。
"""

from vllm.v1.lwd_debug.comm_log import LwdCommLog
from vllm.v1.lwd_debug.control_log import (
    LWD_FLIGHT_LOG_INTERVAL_S,
    LwdControlLog,
)
from vllm.v1.lwd_debug.data_log import LwdDataLog
from vllm.v1.lwd_debug.log_base import LwdLogBase
from vllm.v1.lwd_debug.lwd_debug import LwdDebug
from vllm.v1.lwd_debug.model_log import LwdModelLog
from vllm.v1.lwd_debug.worker_log import LwdWorkerLog

# 过渡别名:旧实例式 LwdLog 由 LwdControlLog 顶替,收编完调用点后删
LwdLog = LwdControlLog

__all__ = [
    "LWD_FLIGHT_LOG_INTERVAL_S",
    "LwdCommLog",
    "LwdControlLog",
    "LwdDataLog",
    "LwdDebug",
    "LwdLog",
    "LwdLogBase",
    "LwdModelLog",
    "LwdWorkerLog",
]
