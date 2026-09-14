# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""worker 层日志:边/云 worker 的批提交与执行轨迹。"""

from __future__ import annotations

from vllm.v1.lwd_debug.log_base import LwdLogBase


class LwdWorkerLog(LwdLogBase):
    """worker 层:EMBED/UNEMBED 批派发、worker 生命周期事件。"""

    LAYER = "worker"
