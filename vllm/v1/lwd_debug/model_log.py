# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""model 层日志:model runner 的输入注入/采样恢复轨迹。"""

from __future__ import annotations

from vllm.v1.lwd_debug.log_base import LwdLogBase


class LwdModelLog(LwdLogBase):
    """model 层:embeds 注入窗口、fill loop、token 恢复等模型侧事件。"""

    LAYER = "model"
