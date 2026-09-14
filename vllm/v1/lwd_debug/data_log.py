# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""数据面层日志:HCCL 通道(UP 广播 / DOWN isend-irecv)张量收发。"""

from __future__ import annotations

from vllm.v1.lwd_debug.log_base import LwdLogBase


class LwdDataLog(LwdLogBase):
    """数据面层:UP/DOWN 张量通道的发送、接收与配对事件。"""

    LAYER = "data"
