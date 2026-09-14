# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""通信层日志:控制面 ZMQ pub/sub、HELLO 发现与通告收发。"""

from __future__ import annotations

from vllm.v1.lwd_debug.log_base import LwdLogBase


class LwdCommLog(LwdLogBase):
    """通信层:ZMQ 通道建连、发布/订阅、HELLO 握手事件。"""

    LAYER = "comm"
