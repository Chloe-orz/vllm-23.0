# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Project the resolved topology onto the existing no-PP control transport.

YAML is read only by the core config parser. This adapter never reads files,
raw additional_config or environment variables. PUSH/PULL remains unchanged:
cloud addr/ctrl_port is PRE_OUT, edge addr is POST_OUT and TCPStore host.
POST_OUT port is the temporary environment snapshot captured by core config.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.config.lwd import LWD_POST_OUT_PORT_DEFAULT, LWD_WIRE_STORE_PORT_DEFAULT
from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
    LWD_PUBLISH_QUEUE_MAX,
)

LWD_PRE_OUT_PORT_DEFAULT = 5558
LWD_HELLO_TIMEOUT_S_DEFAULT = 600.0


@dataclass(frozen=True)
class LwdConfig:
    """Plain transport values; no device or connection state."""

    is_edge_node: bool = True
    pre_out_host: str = "127.0.0.1"
    pre_out_port: int = LWD_PRE_OUT_PORT_DEFAULT
    post_out_host: str = ""
    post_out_port: int = LWD_POST_OUT_PORT_DEFAULT
    post_out_bind: str = "*"
    wire_store_port: int = LWD_WIRE_STORE_PORT_DEFAULT
    hello_timeout_s: float = LWD_HELLO_TIMEOUT_S_DEFAULT
    scheduler_name: str = "prefill_first"
    publish_queue_max: int = LWD_PUBLISH_QUEUE_MAX
    debug: bool = False

    def lwd_pre_out_endpoint(self) -> str:
        return f"tcp://{self.pre_out_host}:{self.pre_out_port}"

    def lwd_post_out_bind_endpoint(self) -> str:
        return f"tcp://{self.post_out_bind}:{self.post_out_port}"

    def lwd_post_out_connect_endpoint(self) -> str:
        return f"tcp://{self.post_out_host}:{self.post_out_port}"

    def lwd_wire_store_init_method(self) -> str:
        return f"tcp://{self.post_out_host}:{self.wire_store_port}"

    @classmethod
    def from_vllm_config(cls, vllm_config) -> LwdConfig:
        effective = getattr(vllm_config, "lwd_config", None)
        if effective is None or not effective.enabled or effective.topology is None:
            raise ValueError("[LWD] Transport requires a resolved lwd_config.path")
        topology = effective.topology
        topology.validate_single_dp_runtime()
        edge = topology.dp("edge", 0, 0)
        cloud = topology.dp("cloud", 0, 0)
        assert cloud.ctrl_port is not None
        return cls(
            is_edge_node=effective.is_edge,
            pre_out_host=cloud.addr,
            pre_out_port=cloud.ctrl_port,
            post_out_host=edge.addr,
            post_out_port=effective.post_out_port,
        )


def is_lwd_prefill_only(vllm_config) -> bool:
    """Use only the resolved config; no legacy JSON or environment fallback."""
    effective = getattr(vllm_config, "lwd_config", None)
    return bool(
        effective is not None
        and effective.enabled
        and effective.mode == "prefill_only"
    )
