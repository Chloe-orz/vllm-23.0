# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""边云共享的配置支撑:把已解析拓扑投影成控制面端点(ROUTER/DEALER)与
is_lwd_prefill_only(模式判定唯一实现)。

YAML 只由核心配置层解析(vllm/config/lwd_topology);本文件不读文件、
不读 additional_config、不读 env。连接方向永远是"边连云":云侧每个 dp
用 ROUTER bind 自己的 ctrl_port,边侧对每条 dp 级连接开一条 DEALER
connect(带稳定 ZMQ_IDENTITY);端点全部由拓扑推导,两侧无需互通端点。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from vllm.config.lwd import LWD_WIRE_STORE_PORT_DEFAULT
from vllm.logger import init_logger
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LWD_WIRE_VERSION,
)

logger = init_logger(__name__)

LWD_HELLO_TIMEOUT_S_DEFAULT = 600.0

# dp 级连接键 (edge_id, cloud_id, dp_idx):调度器选路、消息归属、
# DEALER 表共用的最小定位单元(cloud_id 由到达端口隐含,键内显式
# 携带是为边侧多云选路可用)
LwdLinkKey = tuple[int, int, int]


@dataclass(frozen=True)
class LwdConfig:
    """多实例形态的通信投影:装配期一次成型,plain 值。"""

    role: str = "edge"
    instance_id: int = 0
    my_links: tuple[LwdLinkKey, ...] = ()
    """本实例参与的 dp 级连接(从 instance_links 按 role/id 过滤)。"""
    router_bind_endpoint: str | None = None
    """云侧 ROUTER bind 端点(tcp://*:{本dp.ctrl_port});边侧恒 None。"""
    dealer_endpoints: dict[LwdLinkKey, str] = field(default_factory=dict)
    """边侧 per-link DEALER 端点(link -> tcp://{云dp.addr}:{ctrl_port})。"""
    dealer_identity: bytes | None = None
    """边侧稳定身份 edge-{id}-{dp_idx}:断线重连后云侧按 identity 认回。"""
    wire_store_init_method: str = ""
    """边云共享世界 rendezvous(tcp://全局 rank0 机器:29600,四场景
    该位置恒为 edges[0].dp[0])。"""
    topology_digest: str = ""
    edge_npu_count: int = 0
    cloud_npu_count: int = 0
    """互校三元组(register 携带,公共校验函数消费)。"""
    hello_timeout_s: float = LWD_HELLO_TIMEOUT_S_DEFAULT

    @property
    def is_edge_node(self) -> bool:
        return self.role == "edge"

    @classmethod
    def from_vllm_config(cls, vllm_config) -> LwdConfig:
        effective = getattr(vllm_config, "lwd_config", None)
        if effective is None or not effective.enabled or effective.topology is None:
            raise ValueError("[Lwd] Transport requires a resolved lwd_config.path")
        topology = effective.topology
        topology.validate_single_dp_runtime()
        role, instance_id = effective.role, effective.instance_id
        dp_idx = 0  # 单 dp 运行门内的唯一取值;多 dp 执行随入口字段放开
        links = tuple(
            (link.edge, link.cloud, dp_idx)
            for link in topology.instance_links
            if (link.edge if role == "edge" else link.cloud) == instance_id
        )
        edge0 = topology.edges[0].dp[0]
        cloud_dps = {
            link: topology.dp("cloud", link[1], dp_idx) for link in links
        }
        config = cls(
            role=role,
            instance_id=instance_id,
            my_links=links,
            router_bind_endpoint=(
                f"tcp://*:{topology.dp('cloud', instance_id, dp_idx).ctrl_port}"
                if role == "cloud"
                else None
            ),
            dealer_endpoints={
                link: f"tcp://{dp.addr}:{dp.ctrl_port}"
                for link, dp in cloud_dps.items()
            }
            if role == "edge"
            else {},
            dealer_identity=(
                f"edge-{instance_id}-{dp_idx}".encode() if role == "edge" else None
            ),
            wire_store_init_method=f"tcp://{edge0.addr}:{LWD_WIRE_STORE_PORT_DEFAULT}",
            topology_digest=topology.digest,
            edge_npu_count=len(edge0.ranks),
            cloud_npu_count=len(topology.clouds[0].dp[0].ranks),
        )
        # 该投影被多个引擎/executor 入口调用:每进程报一次实际取值
        logger.info_once(
            "[Lwd][config][transport] role=%s instance_id=%d wire=%d digest=%s "
            "links=%s values=%s",
            role,
            instance_id,
            LWD_WIRE_VERSION,
            config.topology_digest or "-",
            [list(link) for link in config.my_links],
            json.dumps(
                {
                    "router_bind": config.router_bind_endpoint,
                    "dealers": {str(k): v for k, v in config.dealer_endpoints.items()},
                    "identity": (config.dealer_identity or b"").decode(),
                    "wire_store": config.wire_store_init_method,
                },
                ensure_ascii=False,
            ),
        )
        return config


def is_lwd_prefill_only(vllm_config) -> bool:
    """Use only the resolved config; no legacy JSON or environment fallback."""
    effective = getattr(vllm_config, "lwd_config", None)
    return bool(
        effective is not None
        and effective.enabled
        and effective.mode == "prefill_only"
    )
