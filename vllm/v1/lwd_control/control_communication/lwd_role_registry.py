"""多边多云角色注册表(复用旧架构 RoleRegistry 语义)。

静态角色表回答三件事:我是谁(``self_role``/``edge_id``/``cloud_id``)、
我的对端是谁(``edges``/``clouds``)、某个 ``(edge, cloud)`` 数据面通信域
的端点 rank 是什么。多实例场景全员挂载同一份 YAML,由 ``LwdConfig.
role_registry_path`` 指定;未配置时退化单边一云 ``(edge0, cloud0)`` 域,
行为与原 prefill_only 完全一致(非侵入式)。

数据面只消费 rank 映射(``edge_rank``/``cloud_rank``);ZMQ 端点路由仍由
``LwdConfig``/``lwd_edge_assemble`` 按 prefill_only 原有语义处理。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class LwdPeerInfo:
    """单个边/云实例的静态身份。"""

    peer_id: int
    addr: str
    ranks: list[int]
    zmq_base_port: int


class LwdRoleRegistry:
    """边云静态角色表。

    ``edges``/``clouds`` 以 id 为键;``pairs()`` 给出全部 ``(edge_id,
    cloud_id)`` 通信域;``edge_rank``/``cloud_rank`` 返回该通信域的两端
    端点 rank(边入口 rank / 云 TP0 rank),供数据面建组与选通道。
    """

    def __init__(self, edges: dict[int, LwdPeerInfo],
                 clouds: dict[int, LwdPeerInfo]) -> None:
        self._edges = edges
        self._clouds = clouds

    @property
    def edges(self) -> dict[int, LwdPeerInfo]:
        return self._edges

    @property
    def clouds(self) -> dict[int, LwdPeerInfo]:
        return self._clouds

    @property
    def edge_ids(self) -> list[int]:
        return sorted(self._edges)

    @property
    def cloud_ids(self) -> list[int]:
        return sorted(self._clouds)

    @property
    def is_multi_instance(self) -> bool:
        return len(self._edges) > 1 or len(self._clouds) > 1

    def edge_ranks(self, edge_id: int) -> list[int]:
        return list(self._edges[edge_id].ranks)

    def cloud_ranks(self, cloud_id: int) -> list[int]:
        return list(self._clouds[cloud_id].ranks)

    def edge_rank(self, edge_id: int) -> int:
        """边入口 rank(= 边 ranks 首卡)。"""
        return self._edges[edge_id].ranks[0]

    def cloud_rank(self, cloud_id: int) -> int:
        """云端点 rank(= 云 TP0 首卡)。"""
        return self._clouds[cloud_id].ranks[0]

    def cloud_endpoint_rank(self, edge_id: int, cloud_id: int) -> int:
        """云端点 rank:随 edge_id 在该云各 rank 间轮转。

        edge 0 -> ranks[0], edge 1 -> ranks[1], ... 取模回到 ranks[0],
        把不同边的跨机 P2P 分摊到云内不同卡,避免全部压在 TP0 首卡。
        单卡云(单边一云退化域) len(ranks)==1 恒取 ranks[0],与原行为一致。
        """
        ranks = self._clouds[cloud_id].ranks
        return ranks[edge_id % len(ranks)]

    def pairs(self) -> list[tuple[int, int]]:
        return [(e, c) for e in self.edge_ids for c in self.cloud_ids]

    def self_ids(self, my_rank: int) -> tuple[str | None, int, int]:
        """由全局 rank 判定本进程身份:(role, edge_id, cloud_id)。"""
        for edge_id, peer in self._edges.items():
            if my_rank in peer.ranks:
                return "edge", edge_id, 0
        for cloud_id, peer in self._clouds.items():
            if my_rank in peer.ranks:
                return "cloud", 0, cloud_id
        return None, 0, 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LwdRoleRegistry":
        edges = {
            int(e["id"]): LwdPeerInfo(
                peer_id=int(e["id"]),
                addr=str(e.get("addr", "")),
                ranks=[int(r) for r in e.get("ranks", [])],
                zmq_base_port=int(e.get("zmq_base_port", 0)),
            )
            for e in raw.get("edges", [])
        }
        clouds = {
            int(c["id"]): LwdPeerInfo(
                peer_id=int(c["id"]),
                addr=str(c.get("addr", "")),
                ranks=[int(r) for r in c.get("ranks", [])],
                zmq_base_port=int(c.get("zmq_base_port", 0)),
            )
            for c in raw.get("clouds", [])
        }
        if not edges or not clouds:
            raise ValueError("role registry requires at least one edge and one cloud")
        return cls(edges, clouds)

    @classmethod
    def single_pair(cls, edge_rank: int, cloud_rank: int) -> "LwdRoleRegistry":
        """单边一云退化域(未配置 role_registry 时)。"""
        return cls(
            {0: LwdPeerInfo(0, "", [edge_rank], 0)},
            {0: LwdPeerInfo(0, "", [cloud_rank], 0)},
        )


def load_role_registry(role_registry_path: str,
                       fallback: LwdRoleRegistry | None = None
                       ) -> LwdRoleRegistry:
    """按路径加载角色表;路径为空/加载失败回退 fallback。"""
    if not role_registry_path:
        if fallback is None:
            raise ValueError("role_registry_path is empty and no fallback registry")
        return fallback
    try:
        with open(role_registry_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return LwdRoleRegistry.from_dict(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[lwd] failed to load role registry %s: %s",
                       role_registry_path, exc)
        if fallback is not None:
            return fallback
        raise
