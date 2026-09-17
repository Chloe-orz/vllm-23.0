"""云侧复用(cloud reuse,多边多云)角色注册表。

静态角色表回答三件事:我是谁(``validate_self``/``edge_id``/``cloud_id``)、
我的对端是谁(``edges``/``clouds``)、某个 ``(edge, cloud)`` 数据面通信域
的端点 rank 是什么。云侧复用场景全员挂载同一份 YAML,由
``--role-registry`` 指定;未配置时退化单边一云 ``(edge0, cloud0)`` 域,
行为与原 prefill_only 完全一致(非侵入式)。

控制面端点规划(ROUTER-ROUTER):只有云侧 bind,每云单值 ``zmq_port``
服务全部边;边侧 0 端口 connect 出去。端点公式唯一事实源在
``endpoint``/``bind_endpoint``;数据面只消费 rank 映射
(``edge_rank``/``cloud_rank``)。

registry YAML 示例(2 边 2 云,全场共享同一份)::

    world: { master_addr: 10.1.0.1, master_port: 29500 }
    edges:
      - { id: 0, addr: 10.0.0.1, ranks: [0, 1] }           # 不 bind,无端口字段
      - { id: 1, addr: 10.0.0.2, ranks: [2, 3] }           # addr 仅诊断用,可省
    clouds:
      - { id: 0, addr: 10.1.0.1, ranks: [4, 5, 6, 7], zmq_port: 5700 }
      - { id: 1, addr: 10.1.0.2, ranks: [8, 9, 10, 11], zmq_port: 5700 }
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import yaml
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class LwdPeerInfo:
    """单个边/云实例的静态身份。

    ``ranks`` 为该实例的全局 rank 列表(HCCL 组按全局 rank 建);
    ``zmq_port`` 仅云侧有值(边不 bind);``addr`` 边侧仅诊断用。
    """

    peer_id: int
    addr: str
    ranks: list[int]
    zmq_port: int = 0


class LwdRoleRegistry:
    """边云静态角色表。

    ``edges``/``clouds`` 以 id 为键;``pairs()`` 给出全部 ``(edge_id,
    cloud_id)`` 通信域;``edge_rank``/``cloud_rank`` 返回该通信域的两端
    端点 rank(边入口 rank / 云 TP0 rank),供数据面建组与选通道;
    ``config_digest`` 供全场配置一致性校验。
    """

    def __init__(self, edges: dict[int, LwdPeerInfo],
                 clouds: dict[int, LwdPeerInfo],
                 config_digest: str = "") -> None:
        self._edges = edges
        self._clouds = clouds
        self._config_digest = config_digest

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
    def config_digest(self) -> str:
        """全场配置摘要;全员必须算出同一值(启动一致性校验)。"""
        return self._config_digest

    @property
    def is_cloud_reuse(self) -> bool:
        """云侧复用拓扑(多于一边或多于一云)为 True。"""
        return len(self._edges) > 1 or len(self._clouds) > 1

    # ------------------------------------------------------------------ #
    # 身份 / 在场校验                                                     #
    # ------------------------------------------------------------------ #
    def validate_self(self, role: str, instance_id: int) -> None:
        """校验本进程声明的 (role, id) 在表中在场;缺位 fail-fast。"""
        table = self._edges if role == "edge" else self._clouds
        if instance_id not in table:
            raise ValueError(
                f"[lwd] role={role} id={instance_id} not found in registry; "
                f"known ids={sorted(table)}"
            )

    def assert_rank_membership(self, role: str, instance_id: int,
                               global_rank: int) -> None:
        """校验实际全局 rank 与 registry 声明一致(防 HCCL 组错进程)。"""
        table = self._edges if role == "edge" else self._clouds
        expected = table[instance_id].ranks
        if global_rank not in expected:
            raise ValueError(
                f"[lwd] rank/registry mismatch: role={role} id={instance_id} "
                f"process has global_rank={global_rank}, but registry says "
                f"ranks={expected}"
            )

    def self_ids(self, my_rank: int) -> tuple[str | None, int, int]:
        """由全局 rank 判定本进程身份:(role, edge_id, cloud_id)。"""
        for edge_id, peer in self._edges.items():
            if my_rank in peer.ranks:
                return "edge", edge_id, 0
        for cloud_id, peer in self._clouds.items():
            if my_rank in peer.ranks:
                return "cloud", 0, cloud_id
        return None, 0, 0

    # ------------------------------------------------------------------ #
    # 数据面端点(rank 映射)                                              #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 控制面端点(唯一事实源)                                             #
    # ------------------------------------------------------------------ #
    def endpoint(self, cloud_id: int) -> str:
        """云控制面端点(边侧 connect 用):``tcp://{cloud.addr}:{zmq_port}``。"""
        cloud = self._clouds[cloud_id]
        return f"tcp://{cloud.addr}:{cloud.zmq_port}"

    def bind_endpoint(self, cloud_id: int) -> str:
        """云控制面 bind 端点:addr 换为 ``*``,每云单端口服务全部边。"""
        return f"tcp://*:{self._clouds[cloud_id].zmq_port}"

    # ------------------------------------------------------------------ #
    # 构造                                                               #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LwdRoleRegistry":
        edges = {
            int(e["id"]): LwdPeerInfo(
                peer_id=int(e["id"]),
                addr=str(e.get("addr", "")),
                ranks=[int(r) for r in e.get("ranks", [])],
            )
            for e in raw.get("edges", [])
        }
        clouds = {
            int(c["id"]): LwdPeerInfo(
                peer_id=int(c["id"]),
                addr=str(c.get("addr", "")),
                ranks=[int(r) for r in c.get("ranks", [])],
                # ROUTER-ROUTER:每云单端口;兼容旧 zmq_base_port 字段名
                zmq_port=int(c.get("zmq_port", c.get("zmq_base_port", 0))),
            )
            for c in raw.get("clouds", [])
        }
        if not edges or not clouds:
            raise ValueError(
                "[lwd] role registry requires at least one edge and one cloud"
            )
        for cloud_id, cloud in clouds.items():
            if cloud.zmq_port <= 0:
                raise ValueError(
                    f"[lwd] cloud {cloud_id} requires a positive zmq_port "
                    f"in the role registry (ROUTER bind)"
                )
        digest = hashlib.sha256(
            json.dumps(raw, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        return cls(edges, clouds, digest)

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


# ---------------------------------------------------------------------- #
# 进程级单例(控制面装配期加载,运行期只读)                                #
# ---------------------------------------------------------------------- #
_REGISTRY: LwdRoleRegistry | None = None


def init_role_registry(path: str) -> LwdRoleRegistry:
    """进程级单例:首次加载并缓存 registry,重复调用返回同一份。"""
    global _REGISTRY
    if _REGISTRY is None:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        _REGISTRY = LwdRoleRegistry.from_dict(raw)
        logger.info(
            "[lwd] role registry loaded from %s: edges=%s clouds=%s digest=%s",
            path, _REGISTRY.edge_ids, _REGISTRY.cloud_ids,
            _REGISTRY.config_digest,
        )
    return _REGISTRY


def get_role_registry() -> LwdRoleRegistry | None:
    """取已加载 registry;未启用云侧复用时为 None。"""
    return _REGISTRY
