# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""File schema for LWD, independent of devices and distributed initialization."""

from dataclasses import dataclass
from hashlib import sha256
from ipaddress import ip_address
from pathlib import Path
from typing import Any

import yaml


def _mapping(
    value: Any, field: str, required: set[str], optional: tuple[str, ...] = ()
) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"[LWD] {field} must be a mapping")
    missing = required - value.keys()
    unknown = value.keys() - required - set(optional)
    for old, new in (("sence", "scene"), ("enable_eary_recv", "enable_early_recv")):
        if old in unknown:
            raise ValueError(f"[LWD] {field}.{old} is invalid; use {new}")
    if missing or unknown:
        raise ValueError(
            f"[LWD] {field}: missing fields={sorted(missing)}, "
            f"unknown fields={sorted(map(str, unknown))}"
        )
    return value


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"[LWD] {field} must be an integer >= {minimum}")
    return value


def _list(value: Any, field: str) -> list:
    if not isinstance(value, list) or not value:
        raise ValueError(f"[LWD] {field} must be a non-empty list")
    return value


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys instead of silently replacing values."""

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("[LWD] YAML mapping keys must be strings")
            if key in result:
                raise ValueError(f"[LWD] duplicate YAML key: {key!r}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


@dataclass(frozen=True)
class LwdDeployment:
    mode: int
    scene: str
    hccl_world_size: int
    edges_num: int
    clouds_num: int


@dataclass(frozen=True)
class LwdFeatures:
    enable_early_recv: bool = False
    enable_scramble: bool = False


@dataclass(frozen=True)
class LwdDP:
    dp_idx: int
    addr: str
    ranks: tuple[int, ...]
    ctrl_port: int | None = None


@dataclass(frozen=True)
class LwdInstance:
    id: int
    dp: tuple[LwdDP, ...]


@dataclass(frozen=True)
class LwdLink:
    edge: int
    cloud: int


@dataclass(frozen=True)
class LwdTopology:
    deployment: LwdDeployment
    feature_ctrl: LwdFeatures
    edges: tuple[LwdInstance, ...]
    clouds: tuple[LwdInstance, ...]
    instance_links: tuple[LwdLink, ...]
    digest: str = ""
    """sha256(YAML 原文) 前 16 hex,register 互校用;from_dict 构造
    (测试)无原文,恒空串 = 跳过比对。"""

    @classmethod
    def from_file(cls, path: str) -> "LwdTopology":
        try:
            file_bytes = Path(path).read_bytes()
            with Path(path).open(encoding="utf-8") as stream:
                raw = yaml.load(stream, Loader=_UniqueKeyLoader)
            topology = cls.from_dict(raw)
            return LwdTopology(
                deployment=topology.deployment,
                feature_ctrl=topology.feature_ctrl,
                edges=topology.edges,
                clouds=topology.clouds,
                instance_links=topology.instance_links,
                digest=sha256(file_bytes).hexdigest()[:16],
            )
        except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
            raise ValueError(f"[LWD] Invalid topology file {path!r}: {exc}") from exc

    @classmethod
    def from_dict(cls, raw: Any) -> "LwdTopology":
        raw = _mapping(
            raw,
            "topology",
            {"deployment", "feature_ctrl", "edges", "clouds", "instance_links"},
        )
        deploy = _mapping(
            raw["deployment"],
            "deployment",
            {"mode", "scene", "hccl_world_size", "edges_num", "clouds_num"},
        )
        mode = _integer(deploy["mode"], "deployment.mode")
        if mode != 0:
            raise ValueError(
                "[LWD] deployment.mode: only 0 (prefill_only) is supported"
            )
        scene = deploy["scene"]
        if scene not in ("single_instance", "edge_share", "cloud_share", "lwd_cluster"):
            raise ValueError("[LWD] deployment.scene is not a recognized scene")
        deployment = LwdDeployment(
            mode=mode,
            scene=scene,
            hccl_world_size=_integer(
                deploy["hccl_world_size"], "deployment.hccl_world_size", 1
            ),
            edges_num=_integer(deploy["edges_num"], "deployment.edges_num", 1),
            clouds_num=_integer(deploy["clouds_num"], "deployment.clouds_num", 1),
        )
        features = _mapping(
            raw["feature_ctrl"],
            "feature_ctrl",
            set(),
            ("enable_early_recv", "enable_scramble"),
        )
        for name, value in features.items():
            if type(value) is not bool:
                raise ValueError(f"[LWD] feature_ctrl.{name} must be a boolean")
        edges = _instances(raw["edges"], "edges")
        clouds = _instances(raw["clouds"], "clouds")
        if deployment.edges_num != len(edges) or deployment.clouds_num != len(clouds):
            raise ValueError(
                "[LWD] deployment instance counts do not match edges/clouds"
            )
        links = []
        for index, value in enumerate(_list(raw["instance_links"], "instance_links")):
            field = f"instance_links[{index}]"
            value = _mapping(value, field, {"edge", "cloud"})
            link = LwdLink(
                _integer(value["edge"], f"{field}.edge"),
                _integer(value["cloud"], f"{field}.cloud"),
            )
            if link in links:
                raise ValueError(f"[LWD] {field}: duplicate link")
            if link.edge >= len(edges) or link.cloud >= len(clouds):
                raise ValueError(f"[LWD] {field}: unknown edge/cloud instance")
            links.append(link)
        topology = cls(deployment, LwdFeatures(**features), edges, clouds, tuple(links))
        topology.validate_layout()
        return topology

    def validate_layout(self) -> None:
        """Validate config structure, including the future multi-DP layout.

        Shared edge ranks count once in the world. Cloud DPs are disjoint.
        This does not create workers, connections or communication groups.
        """
        rank_addresses: dict[int, str] = {}
        edge_ranks: set[int] = set()
        cloud_ranks: set[int] = set()
        endpoints: set[tuple[str, int | None]] = set()
        for role, instances in (("edges", self.edges), ("clouds", self.clouds)):
            for instance in instances:
                for dp in instance.dp:
                    if len(set(dp.ranks)) != len(dp.ranks):
                        raise ValueError(
                            f"[LWD] {role}[{instance.id}].dp[{dp.dp_idx}]: "
                            "duplicate ranks within one DP"
                        )
                    for rank in dp.ranks:
                        if rank in rank_addresses and rank_addresses[rank] != dp.addr:
                            raise ValueError(
                                f"[LWD] rank {rank} is assigned to different addresses"
                            )
                        rank_addresses[rank] = dp.addr
                        if role == "clouds" and rank in cloud_ranks:
                            raise ValueError("[LWD] Cloud DP ranks must not overlap")
                    if role == "edges":
                        edge_ranks.update(dp.ranks)
                    else:
                        cloud_ranks.update(dp.ranks)
                        endpoint = (dp.addr, dp.ctrl_port)
                        if endpoint in endpoints:
                            raise ValueError(
                                "[LWD] Cloud DPs on the same address must use "
                                "different ctrl_port values"
                            )
                        endpoints.add(endpoint)
        if edge_ranks & cloud_ranks:
            raise ValueError("[LWD] Edge and cloud ranks must not overlap")
        world = self.deployment.hccl_world_size
        if len(rank_addresses) != world or sorted(rank_addresses) != list(range(world)):
            raise ValueError(
                "[LWD] Unique ranks must cover [0, hccl_world_size) without gaps"
            )
        if edge_ranks != set(range(len(edge_ranks))):
            raise ValueError("[LWD] Global ranks must place all edges before clouds")
        for link in self.instance_links:
            edge = self.instance("edge", link.edge)
            cloud = self.instance("cloud", link.cloud)
            if len(edge.dp) != len(cloud.dp):
                raise ValueError(
                    f"[LWD] Linked edge={link.edge}, cloud={link.cloud} "
                    "must have matching DP indices"
                )

    def validate_single_dp_runtime(self) -> None:
        """Gate execution separately from parsing the future multi-DP schema."""
        if len(self.edges) != 1 or len(self.clouds) != 1:
            raise ValueError(
                "[LWD] Only one edge and one cloud instance are supported yet"
            )
        if len(self.edges[0].dp) != 1 or len(self.clouds[0].dp) != 1:
            raise ValueError(
                "[LWD] Multi-DP topology was parsed, but the current no-PP "
                "runtime supports single DP only; multi-DP execution is not "
                "implemented. Use the single-DP YAML and --data-parallel-size 1."
            )
        if self.instance_links != (LwdLink(0, 0),):
            raise ValueError(
                "[LWD] Single-instance topology requires link edge=0, cloud=0"
            )
        edge, cloud = self.edges[0].dp[0], self.clouds[0].dp[0]
        if any(ip_address(dp.addr).version != 4 for dp in (edge, cloud)):
            raise ValueError(
                "[LWD] The current PUSH/PULL runtime requires IPv4 addresses"
            )
        if edge.addr == cloud.addr:
            raise ValueError("[LWD] Edge and cloud must have different addresses")
        # The existing executor assigns contiguous edge-first global ranks.
        # Check order as well as membership; ranks[0] identifies the endpoint.
        edge_count = len(edge.ranks)
        world = edge_count + len(cloud.ranks)
        if self.deployment.hccl_world_size != world:
            raise ValueError(
                "[LWD] hccl_world_size does not match the edge/cloud ranks"
            )
        if edge.ranks != tuple(range(edge_count)) or cloud.ranks != tuple(
            range(edge_count, world)
        ):
            raise ValueError(
                "[LWD] ranks must cover [0, hccl_world_size) in contiguous "
                "edge-first order without duplicates or gaps"
            )

    def instance(self, role: str, instance_id: int) -> LwdInstance:
        if role not in ("edge", "cloud"):
            raise ValueError("[LWD] role must be 'edge' or 'cloud'")
        instances = self.edges if role == "edge" else self.clouds
        for instance in instances:
            if instance.id == instance_id:
                return instance
        raise ValueError(
            f"[LWD] Instance not found: role={role}, instance_id={instance_id}"
        )

    def dp(self, role: str, instance_id: int, dp_idx: int) -> LwdDP:
        """Resolve an explicit DP; never silently select DP 0 for multi-DP."""
        for dp in self.instance(role, instance_id).dp:
            if dp.dp_idx == dp_idx:
                return dp
        raise ValueError(
            f"[LWD] DP not found: role={role}, instance_id={instance_id}, "
            f"dp_idx={dp_idx}"
        )


def _instances(raw: Any, role: str) -> tuple[LwdInstance, ...]:
    result = []
    for index, value in enumerate(_list(raw, role)):
        field = f"{role}[{index}]"
        value = _mapping(value, field, {"id", "dp"})
        instance_id = _integer(value["id"], f"{field}.id")
        dps = []
        for dp_index, dp in enumerate(_list(value["dp"], f"{field}.dp")):
            dp_field = f"{field}.dp[{dp_index}]"
            keys = {"dp_idx", "addr", "ranks"}
            if role == "clouds":
                keys.add("ctrl_port")
            dp = _mapping(dp, dp_field, keys)
            dp_idx = _integer(dp["dp_idx"], f"{dp_field}.dp_idx")
            addr = dp["addr"]
            if not isinstance(addr, str):
                raise ValueError(f"[LWD] {dp_field}.addr must be an IP address string")
            try:
                ip = ip_address(addr)
            except ValueError as exc:
                raise ValueError(f"[LWD] {dp_field}.addr: invalid IP {addr!r}") from exc
            if ip.is_unspecified or ip.is_multicast:
                raise ValueError(f"[LWD] {dp_field}.addr must be a unicast endpoint IP")
            ranks = tuple(
                _integer(rank, f"{dp_field}.ranks[{i}]")
                for i, rank in enumerate(_list(dp["ranks"], f"{dp_field}.ranks"))
            )
            port = None
            if role == "clouds":
                port = _integer(dp["ctrl_port"], f"{dp_field}.ctrl_port", 1)
                if port > 65535:
                    raise ValueError(f"[LWD] {dp_field}.ctrl_port must be <= 65535")
            dps.append(LwdDP(dp_idx, str(ip), ranks, port))
        dps.sort(key=lambda dp: dp.dp_idx)
        if [dp.dp_idx for dp in dps] != list(range(len(dps))):
            raise ValueError(
                f"[LWD] {field}.dp_idx must be unique and contiguous from 0"
            )
        result.append(LwdInstance(instance_id, tuple(dps)))
    result.sort(key=lambda instance: instance.id)
    if [instance.id for instance in result] != list(range(len(result))):
        raise ValueError(f"[LWD] {role}.id must be unique and contiguous from 0")
    return tuple(result)
