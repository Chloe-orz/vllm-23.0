# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LWD (layerwise disaggregated) configuration.

LWD splits a model's transformer layers across an edge / cloud pair:
the edge process holds the head ``[0, head_k)`` and tail
``[N - tail_k, N)`` ranges (or none in ``embedding_only`` mode), while
the cloud process holds the middle range ``[head_k, N - tail_k)``.
``make_layers`` consumes :meth:`LwdConfig.local_layer_indices` so
weights are created directly on the owning device with
``PPMissingLayer`` placeholders elsewhere.

The config is owned by the vLLM repository: ``additional_config``
key ``lwd_config`` is parsed once in ``VllmConfig.__post_init__``;
The public entry contains path, role and instance_id. A valid path enables LWD;
enabled/mode/layer fields below are internal compatibility values, not CLI inputs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .lwd_topology import LwdDP, LwdInstance, LwdTopology

LWD_WIRE_STORE_PORT_DEFAULT = 29600


def lwd_entry_from_additional(additional: Any) -> dict[str, Any] | None:
    """Identify and validate the CLI entry without opening the YAML file."""
    if not isinstance(additional, dict):
        return None
    if "edge_cloud_config" in additional:
        raise ValueError("[LWD] edge_cloud_config is retired; use lwd_config.path")
    if "lwd_config" not in additional:
        return None
    raw = additional["lwd_config"]
    validate_lwd_entry(raw)
    return raw


def validate_lwd_entry(raw: Any) -> None:
    if not isinstance(raw, dict):
        raise ValueError("[LWD] lwd_config must contain path, role and instance_id")
    if not isinstance(raw.get("path"), str) or not raw["path"].strip():
        raise ValueError("[LWD] lwd_config.path must be a non-empty string")
    unknown = raw.keys() - {"path", "role", "instance_id"}
    if unknown:
        raise ValueError(
            f"[LWD] Unknown lwd_config fields: {sorted(map(str, unknown))}; "
            "use only path, role and instance_id"
        )
    if raw.get("role") not in ("edge", "cloud"):
        raise ValueError("[LWD] lwd_config.role must be 'edge' or 'cloud'")
    if type(raw.get("instance_id")) is not int or raw["instance_id"] < 0:
        raise ValueError("[LWD] lwd_config.instance_id must be a non-negative integer")


_VALID_LWD_ROLES = ("edge", "cloud")
_VALID_LWD_MODES = ("head_tail", "embedding_only", "prefill_only")


@dataclass(frozen=True)
class LwdConfig:
    enabled: bool = False
    role: str = "edge"
    """Process role: "edge" or "cloud"."""
    mode: str = "head_tail"
    """Layer distribution mode: "head_tail", "embedding_only" or "prefill_only"."""
    edge_head_tail_layers: tuple[int, int] = (1, 1)
    """Fixed 2-element (head_k, tail_k) asymmetric splits allowed."""
    path: str | None = None
    instance_id: int = 0
    topology: LwdTopology | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "LwdConfig":
        if raw is None:
            return cls()
        validate_lwd_entry(raw)
        retired_port_env = "VLLM_ASCEND_LWD_POST_OUT_PORT"
        if retired_port_env in os.environ:
            # ROUTER/DEALER 控制面边侧不 bind、无端口;旧脚本带毒即拦
            raise ValueError(
                f"[LWD] {retired_port_env} is retired: the edge side no "
                "longer binds any control-plane port (cloud ROUTER only)"
            )
        topology = LwdTopology.from_file(raw["path"])
        topology.instance(raw["role"], raw["instance_id"])
        return cls(
            enabled=True,
            role=raw["role"],
            mode="prefill_only",
            edge_head_tail_layers=(0, 0),
            path=raw["path"],
            instance_id=raw["instance_id"],
            topology=topology,
        )

    @property
    def instance(self) -> LwdInstance:
        if self.topology is None:
            raise ValueError("[LWD] No topology is configured")
        return self.topology.instance(self.role, self.instance_id)

    @property
    def dp(self) -> LwdDP:
        if len(self.instance.dp) != 1:
            raise ValueError("[LWD] Multiple DPs configured; select a DP explicitly")
        return self.instance.dp[0]

    def apply_to_parallel_config(self, parallel: Any) -> None:
        """Project topology onto the no-PP backend; never overwrite TP."""
        if not self.enabled:
            return
        assert self.topology is not None
        self.topology.validate_single_dp_runtime()
        expected_tp = len(self.dp.ranks)
        if parallel.tensor_parallel_size != expected_tp:
            raise ValueError(
                f"[LWD] TP mismatch in {self.path!r}: role={self.role}, "
                f"instance_id={self.instance_id}, dp_idx={self.dp.dp_idx}, "
                f"CLI tensor_parallel_size={parallel.tensor_parallel_size}, "
                f"YAML ranks count={expected_tp}. Check the CLI or YAML configuration."
            )
        if parallel.data_parallel_size != 1:
            raise ValueError(
                "[LWD] This implementation supports data_parallel_size=1 only"
            )
        if (
            parallel.data_parallel_rank != 0
            or parallel.data_parallel_external_lb
            or parallel.data_parallel_hybrid_lb
            or parallel.data_parallel_size_local not in (None, 1)
        ):
            raise ValueError("[LWD] Single-DP execution does not support DP routing/LB")
        if (
            parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            raise ValueError(
                "[LWD] Context parallelism is not supported in this layout"
            )
        if parallel.pipeline_parallel_size != 1:
            raise ValueError(
                "[LWD] The no-PP backend requires pipeline_parallel_size=1"
            )
        if parallel.nnodes not in (1, 2):
            raise ValueError("[LWD] Single-instance execution requires two nodes")
        if parallel.data_parallel_backend != "mp":
            raise ValueError("[LWD] Single-instance execution requires DP backend mp")
        if parallel.distributed_executor_backend not in (None, "mp", "uni"):
            raise ValueError("[LWD] Single-instance execution requires the mp executor")
        edge = self.topology.edges[0].dp[0]
        cloud = self.topology.clouds[0].dp[0]
        if parallel.master_addr not in ("127.0.0.1", edge.addr):
            raise ValueError(
                f"[LWD] master_addr={parallel.master_addr!r} must match "
                f"the YAML edge address {edge.addr!r}"
            )
        parallel.lwd_config.enable_lwd = True
        parallel.lwd_config.is_edge_node = self.is_edge
        parallel.lwd_config.edge_npu_count = len(edge.ranks)
        parallel.lwd_config.cloud_npu_count = len(cloud.ranks)
        parallel.world_size = self.topology.deployment.hccl_world_size
        parallel.pipeline_parallel_size = 1
        parallel.nnodes = 2
        parallel.node_rank = 0 if self.is_edge else 1
        parallel.master_addr = edge.addr
        parallel.disable_custom_all_reduce = True
        # ParallelConfig may have auto-selected uni before loading the file.
        parallel.distributed_executor_backend = "mp"

    @property
    def is_edge(self) -> bool:
        return self.role == "edge"

    def validate(self) -> None:
        if self.role not in _VALID_LWD_ROLES:
            raise ValueError(
                f"[LWD] role must be one of {_VALID_LWD_ROLES}, got {self.role!r}"
            )
        if self.mode not in _VALID_LWD_MODES:
            raise ValueError(
                f"[LWD] mode must be one of {_VALID_LWD_MODES}, got {self.mode!r}"
            )
        head_k, tail_k = self.edge_head_tail_layers
        if self.mode in ("embedding_only", "prefill_only"):
            if (head_k, tail_k) != (0, 0):
                raise ValueError(
                    f"[LWD] {self.mode} requires edge_head_tail_layers [0, 0], "
                    f"got [{head_k},{tail_k}]"
                )
        elif head_k + tail_k < 1:
            raise ValueError(
                "[LWD] 'head_tail' mode requires at least one edge layer, "
                f"got [{head_k}, {tail_k}]"
            )

    def validate_num_hidden_layers(self, num_hidden_layers: int) -> None:
        head_k, tail_k = self.edge_head_tail_layers
        if head_k + tail_k >= num_hidden_layers:
            raise ValueError(
                "[LWD] layer split must leave a non-empty middle range for "
                f"the cloud: head_k + tail_k ({head_k + tail_k}) must be "
                f"< num_hidden_layers ({num_hidden_layers})"
            )

    def local_layer_indices(self, num_hidden_layers: int) -> set[int]:
        head_k, tail_k = self.edge_head_tail_layers
        if self.mode == "head_tail":
            self.validate_num_hidden_layers(num_hidden_layers)
        if self.is_edge:
            if (head_k, tail_k) == (0, 0):
                return set()
            return set(range(head_k)) | set(
                range(num_hidden_layers - tail_k, num_hidden_layers)
            )
        return set(range(head_k, num_hidden_layers - tail_k))

    def __repr__(self) -> str:
        return (
            f"[LWD] config(enabled={self.enabled}, role={self.role!r}, "
            f"mode={self.mode!r}, path={self.path!r}, "
            f"instance_id={self.instance_id}, "
            f"edge_head_tail_layers={list(self.edge_head_tail_layers)})"
        )
