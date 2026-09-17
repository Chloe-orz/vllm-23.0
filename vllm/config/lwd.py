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
``lwd_config.enabled`` is the single master switch, and the no-CLI
narrow fields ``ParallelConfig.enable_lwd`` / ``is_edge_node`` are
back-filled from it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

from vllm import envs

_VALID_LWD_ROLES = ("edge", "cloud")
_VALID_LWD_MODES = ("head_tail", "embedding_only", "prefill_only")


def _parse_lwd_layer_split(value: Any) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"[LWD] edge_head_tail_layers must be a 2-element array [head_k, tail_k], got {value!r}")
    if len(value) != 2:
        raise ValueError(f"[LWD] edge_head_tail_layers must have exactly 2 elements, got {value!r}")
    split = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"[LWD] edge_head_tail_layers elements must be non-negative integers, got {value!r}")
        split.append(item)
    return split[0], split[1]  # type: ignore[return-value]


@dataclass(frozen=True)
class LwdConfig:
    enabled: bool = False
    role: str = "edge"
    """Process role: "edge" or "cloud"."""
    mode: str = "head_tail"
    """Layer distribution mode: "head_tail", "embedding_only" or "prefill_only"."""
    edge_head_tail_layers: tuple[int, int] = (1, 1)
    """Fixed 2-element (head_k, tail_k) asymmetric splits allowed."""
    pre_out_host: str = "127.0.0.1"
    pre_out_port: int = 5558
    post_out_port: int = 5559
    post_out_bind: str = "*"
    hello_timeout_s: float = 600.0
    scheduler_name: str = "prefill_first"
    publish_queue_max: int = 1000
    debug: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "LwdConfig":
        cfg = cls(
            enabled=bool(raw.get("enabled", False)),
            role=str(raw.get("role", "edge")),
            mode=str(raw.get("mode", "head_tail")),
            edge_head_tail_layers=_parse_lwd_layer_split(raw.get("edge_head_tail_layers", [1, 1])),
            pre_out_host=str(raw.get("pre_out_host", "127.0.0.1")),
            pre_out_port=int(raw.get("pre_out_port", 5558)),
            post_out_port=int(raw.get("post_out_port", 5559)),
            post_out_bind=str(raw.get("post_out_bind", "*")),
            hello_timeout_s=float(raw.get("hello_timeout_s", 600.0)),
            scheduler_name=str(raw.get("scheduler", "prefill_first")),
            publish_queue_max=int(raw.get("publish_queue_max", 1000)),
            debug=bool(raw.get("debug", False)),
        )
        cfg.validate()
        env = {
            "pre_out_host": envs.VLLM_ASCEND_LWD_PRE_OUT_HOST,
            "pre_out_port": envs.VLLM_ASCEND_LWD_PRE_OUT_PORT,
            "post_out_port": envs.VLLM_ASCEND_LWD_POST_OUT_PORT,
            "post_out_bind": envs.VLLM_ASCEND_LWD_POST_OUT_BIND,
            "hello_timeout_s": envs.VLLM_ASCEND_LWD_HELLO_TIMEOUT_S,
            "debug": envs.VLLM_ASCEND_LWD_DEBUG,
        }
        env = {k: v for k, v in env.items() if v}
        return replace(cfg, **env) if env else cfg

    @property
    def is_edge(self) -> bool:
        return self.role == "edge"

    def validate(self) -> None:
        if self.role not in _VALID_LWD_ROLES:
            raise ValueError(f"[LWD] role must be one of {_VALID_LWD_ROLES}, got {self.role!r}")
        if self.mode not in _VALID_LWD_MODES:
            raise ValueError(f"[LWD] mode must be one of {_VALID_LWD_MODES}, got {self.mode!r}")
        head_k, tail_k = self.edge_head_tail_layers
        if self.mode in ("embedding_only", "prefill_only"):
            if (head_k, tail_k) != (0, 0):
                raise ValueError(f"[LWD] {self.mode} requires edge_head_tail_layers [0, 0], got [{head_k},{tail_k}]")
        elif head_k + tail_k < 1:
            raise ValueError(f"[LWD] 'head_tail' mode requires at least one edge layer, got [{head_k}, {tail_k}]")

    def validate_num_hidden_layers(self, num_hidden_layers: int) -> None:
        head_k, tail_k = self.edge_head_tail_layers
        if head_k + tail_k >= num_hidden_layers:
            raise ValueError(
                "[LWD] layer split must leave a non-empty middle range for "
                f"the cloud: head_k + tail_k ({head_k + tail_k}) must be "
                f"< num_hidden_layers ({num_hidden_layers})")

    def local_layer_indices(self, num_hidden_layers: int) -> set[int]:
        head_k, tail_k = self.edge_head_tail_layers
        if self.mode == "head_tail":
            self.validate_num_hidden_layers(num_hidden_layers)
        if self.is_edge:
            if (head_k, tail_k) == (0, 0):
                return set()
            return (set(range(head_k)) | set(range(num_hidden_layers - tail_k, num_hidden_layers)))
        return set(range(head_k, num_hidden_layers - tail_k))

    def __repr__(self) -> str:
        return (f"[LWD] config(enabled={self.enabled}, role={self.role!r}, "
                f"mode={self.mode!r}, edge_head_tail_layers="
                f"{list(self.edge_head_tail_layers)})")
