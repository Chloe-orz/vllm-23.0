# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration-only IaaS API attachment; no service or routing implementation."""

from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any

from .lwd import LwdConfig, lwd_entry_from_additional


def _endpoint(value: str | None, flag: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"[LWD] {flag} requires IP:port")
    host, sep, port = value.rpartition(":")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    elif ":" in host:
        raise ValueError(f"[LWD] {flag}: IPv6 must use [IP]:port")
    if not sep or not port.isascii() or not port.isdecimal():
        raise ValueError(f"[LWD] {flag} requires IP:port")
    if not 1 <= int(port) <= 65535:
        raise ValueError(f"[LWD] {flag}: port must be in [1, 65535]")
    try:
        addr = ip_address(host)
    except ValueError as exc:
        raise ValueError(f"[LWD] {flag}: invalid IP {host!r}") from exc
    if addr.is_unspecified or addr.is_multicast:
        raise ValueError(f"[LWD] {flag}: use a concrete unicast IP")


@dataclass(frozen=True)
class LwdAPIConfig:
    """Retained frontend options. These do not create an API RPC connection.

    Despite its name, api_server_rpc_port contains an IP:port endpoint.
    No options means no IaaS attachment request (including MaaS).
    """

    api_server_rpc_port: str | None = None
    api_server_attach: str | None = None

    def __post_init__(self) -> None:
        if self.api_server_rpc_port is not None and self.api_server_attach is not None:
            raise ValueError(
                "[LWD] --api-server-rpc-port and --api-server-attach "
                "are mutually exclusive"
            )
        _endpoint(self.api_server_rpc_port, "--api-server-rpc-port")
        _endpoint(self.api_server_attach, "--api-server-attach")

    @property
    def requested(self) -> bool:
        return (
            self.api_server_rpc_port is not None or self.api_server_attach is not None
        )

    def validate_identity(self, role: str, instance_id: int) -> None:
        if not self.requested:
            return
        if role != "edge":
            raise ValueError("[LWD] IaaS API options apply to edge instances only")
        if self.api_server_rpc_port is not None and instance_id != 0:
            raise ValueError("[LWD] --api-server-rpc-port requires edge instance_id=0")
        if self.api_server_attach is not None and instance_id == 0:
            raise ValueError(
                "[LWD] --api-server-attach requires a non-primary edge instance"
            )

    @classmethod
    def from_namespace(cls, args: Any) -> "LwdAPIConfig":
        """Validate CLI options without opening the topology file."""
        config = cls(
            api_server_rpc_port=getattr(args, "api_server_rpc_port", None),
            api_server_attach=getattr(args, "api_server_attach", None),
        )
        if config.requested:
            entry = lwd_entry_from_additional(getattr(args, "additional_config", None))
            if entry is None:
                raise ValueError("[LWD] IaaS API options require lwd_config.path")
            config.validate_identity(entry["role"], entry["instance_id"])
            if getattr(args, "headless", False) or getattr(args, "grpc", False):
                raise ValueError("[LWD] IaaS API options require the HTTP frontend")
        return config

    def validate_topology(self, lwd: LwdConfig) -> None:
        """Check the already parsed topology, before the execution capability gate."""
        if not self.requested:
            return
        if not lwd.enabled or lwd.topology is None:
            raise ValueError("[LWD] IaaS API options require lwd_config.path")
        self.validate_identity(lwd.role, lwd.instance_id)
        lwd.topology.instance(lwd.role, lwd.instance_id)
        if len(lwd.topology.edges) < 2 or len(lwd.topology.clouds) < 2:
            raise ValueError(
                "[LWD] IaaS API options require multiple edge/cloud instances"
            )
