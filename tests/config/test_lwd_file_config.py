# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration-only coverage; no device or distributed initialization."""

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from vllm.config.lwd import LwdConfig, lwd_entry_from_additional
from vllm.config.lwd_topology import LwdTopology
from vllm.v1.lwd_control.control_edge_scheduler import lwd_edge_assemble
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
    LwdConfig as TransportConfig,
)

EXAMPLES = Path(__file__).resolve().parents[2] / "etc" / "lwd"


def entry(role="edge", filename="lwd_config.yaml"):
    return {"path": str(EXAMPLES / filename), "role": role, "instance_id": 0}


def parallel(tp):
    # Exercise the projection in isolation; this is not an executor/NPU test.
    return SimpleNamespace(
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_size=1,
        data_parallel_backend="mp",
        data_parallel_size_local=1,
        data_parallel_rank=0,
        data_parallel_external_lb=False,
        data_parallel_hybrid_lb=False,
        nnodes=1,
        master_addr="127.0.0.1",
        distributed_executor_backend="mp",
        lwd_config=SimpleNamespace(),
    )


def _clear_retired_port_env(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_LWD_POST_OUT_PORT", raising=False)


@pytest.mark.parametrize(
    "role,tp,ranks", [("edge", 1, (0,)), ("cloud", 8, tuple(range(1, 9)))]
)
def test_single_dp_projection_and_transport(monkeypatch, role, tp, ranks):
    _clear_retired_port_env(monkeypatch)
    config = LwdConfig.from_dict(entry(role))
    pc = parallel(tp)
    config.apply_to_parallel_config(pc)
    assert config.enabled and config.mode == "prefill_only"
    assert config.dp.ranks == ranks
    assert pc.tensor_parallel_size == tp
    assert pc.pipeline_parallel_size == 1
    assert pc.world_size == 9
    assert pc.node_rank == (0 if role == "edge" else 1)
    assert pc.lwd_config.edge_npu_count == 1
    assert pc.lwd_config.cloud_npu_count == 8
    assert pickle.loads(pickle.dumps(config)) == config

    # Transport is derived from the resolved topology only: endpoints are
    # computed without reopening the YAML and cannot drift via env overrides.
    monkeypatch.setenv("VLLM_ASCEND_LWD_PRE_OUT_PORT", "6501")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", lambda *a, **kw: pytest.fail("YAML was reopened"))
        transport = TransportConfig.from_vllm_config(
            SimpleNamespace(lwd_config=config)
        )
    assert transport.my_links == ((0, 0, 0),)
    assert transport.wire_store_init_method == "tcp://76.76.26.18:29600"
    if role == "cloud":
        assert transport.router_bind_endpoint == "tcp://*:6453"
        assert not transport.dealer_endpoints
        assert transport.dealer_identity is None
    else:
        assert transport.router_bind_endpoint is None
        assert transport.dealer_endpoints == {(0, 0, 0): "tcp://76.76.26.234:6453"}
        assert transport.dealer_identity == b"edge-0-0"
    assert transport.edge_npu_count == 1 and transport.cloud_npu_count == 8
    assert transport.topology_digest  # from_file computes it


def test_multi_dp_parses_but_cannot_execute(monkeypatch):
    _clear_retired_port_env(monkeypatch)
    config = LwdConfig.from_dict(entry(filename="lwd_config_2dp.yaml"))
    assert len(config.instance.dp) == 2
    assert config.topology.deployment.hccl_world_size == 9
    assert config.topology.dp("cloud", 0, 1).ranks == (5, 6, 7, 8)
    assert config.topology.dp("edge", 0, 1).ranks == (0,)
    with pytest.raises(ValueError, match="Multi-DP topology was parsed"):
        config.apply_to_parallel_config(parallel(1))
    with pytest.raises(ValueError, match="select a DP explicitly"):
        _ = config.dp


def test_tp_mismatch_does_not_overwrite(monkeypatch):
    _clear_retired_port_env(monkeypatch)
    pc = parallel(1)
    with pytest.raises(
        ValueError, match="CLI tensor_parallel_size=1.*YAML ranks count=8"
    ):
        LwdConfig.from_dict(entry("cloud")).apply_to_parallel_config(pc)
    assert pc.tensor_parallel_size == 1


@pytest.mark.parametrize(
    "raw", [None, {}, {"enabled": True}, {"path": ""}, {"path": 3}]
)
def test_present_invalid_entry_is_not_disabled(raw):
    with pytest.raises(ValueError, match="lwd_config"):
        lwd_entry_from_additional({"lwd_config": raw})


def test_absent_entry_and_no_early_file_io():
    assert lwd_entry_from_additional({"enable_cpu_binding": True}) is None
    assert not LwdConfig.from_dict(None).enabled
    raw = entry()
    raw["path"] = "/does/not/exist.yaml"
    assert lwd_entry_from_additional({"lwd_config": raw}) == raw
    with pytest.raises(ValueError, match="/does/not/exist.yaml"):
        LwdConfig.from_dict(raw)


@pytest.mark.parametrize(
    "key,value",
    [
        ("enabled", True),
        ("mode", "prefill_only"),
        ("instance_id", True),
        ("instance_id", "0"),
        ("role", "invalid"),
    ],
)
def test_invalid_public_fields(key, value):
    raw = entry()
    raw[key] = value
    with pytest.raises(ValueError):
        lwd_entry_from_additional({"lwd_config": raw})


@pytest.mark.parametrize("port", ["6454", "5559"])
def test_retired_post_out_port_env_rejected(monkeypatch, port):
    # ROUTER/DEALER 控制面边侧无端口;旧脚本带着该 env 一律 fail-fast
    monkeypatch.setenv("VLLM_ASCEND_LWD_POST_OUT_PORT", port)
    with pytest.raises(ValueError, match="POST_OUT_PORT.*retired"):
        LwdConfig.from_dict(entry())


@pytest.mark.parametrize("mode", [1, 2, True, "0"])
def test_invalid_mode(mode):
    raw = yaml.safe_load((EXAMPLES / "lwd_config.yaml").read_text())
    raw["deployment"]["mode"] = mode
    with pytest.raises(ValueError, match="mode"):
        LwdTopology.from_dict(raw)


def test_yaml_duplicate_and_typo(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("deployment: {}\ndeployment: {}\n")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        LwdTopology.from_file(str(path))
    raw = yaml.safe_load((EXAMPLES / "lwd_config.yaml").read_text())
    raw["feature_ctrl"]["enable_eary_recv"] = False
    with pytest.raises(ValueError, match="use enable_early_recv"):
        LwdTopology.from_dict(raw)


def test_feature_defaults_and_rank_validation():
    raw = yaml.safe_load((EXAMPLES / "lwd_config.yaml").read_text())
    raw["feature_ctrl"] = {}
    topology = LwdTopology.from_dict(raw)
    assert not topology.feature_ctrl.enable_early_recv
    assert not topology.feature_ctrl.enable_scramble
    raw["clouds"][0]["dp"][0]["ranks"][-1] = 7
    with pytest.raises(ValueError, match="duplicate ranks"):
        LwdTopology.from_dict(raw)


def test_topology_digest_stable_and_file_bound():
    path = EXAMPLES / "lwd_config.yaml"
    first = LwdTopology.from_file(str(path))
    second = LwdTopology.from_file(str(path))
    assert first.digest and first.digest == second.digest
    assert len(first.digest) == 16
    assert LwdTopology.from_dict(
        yaml.safe_load(path.read_text())
    ).digest == ""  # from_dict has no raw bytes


def test_transport_log_contains_the_returned_config(monkeypatch):
    _clear_retired_port_env(monkeypatch)
    config = LwdConfig.from_dict(entry("cloud"))
    log = Mock()
    monkeypatch.setattr(lwd_edge_assemble.logger, "info_once", log)
    transport = TransportConfig.from_vllm_config(SimpleNamespace(lwd_config=config))
    message = log.call_args.args[0]
    assert "[LWD][config][transport]" in message
    payload = json.loads(log.call_args.args[-1])
    assert payload["router_bind"] == "tcp://*:6453"
    assert payload["wire_store"] == "tcp://76.76.26.18:29600"
    assert asdict(transport)["my_links"] == ((0, 0, 0),)


def test_parsed_log_payload_preserves_all_dps(monkeypatch):
    _clear_retired_port_env(monkeypatch)
    config = LwdConfig.from_dict(entry(filename="lwd_config_2dp.yaml"))
    payload = json.loads(json.dumps(asdict(config)))
    assert payload["path"] == entry(filename="lwd_config_2dp.yaml")["path"]
    assert len(payload["topology"]["edges"][0]["dp"]) == 2
    assert payload["topology"]["clouds"][0]["dp"][1]["ranks"] == [5, 6, 7, 8]
    assert payload["topology"]["feature_ctrl"]["enable_early_recv"] is False
