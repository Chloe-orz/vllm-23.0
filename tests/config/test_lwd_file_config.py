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


@pytest.mark.parametrize(
    "role,tp,ranks", [("edge", 1, (0,)), ("cloud", 8, tuple(range(1, 9)))]
)
def test_single_dp_projection_and_transport(monkeypatch, role, tp, ranks):
    monkeypatch.setenv("VLLM_ASCEND_LWD_POST_OUT_PORT", "6454")
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

    # Transport uses the captured config, not subsequent env/JSON overrides.
    monkeypatch.setenv("VLLM_ASCEND_LWD_POST_OUT_PORT", "6500")
    monkeypatch.setenv("VLLM_ASCEND_LWD_PRE_OUT_PORT", "6501")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", lambda *a, **kw: pytest.fail("YAML was reopened"))
        transport = TransportConfig.from_vllm_config(SimpleNamespace(lwd_config=config))
    assert transport.lwd_pre_out_endpoint() == "tcp://10.1.0.1:5550"
    assert transport.lwd_post_out_connect_endpoint() == "tcp://10.0.0.1:6454"
    assert transport.lwd_wire_store_init_method() == "tcp://10.0.0.1:29600"


def test_multi_dp_parses_but_cannot_execute(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_LWD_POST_OUT_PORT", raising=False)
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
    monkeypatch.delenv("VLLM_ASCEND_LWD_POST_OUT_PORT", raising=False)
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


@pytest.mark.parametrize("port", ["bad", "", "0", "65536", "29600"])
def test_invalid_post_out_port(monkeypatch, port):
    monkeypatch.setenv("VLLM_ASCEND_LWD_POST_OUT_PORT", port)
    with pytest.raises(ValueError, match="POST_OUT_PORT"):
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


def test_transport_log_contains_the_returned_config(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LWD_POST_OUT_PORT", "6454")
    config = LwdConfig.from_dict(entry("cloud"))
    log = Mock()
    monkeypatch.setattr(lwd_edge_assemble.logger, "info_once", log)
    transport = TransportConfig.from_vllm_config(SimpleNamespace(lwd_config=config))
    message, role, instance_id, payload = log.call_args.args
    assert "[LWD][config][transport]" in message
    assert (role, instance_id) == ("cloud", 0)
    assert json.loads(payload) == asdict(transport)


def test_parsed_log_payload_preserves_all_dps(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_LWD_POST_OUT_PORT", raising=False)
    config = LwdConfig.from_dict(entry(filename="lwd_config_2dp.yaml"))
    payload = json.loads(json.dumps(asdict(config)))
    assert payload["path"] == entry(filename="lwd_config_2dp.yaml")["path"]
    assert len(payload["topology"]["edges"][0]["dp"]) == 2
    assert payload["topology"]["clouds"][0]["dp"][1]["ranks"] == [5, 6, 7, 8]
    assert payload["topology"]["feature_ctrl"]["enable_early_recv"] is False
