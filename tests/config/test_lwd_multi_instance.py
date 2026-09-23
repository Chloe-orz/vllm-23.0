# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Multi-instance configuration contracts, without workers or communication."""

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from vllm.config.lwd import LwdConfig
from vllm.config.lwd_api import LwdAPIConfig
from vllm.config.lwd_topology import LwdTopology

EXAMPLES = Path(__file__).resolve().parents[2] / "etc" / "lwd"


def entry(role="edge", instance_id=0, dps=1):
    return {
        "path": str(EXAMPLES / f"lwd_config_multi_instance_{dps}dp.yaml"),
        "role": role,
        "instance_id": instance_id,
    }


@pytest.fixture(autouse=True)
def clear_retired_post_out(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_LWD_POST_OUT_PORT", raising=False)


@pytest.mark.parametrize("dps", [1, 2])
@pytest.mark.parametrize("role", ["edge", "cloud"])
@pytest.mark.parametrize("instance_id", [0, 1])
def test_instance_selection_and_connections(dps, role, instance_id):
    config = LwdConfig.from_dict(entry(role, instance_id, dps))
    topology = config.topology
    assert topology is not None
    assert len(config.instance.dp) == dps
    assert topology.deployment.hccl_world_size == 17
    unique_ranks = {
        rank
        for instance in (*topology.edges, *topology.clouds)
        for dp in instance.dp
        for rank in dp.ranks
    }
    assert unique_ranks == set(range(17))
    assert [c.id for c in topology.downstream_clouds(instance_id)] == [instance_id]
    assert [e.id for e in topology.upstream_edges(instance_id)] == [instance_id]
    assert len(config.connections) == dps
    for dp_idx, link in enumerate(config.connections):
        assert (link.edge_id, link.cloud_id, link.dp_idx) == (
            instance_id,
            instance_id,
            dp_idx,
        )
        assert link.edge.ranks == (0,)
        start = 1 + instance_id * 8 + dp_idx * (8 // dps)
        assert link.cloud.ranks == tuple(range(start, start + 8 // dps))
        assert link.cloud_leader_rank == start
        assert link.cloud.addr == f"10.1.0.{instance_id + 1}"
        assert link.cloud.ctrl_port == 5550 + dp_idx
    assert pickle.loads(pickle.dumps(config)) == config
    assert json.loads(json.dumps(asdict(config)))["instance_id"] == instance_id
    with pytest.raises(ValueError, match="Multi-instance topology was parsed"):
        config.apply_to_parallel_config(SimpleNamespace())


def test_links_are_explicit_not_inferred_from_ids_or_scene():
    raw = yaml.safe_load(Path(entry()["path"]).read_text())
    raw["instance_links"] = [{"edge": 0, "cloud": 1}, {"edge": 1, "cloud": 0}]
    topology = LwdTopology.from_dict(raw)
    assert topology.downstream_clouds(0)[0].id == 1
    assert topology.upstream_edges(0)[0].id == 1
    assert topology.connections("edge", 0)[0].cloud_id == 1
    with pytest.raises(ValueError, match="Instance not found"):
        topology.connections("cloud", 2)


@pytest.mark.parametrize(
    "case", ["world", "rank_address", "cloud_overlap", "port", "link", "dp"]
)
def test_invalid_multi_instance_topologies(case):
    raw = yaml.safe_load(Path(entry(dps=2)["path"]).read_text())
    if case == "world":
        raw["deployment"]["hccl_world_size"] = 18
    elif case == "rank_address":
        raw["edges"][1]["dp"][0]["addr"] = "10.0.0.2"
    elif case == "cloud_overlap":
        raw["clouds"][0]["dp"][1]["ranks"][0] = 1
    elif case == "port":
        raw["clouds"][0]["dp"][1]["ctrl_port"] = 5550
    elif case == "link":
        raw["instance_links"][1]["cloud"] = 2
    else:
        raw["edges"][1]["dp"].pop()
    with pytest.raises(ValueError):
        LwdTopology.from_dict(raw)


@pytest.mark.parametrize("dps", [1, 2])
@pytest.mark.parametrize("instance_id", [0, 1])
def test_iaas_and_maas_config(dps, instance_id):
    config = LwdConfig.from_dict(entry(instance_id=instance_id, dps=dps))
    flag = "api_server_rpc_port" if instance_id == 0 else "api_server_attach"
    kwargs = {flag: "10.0.0.1:29550"}
    api = LwdAPIConfig(**kwargs)
    api.validate_topology(config)
    assert api.requested
    assert pickle.loads(pickle.dumps(api)) == api
    assert json.loads(json.dumps(asdict(api)))[next(iter(kwargs))] == "10.0.0.1:29550"
    LwdAPIConfig().validate_topology(config)  # MaaS need not supply API flags.


@pytest.mark.parametrize(
    "endpoint",
    [
        "29550",
        "",
        "host:1",
        "10.0.0.1:0",
        "10.0.0.1:65536",
        "0.0.0.0:1",
        "224.0.0.1:1",
        "::1:29550",
    ],
)
def test_invalid_api_endpoint(endpoint):
    with pytest.raises(ValueError):
        LwdAPIConfig(api_server_rpc_port=endpoint)


def test_api_endpoint_mutual_exclusion():
    with pytest.raises(ValueError, match="mutually exclusive"):
        LwdAPIConfig("10.0.0.1:29550", "10.0.0.1:29550")
    assert LwdAPIConfig(api_server_attach="[::1]:29550").requested


@pytest.mark.parametrize(
    "role,instance_id,kwargs",
    [
        ("cloud", 0, {"api_server_rpc_port": "10.0.0.1:29550"}),
        ("edge", 1, {"api_server_rpc_port": "10.0.0.1:29550"}),
        ("edge", 0, {"api_server_attach": "10.0.0.1:29550"}),
    ],
)
def test_invalid_api_identity(role, instance_id, kwargs):
    with pytest.raises(ValueError):
        LwdAPIConfig(**kwargs).validate_topology(
            LwdConfig.from_dict(entry(role, instance_id))
        )


def test_api_flags_require_multi_instance_lwd():
    api = LwdAPIConfig(api_server_rpc_port="10.0.0.1:29550")
    with pytest.raises(ValueError, match="require lwd_config.path"):
        api.validate_topology(LwdConfig.from_dict(None))
    raw = {"path": str(EXAMPLES / "lwd_config.yaml"), "role": "edge", "instance_id": 0}
    with pytest.raises(ValueError, match="require multiple"):
        api.validate_topology(LwdConfig.from_dict(raw))


def test_config_boundary_logs_before_runtime_rejection(monkeypatch):
    from unittest.mock import Mock

    import vllm.config.vllm as config_module

    log = Mock()
    monkeypatch.setattr(config_module.logger, "info", log)
    api = LwdAPIConfig(api_server_attach="10.0.0.1:29550")
    # Exercise the real boundary without constructing a model or any workers.
    target = SimpleNamespace(
        additional_config={"lwd_config": entry(instance_id=1, dps=2)},
        lwd_api_config=api,
        parallel_config=SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="Multi-instance topology was parsed"):
        config_module.VllmConfig.__post_init__(target)
    assert target.lwd_api_config == api
    calls = {call.args[0]: call.args[1:] for call in log.call_args_list}
    api_call = next(args for msg, args in calls.items() if "[config][api]" in msg)
    assert json.loads(api_call[0])["api_server_attach"] == "10.0.0.1:29550"
    links_call = next(
        args for msg, args in calls.items() if "[config][connections]" in msg
    )
    links = json.loads(links_call[-1])
    assert len(links) == 2
    assert all(link["edge_id"] == link["cloud_id"] == 1 for link in links)


def test_config_boundary_validates_api_before_execution_gate():
    from vllm.config.vllm import VllmConfig

    target = SimpleNamespace(
        additional_config={"lwd_config": entry(instance_id=1)},
        lwd_api_config=LwdAPIConfig(api_server_rpc_port="10.0.0.1:29550"),
        parallel_config=SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="requires edge instance_id=0"):
        VllmConfig.__post_init__(target)
