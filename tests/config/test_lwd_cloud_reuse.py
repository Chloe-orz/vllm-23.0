# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cloud reuse configuration contracts, not shared-cloud execution tests."""

import json
import logging
import pickle
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from tools.gen_lwd_topology import build_topology, main as generate_main
from vllm.config.lwd import LwdConfig
from vllm.config.lwd_api import LwdAPIConfig
from vllm.config.lwd_topology import LwdTopology
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
    LwdConfig as TransportConfig,
)


@pytest.fixture(autouse=True)
def clear_retired_post_out(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_LWD_POST_OUT_PORT", raising=False)


@pytest.fixture(autouse=True)
def capture_topology_warnings(monkeypatch, caplog):
    # vLLM's logger does not propagate to pytest's root capture handler.
    log = logging.getLogger(LwdTopology.__module__)
    monkeypatch.setattr(log, "handlers", [*log.handlers, caplog.handler])
    monkeypatch.setattr(log, "level", logging.WARNING)


def example(scene, dps):
    clouds = ["10.1.0.1"]
    if scene == "lwd_cluster":
        clouds.append("10.1.0.2")
    return build_topology(
        scene, ["10.0.0.1", "10.0.0.2"], clouds, dps, 5550, False, False
    )


def write_config(tmp_path, raw, role="cloud", instance_id=0):
    path = tmp_path / "lwd_config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return {"path": str(path), "role": role, "instance_id": instance_id}


@pytest.mark.parametrize("scene,nclouds,world", [
    ("cloud_share", 1, 10), ("lwd_cluster", 2, 18),
])
@pytest.mark.parametrize("dps", [1, 2])
def test_every_instance_and_dp_survives_config_downstream(
    tmp_path, monkeypatch, scene, nclouds, world, dps,
):
    raw = example(scene, dps)
    entry = write_config(tmp_path, raw)
    expected_keys = {
        (edge, cloud, dp)
        for edge in range(2)
        for cloud in range(nclouds)
        for dp in range(dps)
    }
    for role, count in (("edge", 2), ("cloud", nclouds)):
        for instance_id in range(count):
            entry.update(role=role, instance_id=instance_id)
            config = LwdConfig.from_dict(entry)
            topology = config.topology
            assert topology.deployment.hccl_world_size == world
            assert len(config.instance.dp) == dps
            local_keys = {
                (c.edge_id, c.cloud_id, c.dp_idx) for c in config.connections
            }
            role_index = 0 if role == "edge" else 1
            assert local_keys == {
                key for key in expected_keys if key[role_index] == instance_id
            }
            for connection in config.connections:
                assert connection.edge.ranks == (connection.edge_id,)
                start = 2 + 8 * connection.cloud_id + (8 // dps) * connection.dp_idx
                assert connection.cloud.ranks == tuple(range(start, start + 8 // dps))
                assert connection.cloud_leader_rank == start
                assert connection.cloud.ctrl_port == 5550 + connection.dp_idx
            # The whole topology is preserved, not replaced by the selected part.
            restored = pickle.loads(pickle.dumps(config))
            assert restored == config
            assert asdict(restored.topology) == asdict(topology)
            LwdAPIConfig().validate_topology(restored)  # No IaaS flags required.
            with monkeypatch.context() as patch:
                patch.setattr(
                    LwdTopology, "from_file",
                    Mock(side_effect=AssertionError("must use the parsed snapshot")),
                )
                for dp_idx in range(dps):
                    transport = TransportConfig.from_vllm_config(SimpleNamespace(
                        lwd_config=restored,
                        parallel_config=SimpleNamespace(data_parallel_index=dp_idx),
                    ))
                    assert set(transport.my_links) == {
                        key for key in local_keys if key[2] == dp_idx
                    }
                    assert transport.topology_digest == topology.digest
                    if role == "cloud":
                        endpoint = f"tcp://*:{5550 + dp_idx}"
                        assert transport.router_bind_endpoint == endpoint
                        assert not transport.dealer_endpoints
                    else:
                        assert transport.router_bind_endpoint is None
                        identity = f"edge-{instance_id}-{dp_idx}".encode()
                        assert transport.dealer_identity == identity
                        assert transport.dealer_endpoints == {
                            (instance_id, cloud, dp_idx):
                            f"tcp://10.1.0.{cloud + 1}:{5550 + dp_idx}"
                            for cloud in range(nclouds)
                        }
            with pytest.raises(ValueError, match="Multi-instance topology was parsed"):
                restored.apply_to_parallel_config(SimpleNamespace())


@pytest.mark.parametrize("scene,nclouds", [
    ("cloud_share", 1), ("lwd_cluster", 2),
])
@pytest.mark.parametrize("dps", [1, 2])
def test_generator_cli_and_config_parser(
    tmp_path, monkeypatch, capsys, scene, dps, nclouds,
):
    path = tmp_path / "lwd_config.yaml"
    monkeypatch.setattr("sys.argv", [
        "gen_lwd_topology.py", "--scene", scene,
        "--edge-machines", "10.0.0.1,10.0.0.2",
        "--cloud-machines", ",".join(f"10.1.0.{i + 1}" for i in range(nclouds)),
        "--dp", str(dps), "--port-base", "5550", "-o", str(path),
    ])
    assert generate_main() == 0
    capsys.readouterr()
    raw = yaml.safe_load(path.read_text())
    assert raw["deployment"]["hccl_world_size"] == 2 + 8 * nclouds
    assert raw["instance_links"] == [
        {"edge": edge, "cloud": cloud}
        for edge in range(2) for cloud in range(nclouds)
    ]
    assert "link=(1,0,0) UP=[1, 2] DOWN=[1, 2]" in path.read_text()
    config = LwdConfig.from_dict({
        "path": str(path), "role": "cloud", "instance_id": 0,
    })
    assert len(config.connections) == 2 * dps
    assert len(config.instance.dp) == dps
    with pytest.raises(ValueError, match="Multi-instance topology was parsed"):
        config.apply_to_parallel_config(SimpleNamespace())


def test_missing_cluster_pair_is_rejected_not_silently_added():
    raw = example("lwd_cluster", 2)
    raw["instance_links"].pop()  # Every instance is still referenced.
    with pytest.raises(ValueError, match="full-mesh"):
        LwdTopology.from_dict(raw)
    assert len(raw["instance_links"]) == 3


@pytest.mark.parametrize("case,message", [
    ("edge_machine", "same rank"),
    ("cloud_size", "eight contiguous ranks"),
    ("unequal_dp", "equally"),
    ("rank_order", "DP-index order"),
    ("same_host_port", "different ctrl_port"),
    ("edge_port", "unknown fields"),
])
def test_invalid_cloud_reuse_layouts(case, message):
    raw = example("cloud_share", 2)
    if case == "edge_machine":
        for dp in raw["edges"][1]["dp"]:
            dp["addr"] = "10.0.0.1"
    elif case == "cloud_size":
        raw["clouds"][0]["dp"][1]["ranks"].pop()
        raw["deployment"]["hccl_world_size"] -= 1
    elif case == "unequal_dp":
        dps = raw["clouds"][0]["dp"]
        dps[1]["ranks"].insert(0, dps[0]["ranks"].pop())
    elif case == "rank_order":
        raw["clouds"][0]["dp"][0]["ranks"].reverse()
    elif case == "same_host_port":
        raw["clouds"][0]["dp"][1]["ctrl_port"] = 5550
    else:
        raw["edges"][0]["dp"][0]["ctrl_port"] = 5550
    with pytest.raises(ValueError, match=message):
        LwdTopology.from_dict(raw)


def test_cloud_dp_may_live_on_different_machines():
    raw = example("cloud_share", 2)
    raw["clouds"][0]["dp"] = [
        {
            "dp_idx": 0, "addr": "10.1.0.1",
            "ranks": list(range(2, 10)), "ctrl_port": 5550,
        },
        {
            "dp_idx": 1, "addr": "10.1.0.2",
            "ranks": list(range(10, 18)), "ctrl_port": 5550,
        },
    ]
    raw["deployment"]["hccl_world_size"] = 18
    topology = LwdTopology.from_dict(raw)
    assert topology.dp("cloud", 0, 1).addr == "10.1.0.2"
    assert topology.dp("cloud", 0, 1).ranks == tuple(range(10, 18))


@pytest.mark.parametrize("scene", ["cloud_share", "lwd_cluster"])
@pytest.mark.parametrize("dps", [1, 2])
def test_serving_logs_all_connections_before_execution_gate(
    tmp_path, monkeypatch, scene, dps,
):
    import vllm.config.vllm as config_module

    log = Mock()
    monkeypatch.setattr(config_module.logger, "info", log)
    for role in ("edge", "cloud"):
        log.reset_mock()
        entry = write_config(tmp_path, example(scene, dps), role)
        target = SimpleNamespace(
            additional_config={"lwd_config": entry},
            lwd_api_config=None,
            parallel_config=SimpleNamespace(),
        )
        with pytest.raises(ValueError, match="Multi-instance topology was parsed"):
            config_module.VllmConfig.__post_init__(target)
        call = next(
            c for c in log.call_args_list if "[config][connections]" in c.args[0]
        )
        connections = json.loads(call.args[-1])
        expected_peers = 2 if role == "cloud" or scene == "lwd_cluster" else 1
        assert len(connections) == expected_peers * dps
        assert not vars(target.parallel_config)  # No runtime projection took place.


def test_cloud_reuse_features_are_preserved_only(tmp_path):
    raw = example("cloud_share", 1)
    raw["feature_ctrl"] = {"enable_early_recv": True, "enable_scramble": True}
    entry = write_config(tmp_path, raw)
    config = LwdConfig.from_dict(entry)
    assert asdict(config.topology.feature_ctrl) == raw["feature_ctrl"]
    with pytest.raises(ValueError, match="Multi-instance topology was parsed"):
        config.apply_to_parallel_config(SimpleNamespace())


@pytest.mark.parametrize("scene", ["cloud_share", "lwd_cluster"])
@pytest.mark.parametrize("dps", [1, 2])
def test_design_examples_do_not_warn(scene, dps, caplog):
    with caplog.at_level(logging.WARNING):
        LwdTopology.from_dict(example(scene, dps))
    assert not caplog.records


@pytest.mark.parametrize("declared,actual", [
    ("cloud_share", "lwd_cluster"),
    ("lwd_cluster", "cloud_share"),
])
def test_scene_mismatch_warns_without_changing_config(declared, actual, caplog):
    raw = example(actual, 1)
    raw["deployment"]["scene"] = declared
    with caplog.at_level(logging.WARNING):
        topology = LwdTopology.from_dict(raw)
    assert f"scene={declared} does not match machine counts" in caplog.text
    assert f"expected scene={actual}" in caplog.text
    assert topology.deployment.scene == declared
    assert [asdict(link) for link in topology.instance_links] == raw["instance_links"]


@pytest.mark.parametrize("scene", ["cloud_share", "lwd_cluster"])
def test_generator_degenerate_scene_warns(tmp_path, monkeypatch, caplog):
    # Both selectors previously allowed 1 edge / 1 cloud without any warning.
    path = tmp_path / "lwd_config.yaml"
    monkeypatch.setattr("sys.argv", [
        "gen_lwd_topology.py", "--scene", scene,
        "--edge-machines", "10.0.0.1", "--cloud-machines", "10.1.0.1",
        "--dp", "1", "-o", str(path),
    ])
    with caplog.at_level(logging.WARNING):
        assert generate_main() == 0
    assert "expected scene=single_instance" in caplog.text
    assert yaml.safe_load(path.read_text())["deployment"]["scene"] == scene


@pytest.mark.parametrize("role", ["edges", "clouds"])
@pytest.mark.parametrize("dps", [1, 2])
def test_swapped_machine_rank_blocks_are_rejected(role, dps):
    raw = example("lwd_cluster", dps)
    first, second = raw[role]
    for left, right in zip(first["dp"], second["dp"]):
        left["ranks"], right["ranks"] = right["ranks"], left["ranks"]
    with pytest.raises(ValueError, match="machine declaration order"):
        LwdTopology.from_dict(raw)


@pytest.mark.parametrize("role", ["edges", "clouds"])
def test_machine_order_is_declaration_order_not_instance_id(role):
    raw = example("lwd_cluster", 2)
    # Renumber identities only: the declared machine/rank order is unchanged.
    # Links are full mesh, so all renamed IDs are still referenced.
    raw[role][0]["id"], raw[role][1]["id"] = 1, 0
    topology = LwdTopology.from_dict(raw)
    instances = getattr(topology, role)
    assert [instance.id for instance in instances] == [1, 0]
    selected = topology.instance("edge" if role == "edges" else "cloud", 1)
    assert selected.dp[0].addr == raw[role][0]["dp"][0]["addr"]
    assert selected.dp[0].ranks == tuple(raw[role][0]["dp"][0]["ranks"])


def test_nonconsecutive_ports_warn_but_remain_usable_config(caplog):
    raw = example("cloud_share", 2)
    raw["clouds"][0]["dp"][1]["ctrl_port"] = 5560
    with caplog.at_level(logging.WARNING):
        topology = LwdTopology.from_dict(raw)
    assert "deviate from port_base + dp_idx" in caplog.text
    assert "inferred port_base=5550" in caplog.text
    assert topology.dp("cloud", 0, 1).ctrl_port == 5560
    assert {c.cloud.ctrl_port for c in topology.connections("cloud", 0)} == {
        5550, 5560,
    }


def test_custom_port_bases_are_independent_per_machine(caplog):
    raw = example("lwd_cluster", 2)
    for cloud, base in zip(raw["clouds"], (6453, 7200)):
        for dp in cloud["dp"]:
            dp["ctrl_port"] = base + dp["dp_idx"]
    with caplog.at_level(logging.WARNING):
        topology = LwdTopology.from_dict(raw)
    assert not caplog.records
    assert topology.dp("cloud", 0, 1).ctrl_port == 6454
    assert topology.dp("cloud", 1, 1).ctrl_port == 7201


def test_cross_machine_dp_may_restart_port_numbering(caplog):
    raw = example("cloud_share", 2)
    raw["deployment"].update(scene="lwd_cluster", hccl_world_size=18)
    for dp, start in zip(raw["clouds"][0]["dp"], (2, 10)):
        dp.update(
            addr=f"10.1.0.{dp['dp_idx'] + 1}",
            ranks=list(range(start, start + 8)),
            ctrl_port=5550,
        )
    with caplog.at_level(logging.WARNING):
        topology = LwdTopology.from_dict(raw)
    assert not caplog.records
    assert topology.dp("cloud", 0, 1).ctrl_port == 5550
