# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Connection identity survives generator, config and scheduler boundaries."""

import pickle
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.gen_lwd_topology import build_topology, connection_comments, output_path
from vllm.config.lwd import LwdConfig
from vllm.config.lwd_topology import LwdTopology
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_phase_scheduler import (
    LwdCloudPhaseScheduler,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LWD_WIRE_VERSION,
    LwdC2eNotify,
    LwdRangeNotify,
    LwdRegisterAckNotify,
    LwdRegisterNotify,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
    LwdConfig as TransportConfig,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    lwd_build_unembed_batch,
)


@pytest.fixture(autouse=True)
def clear_retired_post_out(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_LWD_POST_OUT_PORT", raising=False)


@pytest.mark.parametrize("scene,clouds,world", [
    ("single_instance", ["10.1.0.1"], 9),
    ("edge_share", ["10.1.0.1", "10.1.0.2"], 17),
])
@pytest.mark.parametrize("dps", [1, 2])
def test_generated_yaml_to_dp_transport(tmp_path, scene, clouds, world, dps):
    import yaml

    raw = build_topology(scene, ["10.0.0.1"], clouds, dps, 5550, False, False)
    path = tmp_path / "lwd_config.yaml"
    path.write_text(yaml.safe_dump(raw))
    topology = LwdTopology.from_file(str(path))
    assert topology.deployment.hccl_world_size == world
    assert all(dp.ranks == (0,) for edge in topology.edges for dp in edge.dp)
    assert {
        r for instance in (*topology.edges, *topology.clouds)
        for dp in instance.dp for r in dp.ranks
    } == set(range(world))
    assert "UP=[0, 1] DOWN=[0, 1]" in connection_comments(raw)
    for instance_id in range(len(clouds)):
        for dp_idx in range(dps):
            transports = []
            for role in ("edge", "cloud"):
                config = LwdConfig.from_dict({
                    "path": str(path), "role": role, "instance_id": instance_id,
                })
                # Native DP rank has been normalized to zero; index remains.
                pc = SimpleNamespace(data_parallel_index=dp_idx, data_parallel_rank=0)
                assert config.dp_for_parallel(pc).dp_idx == dp_idx
                transports.append(TransportConfig.from_vllm_config(
                    SimpleNamespace(lwd_config=config, parallel_config=pc)
                ))
            edge, cloud = transports
            key = (instance_id, instance_id, dp_idx)
            assert edge.my_links == cloud.my_links == (key,)
            assert edge.dealer_identity == f"edge-{instance_id}-{dp_idx}".encode()
            assert edge.dealer_endpoints[key] == (
                f"tcp://{clouds[instance_id]}:{5550 + dp_idx}"
            )
            assert cloud.router_bind_endpoint == f"tcp://*:{5550 + dp_idx}"
            assert edge.topology_digest == cloud.topology_digest == topology.digest
    if scene != "single_instance" or dps != 1:
        with pytest.raises(ValueError, match="topology was parsed"):
            topology.validate_single_dp_runtime()


@pytest.mark.parametrize("scene", ["cloud_share", "lwd_cluster"])
@pytest.mark.parametrize("dps", [1, 2])
def test_cloud_reuse_remains_configuration_only(tmp_path, scene, dps):
    import yaml

    clouds = ["10.1.0.1"] if scene == "cloud_share" else ["10.1.0.1", "10.1.0.2"]
    raw = build_topology(
        scene, ["10.0.0.1", "10.0.0.2"], clouds, dps, 5550, False, False
    )
    path = tmp_path / "lwd_config.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = LwdConfig.from_dict({
        "path": str(path), "role": "cloud", "instance_id": 0,
    })
    assert len(config.connections) == 2 * dps
    assert pickle.loads(pickle.dumps(config)) == config
    with pytest.raises(ValueError, match="Multi-instance topology was parsed"):
        config.apply_to_parallel_config(SimpleNamespace())


def test_generator_directory_and_prefix_paths(tmp_path):
    assert output_path(str(tmp_path), 1, batch=False) == tmp_path / "lwd_config.yaml"
    assert output_path(str(tmp_path), 2, batch=True) == tmp_path / "lwd_config_2dp.yaml"
    assert output_path("example.yaml", 2, batch=True) == Path("example_2dp.yaml")


def test_generator_rejects_uneven_dp_split():
    with pytest.raises(SystemExit):
        build_topology(
            "single_instance", ["10.0.0.1"], ["10.1.0.1"], 3, 5550, False, False
        )


def test_runtime_rejects_multiple_edge_ranks_before_device_initialization():
    raw = build_topology(
        "single_instance", ["10.0.0.1"], ["10.1.0.1"], 1, 5550, False, False
    )
    raw["edges"][0]["dp"][0]["ranks"] = [0, 1]
    raw["clouds"][0]["dp"][0]["ranks"] = list(range(2, 10))
    raw["deployment"]["hccl_world_size"] = 10
    topology = LwdTopology.from_dict(raw)
    with pytest.raises(ValueError, match="exactly one edge rank"):
        topology.validate_single_dp_runtime()


def test_cloud_prefill_preserves_notified_connection():
    notify = LwdRangeNotify("1#1#request", 0, 3, 7, edge_id=1, dp_idx=1)
    scheduler = LwdCloudPhaseScheduler.__new__(LwdCloudPhaseScheduler)
    scheduler.vllm_config = SimpleNamespace(lwd_config=SimpleNamespace(instance_id=1))
    scheduler._lwd_next_range = lambda: notify
    scheduler._lwd_schedule_for_visible_reqs = lambda *a, **kw: SimpleNamespace()
    batch = scheduler._schedule_pure_prefill().lwd_batch
    restored = pickle.loads(pickle.dumps(batch))
    assert restored.connection_key == (1, 1, 1)
    assert restored.seqno == 7


def test_down_batch_preserves_connection_and_rejects_wrong_destination():
    notify = LwdC2eNotify(
        hidden_num_elements=32, top_id_ths=[[1]], num_accepted_tokens=[1],
        req_ids=["client-id"], down_seqno=7, edge_id=1, dp_idx=1,
    )
    batch = lwd_build_unembed_batch(notify, (1, 1, 1)).lwd_batch
    assert batch.connection_key == (1, 1, 1)
    assert batch.seqno == 7
    with pytest.raises(ValueError, match="does not match"):
        lwd_build_unembed_batch(notify, (0, 0, 1))


@pytest.mark.parametrize("identity,edge,cloud,dp,accepted", [
    (b"edge-0-0", 0, 0, 0, True),
    (b"wrong", 0, 0, 0, False),
    (b"edge-1-0", 1, 0, 0, False),
    (b"edge-0-1", 0, 0, 1, False),
    (b"edge-0-0", 0, 1, 0, False),
])
def test_registration_matches_the_configured_connection(
    identity, edge, cloud, dp, accepted,
):
    from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
        LwdCloudEngineCore,
    )

    config = SimpleNamespace(
        my_links=((0, 0, 0),), instance_id=0,
        edge_npu_count=1, cloud_npu_count=8, topology_digest="same-file",
    )
    target = SimpleNamespace(
        _lwd_config=config, _lwd_peers={}, _lwd_peer_ids={}, _lwd_io=Mock(),
    )
    msg = LwdRegisterNotify(
        edge, cloud, dp, wire_version=LWD_WIRE_VERSION,
        edge_npu_count=1, cloud_npu_count=8, topology_digest="same-file",
    )
    LwdCloudEngineCore._lwd_handle_register(target, identity, msg)
    assert bool(target._lwd_peers) is accepted
    assert target._lwd_io.send.called is accepted


@pytest.mark.parametrize("cloud,dp,valid", [(0, 0, True), (1, 0, False), (0, 1, False)])
def test_ack_must_match_the_dealer_connection(cloud, dp, valid):
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
        LwdEdgeEngineCore,
    )

    event = threading.Event()
    target = SimpleNamespace(_lwd_ack={(0, 0, 0): event}, _lwd_ack_error=None)
    msg = LwdRegisterAckNotify(cloud, dp, wire_version=LWD_WIRE_VERSION)
    LwdEdgeEngineCore._lwd_on_cloud_msg(target, (0, 0, 0), None, msg)
    assert event.is_set()  # Wake the constructor on either success or failure.
    assert (target._lwd_ack_error is None) is valid


def test_c2e_keeps_down_packet_identity_seqno_size_and_original_request_id():
    from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
        LwdCloudEngineCore,
    )
    from vllm.v1.outputs import LwdC2eMeta

    target = SimpleNamespace(
        _lwd_config=SimpleNamespace(my_links=((0, 0, 0),)),
        _lwd_peer_ids={(0, 0): b"edge-0-0"},
        _lwd_io=SimpleNamespace(send=Mock(return_value=True), closed=False),
        _lwd_c2e_split=LwdCloudEngineCore._lwd_c2e_split,
        _lwd_c2e_build=LwdCloudEngineCore._lwd_c2e_build,
    )
    meta = LwdC2eMeta(
        hidden_num_elements=32, top_id_ths=[[2]], num_accepted_tokens=[1],
        req_ids=["0#0#client#id"], down_seqno=7, connection_key=(0, 0, 0),
    )
    LwdCloudEngineCore._lwd_publish_c2e(target, meta, [-1])
    identity, notify = target._lwd_io.send.call_args.args
    assert identity == b"edge-0-0"
    batch = lwd_build_unembed_batch(notify, (0, 0, 0)).lwd_batch
    assert batch.connection_key == meta.connection_key
    assert batch.seqno == 7
    assert batch.batch_meta.req_ids == ["client#id"]
    assert batch.batch_meta.recv_num_elements == 32
    meta.connection_key = (0, 1, 0)
    with pytest.raises(ValueError, match="matching DOWN packet"):
        LwdCloudEngineCore._lwd_publish_c2e(target, meta, [-1])
