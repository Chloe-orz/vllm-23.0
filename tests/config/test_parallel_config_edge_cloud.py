# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import ParallelConfig


@pytest.mark.parametrize(
    ("is_edge_node", "node_rank"), [(True, 0), (False, 1)]
)
def test_equal_sized_edge_cloud_topology(
    is_edge_node: bool, node_rank: int
) -> None:
    config = ParallelConfig(
        enable_edge_cloud=True,
        edge_npu_count=2,
        cloud_npu_count=2,
        is_edge_node=is_edge_node,
        nnodes=2,
        node_rank=node_rank,
    )

    assert config.world_size == 4
    assert config.pipeline_parallel_size == 2
    assert config.tensor_parallel_size == 2
    assert config.local_world_size == 2
    assert not config.is_shared_model_edge


def test_edge_stage_cannot_exceed_cloud_stage() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        ParallelConfig(
            enable_edge_cloud=True,
            edge_npu_count=3,
            cloud_npu_count=2,
            is_edge_node=True,
        )


def test_equal_sized_edge_cloud_topology_is_normalized_per_dp() -> None:
    config = ParallelConfig(
        enable_edge_cloud=True,
        edge_npu_count=4,
        cloud_npu_count=4,
        data_parallel_size=2,
        is_edge_node=True,
    )

    assert config.edge_npu_count == 2
    assert config.cloud_npu_count == 2
    assert config.world_size == 4
    assert config.tensor_parallel_size == 2
    assert config.local_world_size == 2
    assert not config.is_shared_model_edge
