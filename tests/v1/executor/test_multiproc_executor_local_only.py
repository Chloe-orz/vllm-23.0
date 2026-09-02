# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc


class _RecordingAggregator:
    def __init__(self) -> None:
        self.outputs = None
        self.output_rank = None

    def aggregate(self, outputs, output_rank=0):
        self.outputs = outputs
        self.output_rank = output_rank
        return outputs[output_rank]


def test_local_only_rpc_with_kv_aggregator_reads_local_responses_only():
    executor = object.__new__(MultiprocExecutor)
    executor.rpc_broadcast_mq = MagicMock()
    executor.is_failed = False
    executor.futures_queue = deque()

    success = WorkerProc.ResponseStatus.SUCCESS
    local_response_mqs = [MagicMock(), MagicMock()]
    local_response_mqs[0].dequeue.return_value = (success, "edge-rank-0")
    local_response_mqs[1].dequeue.return_value = (success, "edge-rank-1")
    executor.workers = [
        SimpleNamespace(rank=rank, worker_response_mq=response_mq)
        for rank, response_mq in enumerate(local_response_mqs)
    ]

    remote_response_mqs = [MagicMock(), MagicMock()]
    for response_mq in remote_response_mqs:
        response_mq.dequeue.side_effect = AssertionError(
            "local-only RPC must not wait for a remote response"
        )

    executor.response_mqs = local_response_mqs + remote_response_mqs
    aggregator = _RecordingAggregator()

    result = executor.collective_rpc(
        "execute_model",
        unique_reply_rank=0,
        kv_output_aggregator=aggregator,
        local_only=True,
    )

    assert result == "edge-rank-0"
    assert aggregator.outputs == ["edge-rank-0", "edge-rank-1"]
    assert aggregator.output_rank == 0
    executor.rpc_broadcast_mq.enqueue.assert_called_once_with(
        ("execute_model", (), {}, None), local_only=True
    )
    for response_mq in remote_response_mqs:
        response_mq.dequeue.assert_not_called()
