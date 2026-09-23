# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""C2e rid 前缀切组纯函数测试(无设备/分布式依赖;需完整 vllm import 链)。"""

from types import SimpleNamespace

from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
    LwdCloudEngineCore,
    _lwd_rid_key,
    _lwd_rid_split,
)


def test_rid_prefix_roundtrip_keeps_client_ids_verbatim():
    # 客户端 request_id 自带 '#' 也不影响(maxsplit=2,前缀固定两段)
    rid = "client-req-#42#with#hashes"
    assert _lwd_rid_key(3, 1, rid) == "3#1#" + rid
    assert _lwd_rid_split(_lwd_rid_key(3, 1, rid)) == (3, 1, rid)


def test_c2e_split_groups_rows_by_edge_and_restores_ids():
    meta = SimpleNamespace(
        hidden_num_elements=128,
        top_id_ths=[[1, 2], [3], [4], [5, 6]],
        num_accepted_tokens=[2, 1, 2, 2],
        req_ids=["0#0#a", "1#0#b", "0#0#c", "1#0#d"],
        down_seqno=7,
    )
    groups = LwdCloudEngineCore._lwd_c2e_split(meta)
    assert set(groups) == {(0, 0), (1, 0)}
    edge0 = LwdCloudEngineCore._lwd_c2e_build(
        meta, [-1, -1, -1, -1], groups[(0, 0)], 0, 0
    )
    assert edge0.req_ids == ["a", "c"]  # 还原边侧原始 id
    assert edge0.top_id_ths == [[1, 2], [4]]
    assert edge0.num_accepted_tokens == [2, 2]
    assert edge0.edge_id == 0 and edge0.dp_idx == 0
    assert edge0.down_seqno == 7
    edge1 = LwdCloudEngineCore._lwd_c2e_build(
        meta, [-1, -1, 0, -1], groups[(1, 0)], 1, 0
    )
    # finish_reasons 按行下标对齐切组,不串位
    assert edge1.req_ids == ["b", "d"]
    assert edge1.finish_reasons == [-1, -1]


def test_c2e_split_single_edge_degenerates_to_one_group():
    meta = SimpleNamespace(
        hidden_num_elements=8,
        top_id_ths=[[1], [2]],
        num_accepted_tokens=[1, 1],
        req_ids=["0#0#x", "0#0#y"],
        down_seqno=1,
    )
    groups = LwdCloudEngineCore._lwd_c2e_split(meta)
    assert list(groups) == [(0, 0)]
    notify = LwdCloudEngineCore._lwd_c2e_build(
        meta, [0, -1], groups[(0, 0)], 0, 0
    )
    assert notify.req_ids == ["x", "y"]
    assert notify.finish_reasons == [0, -1]
