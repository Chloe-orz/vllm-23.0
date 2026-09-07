"""边侧 L3 装配:EngineCore.__init__ 守卫调用的单点开关(§8.2/§9.8)。

单向语义:无云结果/水位 drain,步进逻辑全部收编进 LwdEdgeCore.step_with_batch_queue;
本文件只负责装配与生命周期,原 6 个 monkey-patch 由 core.py in-tree 守卫分支替代。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.lwd.lwd_config import LwdConfig
    from vllm.v1.lwd.lwd_edge_channel import LwdEdgeControlPublisher


def lwd_edge_try_assemble(engine_core) -> bool:
    """装配点(core.py __init__ 尾守卫调用):非 PO 立即返回 False,零副作用。

    PO 时:建 PRE_OUT 通道,scheduler_cls 以 partial(LwdEdgeScheduler,
    publisher=...)注入(调度器即控制面出口,§9.12),建 LwdEdgeCore 并赋给
    engine_core.step_wrapper(core.py step 守卫的唯一委托对象,§9.8)。
    """
    ...


def _lwd_edge_build_planes(config: LwdConfig) -> LwdEdgeControlPublisher:
    """建控制面发布通道(单向,仅 PRE_OUT;无结果面,§9.1)。"""
    ...


def lwd_edge_shutdown(engine_core) -> None:
    """通道关停(core.py shutdown 守卫分支调用;装配层掌生命周期)。"""
    ...
