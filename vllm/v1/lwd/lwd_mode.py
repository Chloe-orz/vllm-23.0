"""Lwd prefill-only 模式判定:全仓唯一实现(设计文档 §2.3 / W9,现状 5 处收敛为 1)。"""

from __future__ import annotations


def is_lwd_prefill_only(vllm_config) -> bool:
    """仅读 parallel_config.enable_edge_cloud 与 additional_config 的 prefill_only 段。"""
    ...


def lwd_mark_role(*, edge: bool) -> None:
    """装配期标记本进程角色;此后 in-tree 守卫分支只查进程标志,不再解析配置。"""
    ...


def lwd_is_edge_active() -> bool:
    """本进程已完成边侧装配。"""
    ...


def lwd_is_cloud_active() -> bool:
    """本进程已完成云侧装配。"""
    ...


def lwd_active() -> bool:
    """任一角色已装配;core.py 等上游守卫分支的唯一查询入口。"""
    ...
