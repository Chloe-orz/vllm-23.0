"""Lwd 交互预算 CI 检查(§7.4/§8.4):白名单/getattr/env/torch.distributed 文件数。

预算唯一事实源是 docs/refactor/prefill_only_migration.md 的台账;
本脚本只做静态 grep 组合,超预算即非零退出,可挂 pre-commit/CI。
"""

from __future__ import annotations

import argparse
from pathlib import Path

# 内核文件(torch.distributed 允许名单,§8.4/§9)
LWD_DIST_FILES = ("lwd_edge_embed.py", "lwd_cloud_embeds.py")


def check_import_whitelist(lwd_root: Path) -> "list[str]":
    """内核违禁 import = 0(vllm.envs/v1.request/v1.outputs/engine.core 等,例外见台账)。"""
    ...


def check_getattr_budget(lwd_root: Path) -> "list[str]":
    """内核 getattr = 0;适配层(assemble/launch/adapter)<=10 且逐处注释。"""
    ...


def check_env_parsing(lwd_root: Path) -> "list[str]":
    """内核 os.environ 读取 = 0(唯一入口 LwdConfig.from_env_and_config)。"""
    ...


def check_torch_dist_files(lwd_root: Path) -> "list[str]":
    """torch.distributed 使用仅限 2 个数据面文件。"""
    ...


def main() -> int:
    """汇总各检查;有违规打印清单并返回 1。"""
    ...


if __name__ == "__main__":
    raise SystemExit(main())
