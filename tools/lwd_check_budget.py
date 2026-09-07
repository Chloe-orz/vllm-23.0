"""Lwd 交互预算 CI 检查(§7.4/§8.4/§9.12):白名单/getattr/env。

预算唯一事实源是 docs/refactor/prefill_only_migration.md 的台账;
本脚本只做静态 grep 组合,超预算即非零退出,可挂 pre-commit/CI。
本目录仅控制面(§9.12),torch.distributed 检查已随数据面裁剪移除。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def check_import_whitelist(lwd_root: Path) -> "list[str]":
    """内核违禁 import = 0(vllm.envs/v1.request/v1.outputs/engine.core/
    torch.distributed 等,例外见台账)。"""
    ...


def check_getattr_budget(lwd_root: Path) -> "list[str]":
    """内核 getattr = 0;装配层(assemble/launch)<=10 且逐处注释。"""
    ...


def check_env_parsing(lwd_root: Path) -> "list[str]":
    """内核 os.environ 读取 = 0(唯一入口 LwdConfig.from_env_and_config)。"""
    ...


def main() -> int:
    """汇总各检查;有违规打印清单并返回 1。"""
    ...


if __name__ == "__main__":
    raise SystemExit(main())
