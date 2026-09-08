"""Lwd 交互预算 CI 检查(§7.4/§8.4/§9.12):白名单/getattr/env。

预算唯一事实源是 docs/refactor/prefill_only_migration.md 的台账;
本脚本只做静态 grep 组合,超预算即非零退出,可挂 pre-commit/CI。
本目录仅控制面(§9.12),torch.distributed 检查已随数据面裁剪移除。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# 台账例外:文件名 -> 允许的额外 import 片段(正则);布局依 §10.3/§10.7
_LWD_FILE_EXCEPTIONS = {
    "lwd_notify.py": (r"from vllm\.v1\.engine import",),
    "lwd_edge_scheduler.py": (r"from vllm\.v1\.request import",),
    "lwd_cloud_phase_scheduler.py": (r"from vllm\.v1\.request import",),
    "lwd_edge_assemble.py": (r"from vllm\.v1\.core\.kv_cache_utils import",),
    "lwd_cloud_engine.py": (
        r"from vllm\.v1\.core\.kv_cache_utils import",
        r"from vllm\.v1\.request import",
        r"from vllm\.v1\.outputs import",
        r"from vllm\.sampling_params import",
        r"from vllm\.v1\.engine import",
        r"from vllm\.v1\.engine\.core import",
    ),
}

# 内核文件违禁 import(§7.2/§9.12);L3 装配文件另享 _LWD_FILE_EXCEPTIONS
_LWD_BANNED_IMPORTS = (
    r"import vllm\.envs|from vllm\.envs",
    r"from vllm\.v1\.request import",
    r"from vllm\.v1\.outputs import",
    r"from vllm\.v1\.engine import|from vllm\.v1\.engine\.core import",
    r"from vllm\.v1\.executor import",
    r"from vllm\.tracing import",
    r"torch\.distributed",
    r"pd_separation|vllm_ascend|edge_cloud_comm",
)

_LWD_SCHED_ONLY_IMPORT = re.compile(
    r"from vllm\.v1\.core\.sched\.(output|async_scheduler|request_queue) import"
)
_LWD_SCHEDULER_FILES = {"lwd_edge_scheduler.py", "lwd_cloud_phase_scheduler.py"}
_LWD_ENV_PATTERN = re.compile(r"os\.environ|environ\[|os\.getenv")
_LWD_GETATTR_PATTERN = re.compile(r"getattr\s*\(")
_LWD_GETATTR_BUDGET = 10


def _lwd_iter_files(lwd_root: Path):
    for path in sorted(lwd_root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        yield path


def check_import_whitelist(lwd_root: Path) -> list[str]:
    """内核违禁 import = 0(vllm.envs/v1.request/v1.outputs/engine.core/
    torch.distributed 等,例外见台账)。"""
    violations: list[str] = []
    for path in _lwd_iter_files(lwd_root):
        source = path.read_text(encoding="utf-8")
        allowed = _LWD_FILE_EXCEPTIONS.get(path.name, ())
        for banned in _LWD_BANNED_IMPORTS:
            for match in re.finditer(banned, source):
                snippet = source[match.start() :].splitlines()[0]
                # 例外按前缀匹配(模式已含完整 from..import 边界,含 noqa 后缀不误报)
                if any(re.match(allow, snippet.strip()) for allow in allowed):
                    continue
                violations.append(f"{path.name}: {snippet.strip()}")
        if path.name not in _LWD_SCHEDULER_FILES:
            for match in _LWD_SCHED_ONLY_IMPORT.finditer(source):
                snippet = source[match.start() :].splitlines()[0]
                violations.append(
                    f"{path.name}: scheduler-only import -> {snippet.strip()}"
                )
    return violations


_LWD_GETATTR_TOLERANT_FILES = {
    "lwd_edge_assemble.py",
    "lwd_cloud_engine.py",
}


def check_getattr_budget(lwd_root: Path) -> list[str]:
    """内核 getattr = 0;装配层/照搬移植文件 <=10 且逐处注释。"""
    violations: list[str] = []
    for path in _lwd_iter_files(lwd_root):
        count = len(_LWD_GETATTR_PATTERN.findall(path.read_text(encoding="utf-8")))
        if count == 0:
            continue
        if path.name in _LWD_GETATTR_TOLERANT_FILES:
            if count > _LWD_GETATTR_BUDGET:
                violations.append(
                    f"{path.name}: getattr {count} > budget {_LWD_GETATTR_BUDGET}"
                )
        else:
            violations.append(f"{path.name}: 内核 getattr = {count}(预算 0)")
    return violations


def check_env_parsing(lwd_root: Path) -> list[str]:
    """内核 os.environ 读取 = 0(唯一入口 LwdConfig.from_env_and_config)。"""
    violations: list[str] = []
    for path in _lwd_iter_files(lwd_root):
        if path.name == "lwd_edge_assemble.py":
            continue  # LwdConfig 定义处:唯一 env 入口(§7.3-C3/§10.3)
        if _LWD_ENV_PATTERN.search(path.read_text(encoding="utf-8")):
            violations.append(
                f"{path.name}: env 解析越权(仅 lwd_edge_assemble.py 允许)"
            )
    return violations


def main() -> int:
    """汇总各检查;有违规打印清单并返回 1。"""
    parser = argparse.ArgumentParser(description="Lwd interaction budget checker")
    parser.add_argument(
        "--lwd-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "vllm/v1/lwd_control",
    )
    lwd_root = parser.parse_args().lwd_root
    violations = (
        check_import_whitelist(lwd_root)
        + check_getattr_budget(lwd_root)
        + check_env_parsing(lwd_root)
    )
    for violation in violations:
        print(f"[Lwd budget] {violation}")
    if violations:
        print(f"[Lwd budget] FAILED: {len(violations)} violation(s)")
        return 1
    print("[Lwd budget] OK: import/getattr/env budgets all within ledger")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
