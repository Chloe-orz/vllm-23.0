#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""lwd_acceptance_probe: MTP 接受率分析(纯 Python,Windows 可用)。

用法:
  python lwd_acceptance_probe.py 边日志 [云日志...]

自动识别两种日志格式:
  旧(rank-replay 版): [Lwd][edge-worker] UNEMBED seqno=N reqs=R accepted=A ...
  新(token_id 版):    [Lwd][sched] edge deliver-unembed reqs=N tokens=[a,b,..]
                      [Lwd][cloud-ctrl] publish c2e(token-id): reqs=N tokens=[..]

输出:
  ① 总览:步数/每请求每步平均接受数/最大值
  ② 每请求接受数直方图(1/2/3/4 各占比)——全 1 = draft 从未被接受
  ③ 按并发分组:批内请求数 1 / 2-4 / 5-16 / >16 各组的平均接受数
     ——单请求健康而大并发组崩 = 多请求扰动(draft 首趟兜底污染方向)
  ④ 前半 vs 后半:接受率是否随时间劣化
  ⑤ 判定
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import sys

# 时间戳宽容提取:行内任意位置,两种日期形态,毫秒可缺省
RE_TS_FULL = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:[,.](\d{1,3}))?")
RE_TS_SHORT = re.compile(
    r"(?<!\d)(\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:[,.](\d{1,3}))?")
# 旧版:批级汇总(accepted 为全批之和)
RE_UNEMBED = re.compile(
    r"\[Lwd\]\[edge-worker\] UNEMBED seqno=\d+ reqs=(\d+) "
    r"accepted=(\d+)")
# 新版:逐请求 token 数列表
RE_TOKENS = re.compile(
    r"(?:deliver-unembed|publish c2e\(token-id\)):? reqs=(\d+) "
    r"tokens=\[([\d, ]*)\]")

BUCKETS = [(1, 1, "reqs=1"), (2, 4, "reqs=2-4"),
           (5, 16, "reqs=5-16"), (17, 10 ** 9, "reqs>16")]


def _sec(ts: str, ms: str | None) -> float:
    # 首段 4 位=完整日期;2 位=MM-DD 短格式(年份按 2026 补)
    if len(ts.split("-")[0]) != 4:
        ts = "2026-" + ts
    return _dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp() \
        + (int(ms) / 1000.0 if ms else 0.0)


def _mean(v):
    return sum(v) / len(v) if v else float("nan")


def scan(paths: list[str]):
    """返回 (per_request[(t, reqs, tokens...)], batch[(t, reqs, accepted)],
    raw_samples[原始锚点行样例])."""
    per_req: list[tuple[float, int, list[int]]] = []
    batch: list[tuple[float, int, int]] = []
    raw: list[str] = []
    for path in paths:
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"[probe] 解析 {path} ({size / 1e6:.1f} MB)...",
              file=sys.stderr, flush=True)
        n = 0
        # utf-8-sig:吞掉 Windows 拷贝可能带入的 BOM
        with open(path, encoding="utf-16" if open(path, "rb").read(2) in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig", errors="replace") as fh:
            for line in fh:
                n += 1
                if n % 5_000_000 == 0:
                    print(f"[probe]   已扫描 {n:,} 行", file=sys.stderr,
                          flush=True)
                if "[Lwd]" not in line and "UNEMBED" not in line \
                        and "tokens=[" not in line:
                    continue
                mm = RE_TOKENS.search(line)
                if mm and mm.group(2).strip():
                    toks = [int(x) for x in mm.group(2).split(",")
                            if x.strip()]
                    per_req.append((_t(line, n), len(toks), toks))
                    continue
                mm = RE_UNEMBED.search(line)
                if mm:
                    batch.append((_t(line, n), int(mm.group(1)),
                                  int(mm.group(2))))
                    continue
                if len(raw) < 5 and ("UNEMBED" in line
                                     or "deliver-unembed" in line):
                    raw.append(line.rstrip()[:200])
    per_req.sort(key=lambda x: x[0])
    batch.sort(key=lambda x: x[0])
    return per_req, batch, raw


def _t(line: str, idx: int) -> float:
    """行时间戳;解析不出时用行号作伪时间(仅保序,不影响分组/直方图)。"""
    m = RE_TS_FULL.search(line)
    if m:
        return _sec(m.group(1), m.group(2))
    m = RE_TS_SHORT.search(line)
    if m:
        return _sec(m.group(1), m.group(2))
    return float(idx)


def report(per_req, batch, raw, paths) -> None:
    print("=" * 66, flush=True)
    rows: list[tuple[float, int]] = []   # (t, reqs, 每请求值...) 展平用
    flat: list[int] = []
    if per_req:
        for t, reqs, toks in per_req:
            for x in toks:
                rows.append((t, reqs, x))
                flat.append(x)
        src = "逐请求明细(token_id 版日志)"
    elif batch:
        for t, reqs, acc in batch:
            per = acc / reqs if reqs else 0.0
            rows.append((t, reqs, per))
            flat = None
        src = "批级汇总(UNEMBED 行,accepted/reqs 近似)"
    else:
        print("!! 未找到任何接受率锚点(UNEMBED / deliver-unembed / publish)",
              flush=True)
        if raw:
            print("发现疑似锚点但正则不匹配,前几行原文如下(用于修正正则):",
                  flush=True)
            for s in raw:
                print(f"   | {s!r}", flush=True)
        else:
            print("日志中连 accept/unembed/token 字样的行都没有——先确认"
                  "传的是边/云引擎日志(不是前端/客户端日志)。", flush=True)
            print("自动取证:文件中含这些字样的前 5 行原文:", flush=True)
            import re as _re
            rx = _re.compile(r"accept|unembed|token", _re.I)
            shown = 0
            for path in paths:
                with open(path, encoding="utf-16" if open(path, "rb").read(2) in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig", errors="replace") as fh:
                    for line in fh:
                        if rx.search(line):
                            print(f"   | {line.rstrip()[:200]!r}", flush=True)
                            shown += 1
                            if shown >= 5:
                                break
                if shown >= 5:
                    break
            if shown == 0:
                print("   (整份日志确实没有任何相关行)", flush=True)
        return
    print(f"数据源: {src}", flush=True)

    print("① 总览")
    if flat is not None:
        print(f"   步数={len(per_req)}  每请求接受数: 均值={_mean(flat):.2f}"
              f"  最大={max(flat)}", flush=True)
        print("② 每请求接受数直方图")
        hist: dict[int, int] = {}
        for x in flat:
            hist[x] = hist.get(x, 0) + 1
        for k in sorted(hist):
            print(f"   {k} token: {hist[k]:6d}  {hist[k] * 100 / len(flat):5.1f}%",
                  flush=True)
    else:
        ratios = [r for _, _, r in rows]
        print(f"   步数={len(batch)}  每请求平均接受数: 均值={_mean(ratios):.2f}"
              f"  最大={max(ratios):.2f}", flush=True)
        print("   (批级日志无逐请求明细,直方图以比值代替)", flush=True)
        h: dict[int, int] = {}
        for r in ratios:
            h[round(r)] = h.get(round(r), 0) + 1
        for k in sorted(h):
            print(f"   ≈{k} token: {h[k]:6d}  {h[k] * 100 / len(ratios):5.1f}%",
                  flush=True)

    print("③ 按并发分组(批内请求数)——单请求健康而大并发组崩=多请求扰动")
    for lo, hi, label in BUCKETS:
        g = [r for _, q, r in rows if lo <= q <= hi]
        if g:
            print(f"   {label:<10} 步数={len(g):6d}  平均接受={_mean(g):.2f}",
                  flush=True)

    print("④ 前半 vs 后半")
    mid = rows[len(rows) // 2][0] if rows else 0
    h1 = [r for t, _, r in rows if t < mid]
    h2 = [r for t, _, r in rows if t >= mid]
    if h1 and h2:
        print(f"   前半={_mean(h1):.2f}  后半={_mean(h2):.2f}"
              f"{'  ← 随时间劣化' if _mean(h2) < _mean(h1) * 0.8 else ''}",
              flush=True)

    print("⑤ 判定")
    vs = []
    if flat is not None:
        mean = _mean(flat)
        mx = max(flat)
        acc = mean / mx if mx else float("nan")
        if acc < 0.4:
            vs.append(f"整体接受率≈{acc * 100:.0f}% 偏低→draft 大量被拒")
        solo = [r for _, q, r in rows if q == 1]
        crowd = [r for _, q, r in rows if q >= 5]
        if solo and crowd and _mean(solo) - _mean(crowd) > 0.8:
            vs.append(f"单并发({_mean(solo):.2f}) ≫ 大并发({_mean(crowd):.2f})"
                      f"→【多请求扰动】指向 draft 首趟兜底污染")
        if vs and h1 and h2 and _mean(h2) < _mean(h1) * 0.8:
            vs.append("后半场显著劣化→随积压/扰动发展的污染形态")
        if not vs:
            vs.append(f"接受率≈{acc * 100:.0f}%,形态健康→"
                      f"嫌疑转回步耗时/调度(用 lwd_bench_gap_probe)")
    else:
        vs.append("批级数据仅作初判;要 ③④ 的精细结论需 token_id 版日志")
    for v in vs:
        print(f"   • {v}", flush=True)
    print("=" * 66, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="MTP 接受率分析(纯 Python,Windows 可用)")
    ap.add_argument("logs", nargs="+", help="边日志/云日志(可多个)")
    args = ap.parse_args()
    per_req, batch, raw = scan(args.logs)
    report(per_req, batch, raw, args.logs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
