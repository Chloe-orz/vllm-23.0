#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""lwd_mtp_dbg_probe: MTP 草稿污染机制判决工具(配合 VLLM_ASCEND_LWD_MTP_DBG=1)。

前置:云侧启动环境设 VLLM_ASCEND_LWD_MTP_DBG=1 打出逐请求接受数
     [Lwd][mtp-dbg] verify req=R accepted=N(每 decode 步每请求一条,
     model_runner_v1 的 _bookkeeping_sync,零新增同步)。与相位日志
     [Lwd][sched] cloud step= phase= 同文件,本工具单文件即可判决。

机制命题(草稿"满窗调度但全拒"的三种病因):
  P1 生成侧污染:紧跟 prefill 的 decode 步里生成的草稿本身就是错的
  P2 消费侧老化:草稿没错,但隔了 prefill 步再消费时失效
  P3 自持续:一旦污染,drafter 状态持续坏到交替结束

分桶方法:每个 (请求, decode步) 验证事件按两个维度归类——
  时滞: fresh=草稿生成于紧邻的上一个 decode 步;
        aged=中间隔了 >=1 个 prefill 步才被消费;
        first=该请求首个 decode 步(草稿来自 prefill 完成步)
  生成环境: gen-after-prefill / gen-after-decode(生成那个 decode 步
        的上一步相位)
判决表:
  fresh 且 gen-after-prefill 低、gen-after-decode 高      → P1
  fresh 两组都高、aged 两组都低                            → P2
  交替期所有桶都低                                          → P3

用法: python lwd_mtp_dbg_probe.py 云日志
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import sys

RE_TS_FULL = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:[,.](\d{1,3}))?")
RE_TS_SHORT = re.compile(
    r"(?<!\d)(\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:[,.](\d{1,3}))?")
RE_PHASE = re.compile(
    r"\[Lwd\]\[sched\] cloud step=\d+ phase=(\w+)")
RE_VERIFY = re.compile(
    r"\[Lwd\]\[mtp-dbg\] verify req=(\S+) accepted=(\d+)")


def _t(line: str, idx: int) -> float:
    m = RE_TS_FULL.search(line) or RE_TS_SHORT.search(line)
    if not m:
        return float(idx)
    ts, ms = m.group(1), m.group(2)
    if len(ts.split("-")[0]) != 4:
        ts = "2026-" + ts
    return _dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp() \
        + (int(ms) / 1000.0 if ms else 0.0)


def _mean(v):
    return sum(v) / len(v) if v else float("nan")


def scan(path: str):
    """单遍扫描 -> [(t, kind, payload)],kind in {phase, verify}."""
    events: list[tuple[float, str, object]] = []
    size = os.path.getsize(path) if os.path.exists(path) else 0
    print(f"[probe] 解析 {path} ({size / 1e6:.1f} MB)...",
          file=sys.stderr, flush=True)
    enc = ("utf-16" if open(path, "rb").read(2) in (b"\xff\xfe", b"\xfe\xff")
           else "utf-8-sig")
    n = 0
    with open(path, encoding=enc, errors="replace") as fh:
        for line in fh:
            n += 1
            if "[Lwd]" not in line:
                continue
            m = RE_PHASE.search(line)
            if m:
                events.append((_t(line, n), "phase", m.group(1)))
                continue
            m = RE_VERIFY.search(line)
            if m:
                events.append((_t(line, n), "verify",
                               (m.group(1), int(m.group(2)))))
    events.sort(key=lambda e: e[0])
    return events


def classify(events):
    """按步重建并分桶。返回 buckets 与 phase 计数。"""
    # 步切分:phase 事件定步界;verify 归属最近一个 phase 步
    steps: list[dict] = []           # [{phase, verify: {req: accepted}}]
    for t, kind, payload in events:
        if kind == "phase":
            steps.append({"phase": payload, "verify": {}})
        elif steps:
            req, acc = payload
            # 多 rank 重复:同步内同 req 取首次
            steps[-1]["verify"].setdefault(req, acc)

    # 每个 decode 步的序号环境:上一步/上上个 decode 步的相位
    buckets: dict[str, list[int]] = {}
    last_seen: dict[str, int] = {}   # req -> 上次出现的 decode 步下标
    phase_count: dict[str, int] = {}
    for i, st in enumerate(steps):
        phase_count[st["phase"]] = phase_count.get(st["phase"], 0) + 1
        if st["phase"] != "DECODE":
            continue
        for req, acc in st["verify"].items():
            j = last_seen.get(req)
            if j is None:
                key = "first(from-prefill)"
            else:
                # 生成环境:草稿生成步(j,该请求上个 decode 步)的前相位
                gen_prev = steps[j - 1]["phase"] if j > 0 else "NONE"
                gen_ctx = ("gen-after-prefill" if gen_prev == "PREFILL"
                           else "gen-after-decode")
                # 时滞:生成步(j) 与消费步(i) 之间是否隔了 prefill
                adjacent = all(steps[k]["phase"] == "DECODE"
                               for k in range(j + 1, i))
                key = (f"fresh/{gen_ctx}" if adjacent
                       else f"aged/{gen_ctx}")
            buckets.setdefault(key, []).append(acc)
            last_seen[req] = i
    return buckets, phase_count, len(steps)


def report(buckets, phase_count, n_steps) -> None:
    print("=" * 66, flush=True)
    print(f"步数={n_steps}  构成={phase_count}", flush=True)
    if not buckets:
        print("!! 无 [Lwd][mtp-dbg] verify 行——确认云侧已设 "
              "VLLM_ASCEND_LWD_MTP_DBG=1 且跑了 decode", flush=True)
        return
    print("分桶(每请求每步接受数均值)——判决表:")
    print("   fresh=草稿紧邻生成即消费;aged=中间隔了 prefill 才消费")
    print("   gen-after-prefill=生成步紧跟 prefill")
    order = ["first(from-prefill)",
             "fresh/gen-after-decode", "fresh/gen-after-prefill",
             "aged/gen-after-decode", "aged/gen-after-prefill"]
    for k in order:
        g = buckets.get(k)
        if g:
            print(f"   {k:<28} n={len(g):6d}  均值={_mean(g):.2f}"
                  f"  p50={sorted(g)[len(g) // 2]}", flush=True)
    for k in sorted(buckets):
        if k not in order:
            g = buckets[k]
            print(f"   {k:<28} n={len(g):6d}  均值={_mean(g):.2f}",
                  flush=True)
    print("判决", flush=True)
    vs = []

    def m(key):
        return _mean(buckets[key]) if buckets.get(key) else float("nan")

    fp, fd = m("fresh/gen-after-prefill"), m("fresh/gen-after-decode")
    ad, ap = m("aged/gen-after-decode"), m("aged/gen-after-prefill")
    if fd == fd and fp == fp and fd - fp > 1.0:
        vs.append(f"fresh:after-decode {fd:.2f} ≫ after-prefill {fp:.2f}"
                  f"→【P1 生成侧污染】prefill 紧后的 draft 生成被污染")
    if fd == fd and ad == ad and fd - ad > 1.0:
        vs.append(f"fresh {fd:.2f} ≫ aged {ad:.2f}"
                  f"→【P2 消费侧老化】草稿隔 prefill 步消费即失效")
    lows = [x for x in (fp, fd, ad, ap) if x == x]
    if lows and max(lows) < 1.6:
        vs.append(f"全桶低(最大 {max(lows):.2f})→【P3 自持续】交替期"
                  f"drafter 状态持续损坏")
    if not vs:
        vs.append("无显著分桶差异——贴输出发我人工复核")
    for v in vs:
        print(f"   • {v}", flush=True)
    print("=" * 66, flush=True)


def selftest() -> int:
    import tempfile
    from pathlib import Path
    lines = []
    t = 0
    def st(ms):
        base = _dt.datetime(2026, 9, 18, 12, 0, 0)
        return (base + _dt.timedelta(milliseconds=ms)).strftime(
            "%Y-%m-%d %H:%M:%S,%f")[:-3]
    # 构造:P D(a4) D(a4) P D(a:1 aged? no—上步是prefill,gen ctx...) ...
    # 简化时序,覆盖五桶:
    seq = [
        ("PREFILL", None),
        ("DECODE", {"a": 4, "b": 4}),      # first(草稿来自prefill完成步)
        ("DECODE", {"a": 4, "b": 4}),      # fresh/gen-after-prefill(生成于步1,前步P)
        ("DECODE", {"a": 3, "b": 3}),      # fresh/gen-after-decode(生成于步2,前步D)
        ("PREFILL", None),
        ("DECODE", {"a": 1, "b": 1}),      # aged/gen-after-decode(生成于步3,隔P消费)
        ("DECODE", {"a": 1, "b": 1}),      # fresh/gen-after-prefill(生成于步5,前步P)
        ("PREFILL", None),
        ("DECODE", {"c": 4}),              # first c
        ("DECODE", {"c": 4}),              # fresh/gen-after-prefill
    ]
    for ph, ver in seq:
        lines.append(f"{st(t)} [Lwd][sched] cloud step={len(lines)} "
                     f"phase={ph} seqno= reqs=[x] tokens=1 pending=0 "
                     f"decode_ready=1")
        t += 100
        if ver:
            for r, a in ver.items():
                lines.append(f"{st(t)} [Lwd][mtp-dbg] verify req={r} "
                             f"accepted={a}")
                t += 1
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "c.log"
        f.write_text("\n".join(lines) + "\n")
        events = scan(str(f))
        buckets, pc, ns = classify(events)
        assert ns == 10, ns
        assert buckets["first(from-prefill)"] == [4, 4, 4], buckets
        assert buckets["fresh/gen-after-prefill"] == [4, 4, 1, 1, 4], buckets
        assert buckets["fresh/gen-after-decode"] == [3, 3], buckets
        assert buckets["aged/gen-after-decode"] == [1, 1], buckets
        report(buckets, pc, ns)
    print("SELFTEST PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="MTP 草稿污染机制判决")
    ap.add_argument("cloud_log", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.cloud_log:
        ap.error("需要云日志路径")
    buckets, pc, ns = classify(scan(args.cloud_log))
    report(buckets, pc, ns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
