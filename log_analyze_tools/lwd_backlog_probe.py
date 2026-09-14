#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""lwd_backlog_probe: c2e 积压/背压定位工具(单请求或压测后分析)。

输入:云日志 + 边日志(两份文件路径;时间戳各自独立解析,跨机钟差
敏感的指标会自动跳过)。依赖的日志锚点(全部为现有日志,无新增):

  云: [Lwd][cloud-ctrl] handle_model_output: c2e_meta received
      reqs=N down_seqno=S                    — 每 notify 一条(生产侧)
      [Lwd][perf] cloud-step dt=Xms          — 云步间隔(背压探针)
      [Lwd][perf] bridge-wait op=send ...    — 发送桥接等待(锁步探针)
  边: [Lwd][edge-worker] UNEMBED seqno=S ... — 每条处理一次(消费侧)
      [Lwd][perf] unembed seqno=S post_recv=A wait_tensor=B
      lm_head=C select=D total=Ems           — 边单条成本分段
      [Lwd][perf] harvest dur=Xms            — 引擎收割时长

输出:
  1. 生产/消费速率对比(时钟无关,两边各自算间隔)
  2. 边单条成本分段排名(谁在吃时间)
  3. 尾巴检测:云结束后边"背靠背清账"段的条数与时长(时钟无关)
  4. 判定结论(边瓶颈/背压触发/尾巴=积压清账)

用法:
  python lwd_backlog_probe.py --cloud 云日志 --edge 边日志 [--verbose]
  python lwd_backlog_probe.py --selftest
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

RE_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.](\d{3})")
RE_CLOUD_NOTIFY = re.compile(
    r"\[Lwd\]\[cloud-ctrl\] handle_model_output: c2e_meta received "
    r"reqs=(\S+) down_seqno=(\d+)")
RE_CLOUD_STEP = re.compile(r"\[Lwd\]\[perf\] cloud-step dt=([\d.]+)ms")
RE_BRIDGE = re.compile(
    r"\[Lwd\]\[perf\] bridge-wait op=(\w+) ch=(\w+) dur=([\d.]+)ms")
RE_EDGE_UNEMBED = re.compile(r"\[Lwd\]\[edge-worker\] UNEMBED seqno=(\d+)")
RE_EDGE_PERF = re.compile(
    r"\[Lwd\]\[perf\] unembed seqno=(\S+) post_recv=([\d.]+) "
    r"wait_tensor=([\d.]+) lm_head=([\d.]+) select=([\d.]+) "
    r"total=([\d.]+)ms")
RE_HARVEST = re.compile(r"\[Lwd\]\[perf\] harvest dur=([\d.]+)ms")

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def read_lines(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        sys.exit(f"[backlog-probe] 文件不存在: {path}")
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "utf-8", "gb18030", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover
        text = raw.decode("utf-8", errors="replace")
    return [ANSI.sub("", l) for l in text.splitlines()]


def parse_ts(line: str) -> float | None:
    m = RE_TS.match(line)
    if not m:
        return None
    base = _dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    return base.timestamp() + int(m.group(2)) / 1000.0


def stats(vals: list[float]) -> str:
    if not vals:
        return "n=0"
    s = sorted(vals)

    def pct(p: float) -> float:
        return s[min(len(s) - 1, int(len(s) * p))]

    return (f"n={len(s)} 均值={sum(s)/len(s):.1f} "
            f"p50={pct(0.5):.1f} p90={pct(0.9):.1f} max={s[-1]:.1f}")


def diffs(vals: list[float]) -> list[float]:
    return [b - a for a, b in zip(vals, vals[1:]) if b > a]


def ms(v):
    """秒→毫秒,接受标量或列表。"""
    if isinstance(v, list):
        return [x * 1000.0 for x in v]
    return v * 1000.0


def analyze(cloud_log: str, edge_log: str, verbose: bool) -> None:
    # ---- 云侧 ----
    cloud_ts: list[float] = []
    cloud_seq: list[int] = []
    step_dt: list[float] = []
    step_ts: list[float] = []
    bridge: dict[tuple[str, str], list[float]] = {}
    for line in read_lines(cloud_log):
        t = parse_ts(line)
        m = RE_CLOUD_NOTIFY.search(line)
        if m:
            if t is not None:
                cloud_ts.append(t)
                cloud_seq.append(int(m.group(2)))
            continue
        m = RE_CLOUD_STEP.search(line)
        if m:
            step_dt.append(float(m.group(1)))
            if t is not None:
                step_ts.append(t)
            continue
        m = RE_BRIDGE.search(line)
        if m:
            bridge.setdefault((m.group(1), m.group(2)), []).append(
                float(m.group(3)))
    # 引擎 handle_model_output 行被清理后,云事件流回退到探针行时间戳
    if len(cloud_ts) < 10 and step_ts:
        cloud_ts = step_ts

    # ---- 边侧 ----
    # 事件锚点优先级:UNEMBED 派发行(旧)> [Lwd][perf] unembed 行(现有,
    # 携带 seqno)——后者在"移除调试日志"提交后是唯一可靠锚
    edge_ts: list[float] = []
    edge_seq: list[int] = []
    seg: dict[str, list[float]] = {}
    for line in read_lines(edge_log):
        t = parse_ts(line)
        m = RE_EDGE_UNEMBED.search(line)
        if m and t is not None:
            edge_ts.append(t)
            edge_seq.append(int(m.group(1)))
            continue
        m = RE_EDGE_PERF.search(line)
        if m:
            if t is not None and not edge_ts:
                edge_ts.append(t)
                try:
                    edge_seq.append(int(m.group(1)))
                except ValueError:
                    pass
            for name, idx in (("post_recv", 2), ("wait_tensor", 3),
                              ("lm_head", 4), ("select", 5), ("total", 6)):
                seg.setdefault(name, []).append(float(m.group(idx)))
            continue

    print("=" * 64)
    print("① 速率对比(时钟无关:两边各自算事件间隔)")
    c_iv = ms(diffs(cloud_ts))
    e_iv = ms(diffs(edge_ts))
    print(f"   云 notify 间隔: {stats(c_iv)} ms")
    print(f"   边 UNEMBED 间隔: {stats(e_iv)} ms")
    if step_dt:
        print(f"   云步间隔(探针): {stats(step_dt)} ms")
    for k, v in sorted(bridge.items()):
        print(f"   bridge-wait {k[0]}/{k[1]}: {stats(v)} ms")

    print("=" * 64)
    print("② 边单条成本分段(积压根源)")
    for name in ("post_recv", "wait_tensor", "lm_head", "select", "total"):
        print(f"   {name:<11}: {stats(seg.get(name, []))} ms")

    print("=" * 64)
    print("③ 尾巴检测(时钟无关:边事件从'等云节奏'切到'背靠背清账')")
    tail_cnt = tail_ms = 0
    if e_iv and seg.get("total"):
        svc = sum(seg["total"]) / len(seg["total"])  # 边服务时长
        thr = max(svc * 3.0, 30.0)  # 背靠背段:间隔≈服务时长(无等云空隙)
        i = len(e_iv)
        while i > 0 and e_iv[i - 1] < thr:
            i -= 1
        tail_cnt = len(e_iv) - i + (1 if i < len(e_iv) else 0)
        tail_ms = sum(e_iv[i:])
    print(f"   背靠背段条数: {tail_cnt}  时长: {tail_ms:.0f} ms")

    print("=" * 64)
    print("④ 判定")
    verdicts = []
    if seg.get("total") and c_iv:
        edge_cost = sum(seg["total"]) / len(seg["total"])
        cloud_pace = sum(c_iv) / len(c_iv)
        if edge_cost > cloud_pace:
            verdicts.append(
                f"边瓶颈坐实:边单条 {edge_cost:.1f}ms > 云步 {cloud_pace:.1f}ms,"
                f"每 token 净欠 {edge_cost - cloud_pace:.1f}ms → 必然积压")
        else:
            verdicts.append(
                f"边单条({edge_cost:.1f}ms)≤云步({cloud_pace:.1f}ms):"
                "稳态不应积压,若有尾巴查唤醒链/突发")
    if tail_cnt > 50:
        verdicts.append(
            f"尾巴 = 积压清账:云结束后边独自清了 {tail_cnt} 条 × "
            f"{(tail_ms / tail_cnt) if tail_cnt else 0:.0f}ms/条")
    if step_dt and c_iv:
        dt_mean = sum(step_dt) / len(step_dt)
        pace = sum(c_iv) / len(c_iv)
        if dt_mean > pace * 1.3:
            verdicts.append(
                f"云步间隔({dt_mean:.1f}ms)明显大于 notify 节奏"
                f"({pace:.1f}ms):队列满背压已生效")
    for v in verdicts or ["样本不足或无明显异常,看上面原始数字"]:
        print(f"   • {v}")

    if verbose:
        print("=" * 64)
        print("原始计数:",
              f"云 notify={len(cloud_seq)}(seqno {cloud_seq[0] if cloud_seq else '-'}"
              f"~{cloud_seq[-1] if cloud_seq else '-'})",
              f"边 UNEMBED={len(edge_seq)}(seqno {edge_seq[0] if edge_seq else '-'}"
              f"~{edge_seq[-1] if edge_seq else '-'})")


def selftest() -> None:
    import tempfile
    cloud = "\n".join([
        "2026-09-12 10:00:00,100 INFO [Lwd][cloud-ctrl] handle_model_output: "
        "c2e_meta received reqs=1 down_seqno=0",
        "2026-09-12 10:00:00,115 INFO [Lwd][perf] cloud-step dt=15.0ms",
        "2026-09-12 10:00:00,115 INFO [Lwd][cloud-ctrl] handle_model_output: "
        "c2e_meta received reqs=1 down_seqno=1",
        "2026-09-12 10:00:00,130 INFO [Lwd][perf] cloud-step dt=15.0ms",
    ])
    edge = "\n".join([
        "2026-09-12 10:00:00,105 INFO [Lwd][edge-worker] UNEMBED seqno=0 reqs=1",
        "2026-09-12 10:00:00,105 INFO [Lwd][perf] unembed seqno=0 post_recv=0.5 "
        "wait_tensor=1.0 lm_head=2.0 select=10.0 total=14.0ms reqs=1",
        "2026-09-12 10:00:00,124 INFO [Lwd][edge-worker] UNEMBED seqno=1 reqs=1",
        "2026-09-12 10:00:00,124 INFO [Lwd][perf] unembed seqno=1 post_recv=0.5 "
        "wait_tensor=1.0 lm_head=2.0 select=10.0 total=14.0ms reqs=1",
    ])
    with tempfile.TemporaryDirectory() as d:
        for name, body in (("c.log", cloud), ("e.log", edge)):
            Path(d, name).write_text(body, encoding="utf-8")
        analyze(str(Path(d, "c.log")), str(Path(d, "e.log")), True)


def main() -> None:
    ap = argparse.ArgumentParser(description="lwd c2e backlog probe")
    ap.add_argument("--cloud", help="云侧日志路径")
    ap.add_argument("--edge", help="边侧日志路径")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.cloud and args.edge):
        ap.error("需要 --cloud 与 --edge(或 --selftest)")
    analyze(args.cloud, args.edge, args.verbose)


if __name__ == "__main__":
    main()
