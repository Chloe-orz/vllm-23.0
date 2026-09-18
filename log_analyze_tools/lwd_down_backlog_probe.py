#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""lwd_down_backlog_probe: DOWN 通道积压/爆显存归因工具。

问题背景:bench 打到一半云侧爆显存,怀疑"边侧 embed 优先 + MTP unembed
尾段 ×4 → 云侧 DOWN 快照 clone 积压"。本工具用日志定量裁决。

与 lwd_backlog_probe 的分工:老工具刻意回避一切跨机 join(钟差敏感);
本工具的核心指标 **seqno 滞后曲线** 对常数钟差免疫——
lag(N) = t_edge_exec(N) - t_cloud_send(N),两边机器的钟差是一个常数,
被整体减去(以开局段中位数归零),剩下的**斜率**就是积压增长率,
钟漂移(频率差)才可能污染斜率,工具会用同机指标 c2e_pending 斜率交叉验证。

输入日志锚点(全部为现有日志,无新增):
  云: [Lwd][cloud-worker] DOWN send seqno=S numel=E      — 每步一条(生产)
      [Lwd][perf] cloud-step seqno=S sample=A send=B total=Tms rows=R
  边: [Lwd][edge-worker] UNEMBED seqno=S reqs=N accepted=A num_elements=E
      [Lwd][edge-worker] EMBED seqno=S reqs=N tokens=T
      [Lwd][perf] unembed seqno=S post_recv=.. wait_tensor=.. lm_head=..
      select=.. total=Tms reqs=N [sync=1]
      [Lwd][perf] embed seqno=S forward=.. submit_send=.. total=Tms tokens=N
      [Lwd][sched] edge dispatch-unembed seqno=S reqs=N rows=R
      ahead_unemb=U ahead_emb=E c2e_wait=Wms c2e_pending=P
      [Lwd][sched] edge step ... c2e_pending=P

输出:
  ① seqno 滞后曲线(钟差免疫):开局/中段/结尾读数、斜率、单步增量
  ② 生产/消费速率差(各自单机时钟):云 DOWN/s vs 边 UNEMBED/s
  ③ 云侧显存增长推算:速率差 × (numel×2B×2份[cat源+clone])
  ④ 边侧水位:c2e_pending / ahead_unemb / ahead_emb 趋势
  ⑤ 边侧分段:wait_tensor/lm_head/select p50/p95,select 占比
  ⑥ 判定:积压方向 / 消费瓶颈段 / 或"无积压→嫌疑转向清单"

用法:
  python lwd_down_backlog_probe.py --cloud 云日志 [云日志2..] --edge 边日志 [边日志2..]
  python lwd_down_backlog_probe.py --selftest
  可选 --plot out.png(需 matplotlib,缺失自动跳过)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

RE_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.](\d{3})")

RE_CLOUD_DOWN = re.compile(
    r"\[Lwd\]\[cloud-worker\] DOWN send seqno=(\d+) numel=(\d+)")
RE_CLOUD_STEP = re.compile(
    r"\[Lwd\]\[perf\] cloud-step seqno=(\d+) sample=([\d.]+) "
    r"send=([\d.]+) total=([\d.]+)ms rows=(\d+)")
RE_EDGE_UNEMBED = re.compile(
    r"\[Lwd\]\[edge-worker\] UNEMBED seqno=(\d+) reqs=(\d+) "
    r"accepted=(\d+) num_elements=(\d+)")
RE_EDGE_EMBED = re.compile(
    r"\[Lwd\]\[edge-worker\] EMBED seqno=(\d+) reqs=(\d+) tokens=(\d+)")
RE_EDGE_UNEMBED_PERF = re.compile(
    r"\[Lwd\]\[perf\] unembed seqno=(\S+) post_recv=([\d.]+) "
    r"wait_tensor=([\d.]+) lm_head=([\d.]+) select=([\d.]+) "
    r"total=([\d.]+)ms reqs=(\d+)(.*sync=1)?")
RE_EDGE_EMBED_PERF = re.compile(
    r"\[Lwd\]\[perf\] embed seqno=(\S+) forward=([\d.]+) "
    r"submit_send=([\d.]+) total=([\d.]+)ms tokens=(\d+)")
RE_DISP_UNEMB = re.compile(
    r"\[Lwd\]\[sched\] edge dispatch-unembed seqno=(\S+) reqs=(\d+) "
    r"rows=(\d+) ahead_unemb=(\d+) ahead_emb=(\d+) "
    r"c2e_wait=([\d.]+)ms c2e_pending=(\d+)")
RE_DISP_EMB = re.compile(
    r"\[Lwd\]\[sched\] edge dispatch-embed seqno=(\d+) req=\S+ "
    r"tokens=(\d+) ahead_unemb=(\d+) ahead_emb=(\d+) c2e_pending=(\d+)")
RE_EDGE_STEP = re.compile(
    r"\[Lwd\]\[sched\] edge step harvest_emb=\d+ harvest_unemb=\d+ "
    r"disp_emb=(\d+) disp_unemb=(\d+) queue=(\d+)emb/(\d+)unemb "
    r"c2e_pending=(\d+)")


def _ts_to_sec(ts: str, ms: str) -> float:
    t = _dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    return t.timestamp() + int(ms) / 1000.0


def _parse(files: list[str]) -> list[tuple[float, str]]:
    """多文件合并(同机多进程日志),按时间稳定排序。"""
    out: list[tuple[float, int, str]] = []
    seq = 0
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = RE_TS.match(line)
                if not m:
                    continue
                out.append((_ts_to_sec(m.group(1), m.group(2)), seq, line))
                seq += 1
    out.sort(key=lambda x: (x[0], x[1]))
    return [(t, line) for t, _, line in out]


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
    return s[k]


def _slope_ms_per_unit(x: list[float], y: list[float]) -> float:
    """OLS 斜率,y 单位换算为 ms/unit。样本过少或 x 无跨度返回 nan。"""
    n = len(x)
    if n < 5:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    if sxx < 1e-9:
        return float("nan")
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / sxx  # y 的单位原样保留(调用方自行换算)


def analyze(cloud_files: list[str], edge_files: list[str]) -> dict:
    rep: dict = {}
    cloud_lines = _parse(cloud_files) if cloud_files else []
    edge_lines = _parse(edge_files) if edge_files else []

    cloud_send: dict[int, float] = {}
    numel: list[int] = []
    cloud_rows: list[int] = []
    cloud_total: list[float] = []
    cloud_sample: list[float] = []
    for t, line in cloud_lines:
        m = RE_CLOUD_DOWN.search(line)
        if m:
            cloud_send[int(m.group(1))] = t
            numel.append(int(m.group(2)))
            continue
        m = RE_CLOUD_STEP.search(line)
        if m:
            cloud_sample.append(float(m.group(2)))
            cloud_total.append(float(m.group(4)))
            cloud_rows.append(int(m.group(5)))

    edge_exec: dict[int, float] = {}
    edge_reqs: list[int] = []
    edge_accepted: list[int] = []
    seg: dict[str, list[float]] = {k: [] for k in (
        "post_recv", "wait_tensor", "lm_head", "select", "total")}
    embed_total: list[float] = []
    sync_probes = 0
    c2e_pending: list[tuple[float, int]] = []
    ahead_unemb: list[int] = []
    ahead_emb: list[int] = []
    disp_unemb_count = 0
    disp_emb_count = 0
    for t, line in edge_lines:
        m = RE_EDGE_UNEMBED.search(line)
        if m:
            edge_exec[int(m.group(1))] = t
            edge_reqs.append(int(m.group(2)))
            edge_accepted.append(int(m.group(3)))
            continue
        m = RE_EDGE_UNEMBED_PERF.search(line)
        if m:
            for k in ("post_recv", "wait_tensor", "lm_head", "select", "total"):
                seg[k].append(float(m.group(
                    2 + ("post_recv", "wait_tensor", "lm_head", "select",
                         "total").index(k))))
            if m.group(8):
                sync_probes += 1
            continue
        m = RE_EDGE_EMBED_PERF.search(line)
        if m:
            embed_total.append(float(m.group(4)))
            continue
        m = RE_DISP_UNEMB.search(line)
        if m:
            disp_unemb_count += 1
            ahead_unemb.append(int(m.group(4)))
            ahead_emb.append(int(m.group(5)))
            c2e_pending.append((t, int(m.group(7))))
            continue
        m = RE_DISP_EMB.search(line)
        if m:
            disp_emb_count += 1
            ahead_unemb.append(int(m.group(3)))
            ahead_emb.append(int(m.group(4)))
            c2e_pending.append((t, int(m.group(5))))
            continue
        m = RE_EDGE_STEP.search(line)
        if m:
            c2e_pending.append((t, int(m.group(5))))

    rep["cloud_send"] = cloud_send
    rep["edge_exec"] = edge_exec
    rep["numel"] = numel
    rep["seg"] = seg
    rep["sync_probes"] = sync_probes

    # ① seqno 滞后曲线(常数钟差归零)
    common = sorted(set(cloud_send) & set(edge_exec))
    lags = [(n, edge_exec[n] - cloud_send[n]) for n in common]
    if len(lags) >= 10:
        base = _pct([l for _, l in lags[: max(5, len(lags) // 10)]], 5)
        norm = [(n, (l - base) * 1000.0) for n, l in lags]  # ms
        k = max(1, len(norm) // 10)
        head = _pct([l for _, l in norm[:k]], 50)
        mid = _pct([l for _, l in norm], 50)
        tail = _pct([l for _, l in norm[-k:]], 50)
        slope = _slope_ms_per_unit([float(n) for n, _ in norm],
                                   [l for _, l in norm])
        rep["lag"] = {
            "n": len(norm), "seqno_span": (norm[0][0], norm[-1][0]),
            "head_ms": head, "mid_ms": mid, "tail_ms": tail,
            "max_ms": max(l for _, l in norm), "slope_ms_per_seq": slope,
        }
    else:
        rep["lag"] = None

    # ② 速率(各自单机时钟,取内窗避开起停)
    def _rate(ts: list[float]) -> float:
        if len(ts) < 5:
            return float("nan")
        ts = sorted(ts)
        margin = min(5.0, max(0.0, (ts[-1] - ts[0]) * 0.1))
        lo, hi = ts[0] + margin, ts[-1] - margin
        inner = [v for v in ts if lo <= v <= hi]
        if len(inner) < 2 or hi <= lo:
            return float("nan")
        return (len(inner) - 1) / (inner[-1] - inner[0])

    rep["rate"] = {
        "cloud_down_per_s": _rate(list(cloud_send.values())),
        "edge_unembed_per_s": _rate(list(edge_exec.values())),
        "span_s": (min(list(cloud_send.values()) or [0]),
                   max(list(cloud_send.values()) or [0])),
    }

    # ③ 显存增长推算:速率差 × numel×2B(bf16) × 2份(cat源+clone)
    rc, re_ = rep["rate"]["cloud_down_per_s"], rep["rate"]["edge_unembed_per_s"]
    if numel and rc == rc and re_ == re_:
        mean_numel = sum(numel) / len(numel)
        per_step_bytes = mean_numel * 2 * 2
        deficit = rc - re_
        rep["mem"] = {
            "mean_numel": mean_numel, "per_step_mb": per_step_bytes / 1e6,
            "deficit_per_s": deficit,
            "growth_mb_per_min": deficit * per_step_bytes * 60 / 1e6,
        }
    else:
        rep["mem"] = None

    # ④ 水位趋势
    if len(c2e_pending) >= 10:
        rep["c2e_series"] = c2e_pending
        xs = [t for t, _ in c2e_pending]
        ys = [float(v) for _, v in c2e_pending]
        k = max(1, len(ys) // 10)
        rep["water"] = {
            "c2e_pending_p50": _pct(ys, 50), "c2e_pending_p95": _pct(ys, 95),
            "c2e_pending_max": max(ys),
            "c2e_head": _pct(ys[:k], 50), "c2e_tail": _pct(ys[-k:], 50),
            "c2e_slope_per_min": _slope_ms_per_unit(xs, ys) * 60,
            "ahead_unemb_max": max(ahead_unemb) if ahead_unemb else -1,
            "ahead_emb_max": max(ahead_emb) if ahead_emb else -1,
            "disp_emb": disp_emb_count, "disp_unemb": disp_unemb_count,
        }
    else:
        rep["water"] = None

    # ⑤ 分段统计
    rep["seg_stat"] = {
        k: {"p50": _pct(v, 50), "p95": _pct(v, 95), "max": max(v)}
        for k, v in seg.items() if v}
    rep["embed_total"] = (
        {"p50": _pct(embed_total, 50), "p95": _pct(embed_total, 95)}
        if embed_total else None)
    rep["cloud_step"] = (
        {"total_p50": _pct(cloud_total, 50), "total_p95": _pct(cloud_total, 95),
         "sample_p50": _pct(cloud_sample, 50),
         "rows_mean": (sum(cloud_rows) / len(cloud_rows)) if cloud_rows
         else float("nan")}
        if cloud_total else None)
    rep["reqs"] = {
        "unembed_batches": len(edge_reqs),
        "reqs_mean": (sum(edge_reqs) / len(edge_reqs)) if edge_reqs
        else float("nan"),
        "accept_mean": (sum(edge_accepted) / len(edge_accepted))
        if edge_accepted else float("nan"),
    }
    return rep


def _fmt(v: float, nd: int = 1) -> str:
    return f"{v:.{nd}f}" if v == v else "n/a"


def print_report(rep: dict) -> None:
    print("=" * 72)
    print("① seqno 滞后曲线(常数钟差已归零;斜率=单步积压增量,单位 ms/seq)")
    lag = rep["lag"]
    if lag:
        print(f"   配对 seqno 数={lag['n']}  跨度=[{lag['seqno_span'][0]},"
              f"{lag['seqno_span'][1]}]")
        print(f"   开局 p50={_fmt(lag['head_ms'])}ms  全程 p50="
              f"{_fmt(lag['mid_ms'])}ms  结尾 p50={_fmt(lag['tail_ms'])}ms"
              f"  最大={_fmt(lag['max_ms'])}ms")
        print(f"   斜率={_fmt(lag['slope_ms_per_seq'], 2)} ms/seq  "
              f"(全程增长={_fmt(lag['tail_ms'] - lag['head_ms'])}ms)")
    else:
        print("   配对样本不足(<10)——检查两侧 seqno 日志是否齐全")

    print("② 生产/消费速率(各自单机时钟,内窗)")
    r = rep["rate"]
    print(f"   云 DOWN send : {_fmt(r['cloud_down_per_s'], 2)} 条/s")
    print(f"   边 UNEMBED 执行: {_fmt(r['edge_unembed_per_s'], 2)} 条/s"
          f"   (云侧时间跨度 {_fmt(r['span_s'][1] - r['span_s'][0], 0)}s)")

    print("③ 云侧显存增长推算(速率差 × numel×2B×2份[cat源+clone])")
    m = rep["mem"]
    if m:
        print(f"   平均 numel={_fmt(m['mean_numel'], 0)}  每步滞留="
              f"{_fmt(m['per_step_mb'], 2)}MB  速率差="
              f"{_fmt(m['deficit_per_s'], 2)} 条/s")
        print(f"   → 预计云侧显存增速 {max(0.0, m['growth_mb_per_min']):.1f}"
              f" MB/min(负值=无积压)")
    else:
        print("   数据不足")

    print("④ 边侧水位(同机时钟,可交叉验证①的斜率)")
    w = rep["water"]
    if w:
        print(f"   c2e_pending p50={_fmt(w['c2e_pending_p50'], 0)} "
              f"p95={_fmt(w['c2e_pending_p95'], 0)} "
              f"max={_fmt(w['c2e_pending_max'], 0)}  "
              f"开局→结尾 {_fmt(w['c2e_head'], 0)}→{_fmt(w['c2e_tail'], 0)}"
              f"  斜率={_fmt(w['c2e_slope_per_min'], 1)}/min")
        print(f"   ahead_unemb max={w['ahead_unemb_max']}(EMBED 前压着的"
              f" UNEMBED)  ahead_emb max={w['ahead_emb_max']}")
        print(f"   派发计数 embed={w['disp_emb']} unembed={w['disp_unemb']}")
    else:
        print("   水位样本不足(<10)")

    print("⑤ 边侧 unembed 分段 ms(sync 探针 "
          f"{rep['sync_probes']} 条)")
    for k in ("wait_tensor", "lm_head", "select", "post_recv", "total"):
        s = rep["seg_stat"].get(k)
        if s:
            print(f"   {k:<12} p50={_fmt(s['p50'], 2)}  p95={_fmt(s['p95'], 2)}"
                  f"  max={_fmt(s['max'], 2)}")
    if rep["embed_total"]:
        e = rep["embed_total"]
        print(f"   (对照 embed total p50={_fmt(e['p50'], 2)} "
              f"p95={_fmt(e['p95'], 2)})")
    if rep["cloud_step"]:
        c = rep["cloud_step"]
        print(f"   (云步 total p50={_fmt(c['total_p50'], 2)} "
              f"p95={_fmt(c['total_p95'], 2)}  sample p50="
              f"{_fmt(c['sample_p50'], 2)}  平均rows="
              f"{_fmt(c['rows_mean'], 1)})")
    q = rep["reqs"]
    print(f"   (unembed 批数={q['unembed_batches']}  平均reqs="
          f"{_fmt(q['reqs_mean'], 1)}  平均accepted={_fmt(q['accept_mean'], 2)})")

    print("⑥ 判定")
    verdicts: list[str] = []
    growing = (lag and lag["slope_ms_per_seq"] == lag["slope_ms_per_seq"]
               and lag["slope_ms_per_seq"] > 1.0
               and lag["tail_ms"] - lag["head_ms"] > 1000.0)
    flat = (lag and lag["slope_ms_per_seq"] == lag["slope_ms_per_seq"]
            and abs(lag["tail_ms"] - lag["head_ms"]) <= 1000.0)
    if growing:
        verdicts.append(
            f"DOWN 积压持续增长(+{_fmt(lag['slope_ms_per_seq'], 1)}ms/seq,"
            f"全程 +{_fmt((lag['tail_ms'] - lag['head_ms']) / 1000, 1)}s)"
            f"→ 云侧快照 clone 累积、爆显存方向【成立】")
    elif flat:
        verdicts.append(
            "seqno 滞后平稳(±1s 内)→ 无 DOWN 积压,爆显存另有原因:"
            "查边侧 sort 工作区峰值/云侧 MTP draft 瞬态/分配器碎片,"
            "并先确认 OOM 落在哪侧哪个 rank")
    else:
        verdicts.append("滞后数据不足,无法判定积压方向")
    if w and w["c2e_tail"] - w["c2e_head"] > 50:
        verdicts.append(
            f"c2e_pending 水位爬升({_fmt(w['c2e_head'], 0)}→"
            f"{_fmt(w['c2e_tail'], 0)})→ 与积压方向互相印证")
    if m and m["growth_mb_per_min"] > 100:
        verdicts.append(
            f"推算云侧显存增速 {m['growth_mb_per_min']:.0f} MB/min"
            f"——与 npu-smi 观测斜率对账")
    s = rep["seg_stat"]
    if s.get("select") and s.get("total"):
        share = s["select"]["p50"] / max(1e-9, s["total"]["p50"])
        verdicts.append(
            f"边侧消费瓶颈段: select 占 total 的 {share * 100:.0f}%"
            f"(p50 {_fmt(s['select']['p50'], 1)}/{_fmt(s['total']['p50'], 1)}"
            f"ms)→ 排序是拖慢清账的主项,行压缩+topk 方向有效"
            if share > 0.5 else
            f"select 占比 {share * 100:.0f}%,非主导——看 wait_tensor/lm_head")
    for v in verdicts:
        print(f"   • {v}")
    print("=" * 72)


def selftest() -> int:
    import tempfile
    fmt = "%Y-%m-%d %H:%M:%S,%f"
    def _st(t: _dt.datetime) -> str:
        return t.strftime(fmt)[:-3]  # %f 6位→3位毫秒
    t0 = _dt.datetime(2026, 9, 18, 10, 0, 0)
    cloud, edge = [], []
    # 云 15ms/步产出,边 20ms/步消费 → 每步落后 5ms;c2e_pending 线性爬升
    t_c, t_e = t0, t0 + _dt.timedelta(seconds=1)
    pending = 0
    for i in range(300):
        cloud.append(
            f"{_st(t_c)} [lwd_cloud_worker.py:225] "
            f"[Lwd][cloud-worker] DOWN send seqno={i} numel=8388608")
        cloud.append(
            f"{_st(t_c)} [lwd_cloud_worker.py:245] "
            f"[Lwd][perf] cloud-step seqno={i} sample=2.00 send=1.00 "
            f"total=15.00ms rows=4")
        edge.append(
            f"{_st(t_e)} [lwd_edge_worker.py:140] "
            f"[Lwd][edge-worker] UNEMBED seqno={i} reqs=1 accepted=2 "
            f"num_elements=8388608")
        edge.append(
            f"{_st(t_e)} [lwd_edge_worker.py:251] "
            f"[Lwd][perf] unembed seqno={i} post_recv=0.10 wait_tensor=1.00 "
            f"lm_head=3.00 select=30.00 total=35.00ms reqs=1")
        edge.append(
            f"{_st(t_e)} [lwd_edge_engine.py:357] "
            f"[Lwd][sched] edge dispatch-unembed seqno={i} reqs=1 rows=4 "
            f"ahead_unemb=0 ahead_emb=0 c2e_wait=0.50ms c2e_pending={pending}")
        pending += 1
        t_c += _dt.timedelta(milliseconds=15)
        t_e += _dt.timedelta(milliseconds=20)
    with tempfile.TemporaryDirectory() as d:
        cf, ef = Path(d) / "c.log", Path(d) / "e.log"
        cf.write_text("\n".join(cloud) + "\n")
        ef.write_text("\n".join(edge) + "\n")
        rep = analyze([str(cf)], [str(ef)])
        lag = rep["lag"]
        assert lag and lag["slope_ms_per_seq"] > 3.0, \
            f"斜率应≈5ms/seq,实测 {lag and lag['slope_ms_per_seq']}"
        assert rep["water"]["c2e_tail"] > rep["water"]["c2e_head"] + 100
        assert rep["mem"]["growth_mb_per_min"] > 500, \
            f"推算增速应≈588MB/min,实测 {rep['mem']['growth_mb_per_min']}"
        print_report(rep)
    print("SELFTEST PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--cloud", nargs="+", help="云侧日志文件(可多个)")
    ap.add_argument("--edge", nargs="+", help="边侧日志文件(可多个)")
    ap.add_argument("--plot", help="可选:输出 PNG(需 matplotlib)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.cloud or not args.edge:
        ap.error("--cloud 与 --edge 均需至少一个日志文件")
    rep = analyze(args.cloud, args.edge)
    print_report(rep)
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
            if rep["lag"]:
                ns = sorted(set(rep["cloud_send"]) & set(rep["edge_exec"]))
                ys = [(rep["edge_exec"][n] - rep["cloud_send"][n]) * 1000
                      for n in ns]
                ax1.plot(ns, ys, lw=0.8)
                ax1.set_xlabel("DOWN seqno")
                ax1.set_ylabel("lag (ms, 未归零)")
                ax1.set_title("DOWN seqno lag: edge exec - cloud send")
            series = rep.get("c2e_series") or []
            if series:
                t0 = series[0][0]
                ax2.plot([t - t0 for t, _ in series],
                         [v for _, v in series], lw=0.8)
                ax2.set_xlabel("边侧本地时间 (s)")
                ax2.set_ylabel("c2e_pending")
                ax2.set_title("edge c2e_pending watermark")
            else:
                ax2.set_visible(False)
            fig.tight_layout()
            fig.savefig(args.plot, dpi=120)
            print(f"plot -> {args.plot}")
        except ImportError:
            print("matplotlib 不可用,跳过绘图")
    return 0


if __name__ == "__main__":
    sys.exit(main())
