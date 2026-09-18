#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""lwd_bench_gap_probe: bench 与集中式差距归因工具(token_id 版日志)。

问题:token_id 回传版单请求与集中式无差,bench(多请求)明显偏慢。
本工具用现有日志定量三个"只在多请求下发作"的嫌疑:

  ① 相位稀释:相位调度器 PREFILL/DECODE 互斥交替,bench 期持续到达的
     prefill 步把全体 decode 的步进稀释(decode 间隔被拉长)。
     集中式原生混批(chunked prefill 与 decode 同 forward)无此税。
  ② embed 准入:prefill 输入经"边侧嵌入→UP→云注入"绕行,bench 到达率
     高时在边侧排队,准入吞吐被钉死。
  ③ MTP 接受率漂移:批组合变化使接受数下降,tokens/step 降低。

日志锚点(全部现有,无新增):
  云: [Lwd][sched] cloud step=N phase=P seqno=S reqs=[..] tokens=T
      pending_notify=K decode_ready=D
      [Lwd][perf] cloud-step exec=Xms / cloud-exec total=Xms
      [Lwd][cloud-ctrl] publish c2e(token-id): reqs=N tokens=[a,b,..]
      [Lwd][cloud-ctrl] RangeNotify req=R num=N seqno=S
  边: [Lwd][sched] edge dispatch-embed seqno=S req=R tokens=T queue=Qemb
      c2e_pending=P
      [Lwd][sched] edge step harvest_emb=H deliver_unemb=U disp_emb=D
      queue=Qemb c2e_pending=P
      [Lwd][sched] edge deliver-unembed reqs=N tokens=[a,b,..]
      c2e_wait=Wms c2e_pending=P

用法:
  python lwd_bench_gap_probe.py --cloud 云日志 --edge 边日志
  python lwd_bench_gap_probe.py --selftest
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys

RE_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.](\d{3})")

RE_PHASE = re.compile(
    r"\[Lwd\]\[sched\] cloud step=\d+ phase=(\w+) seqno=(\S+) "
    r"reqs=\[(.*?)\] tokens=(\d+) pending_notify=(\d+) decode_ready=(\d+)")
RE_EXEC = re.compile(r"\[Lwd\]\[perf\] cloud-step exec=([\d.]+)ms")
RE_CEXEC = re.compile(r"\[Lwd\]\[perf\] cloud-exec total=([\d.]+)ms")
RE_PUB = re.compile(
    r"\[Lwd\]\[cloud-ctrl\] publish c2e\(token-id\): reqs=(\d+) "
    r"tokens=\[([\d, ]*)\]")
RE_RANGE = re.compile(r"\[Lwd\]\[cloud-ctrl\] RangeNotify req=\S+ num=\S+")
RE_DISP_EMB = re.compile(
    r"\[Lwd\]\[sched\] edge dispatch-embed seqno=\d+ req=\S+ tokens=(\d+) "
    r"queue=(\d+)emb c2e_pending=(\d+)")
RE_STEP = re.compile(
    r"\[Lwd\]\[sched\] edge step harvest_emb=(\d+) deliver_unemb=(\d+) "
    r"disp_emb=(\d+) queue=(\d+)emb c2e_pending=(\d+)")
RE_DELIVER = re.compile(
    r"\[Lwd\]\[sched\] edge deliver-unembed reqs=(\d+) tokens=\[([\d, ]*)\]")


def _sec(ts: str, ms: str) -> float:
    return _dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp() \
        + int(ms) / 1000.0


def _parse(path: str):
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RE_TS.match(line)
            if m:
                out.append((_sec(m.group(1), m.group(2)), line))
    out.sort(key=lambda x: x[0])
    return out


def _pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))]


def _rate(ts):
    if len(ts) < 5:
        return float("nan")
    ts = sorted(ts)
    margin = min(5.0, max(0.0, (ts[-1] - ts[0]) * 0.1))
    inner = [t for t in ts if ts[0] + margin <= t <= ts[-1] - margin]
    if len(inner) < 2 or inner[-1] <= inner[0]:
        return float("nan")
    return (len(inner) - 1) / (inner[-1] - inner[0])


def analyze(cloud_path: str, edge_path: str) -> dict:
    rep: dict = {}
    phases: list[tuple[float, str, int, int]] = []  # (t, phase, tokens, pending)
    exec_ms: list[float] = []
    cexec_ms: list[float] = []
    pub_tokens: list[list[int]] = []
    pub_t: list[float] = []
    range_t: list[float] = []
    for t, line in _parse(cloud_path):
        m = RE_PHASE.search(line)
        if m:
            phases.append((t, m.group(1), int(m.group(4)), int(m.group(5))))
            continue
        m = RE_EXEC.search(line)
        if m:
            exec_ms.append(float(m.group(1)))
            continue
        m = RE_CEXEC.search(line)
        if m:
            cexec_ms.append(float(m.group(1)))
            continue
        m = RE_PUB.search(line)
        if m:
            toks = [int(x) for x in m.group(2).split(",") if x.strip()]
            pub_tokens.append(toks)
            pub_t.append(t)
            continue
        if RE_RANGE.search(line):
            range_t.append(t)

    disp_t: list[float] = []
    disp_tokens: list[int] = []
    emb_queue: list[int] = []
    step_c2e: list[int] = []
    step_t: list[float] = []
    deliv_rows: list[int] = []
    for t, line in _parse(edge_path):
        m = RE_DISP_EMB.search(line)
        if m:
            disp_t.append(t)
            disp_tokens.append(int(m.group(1)))
            emb_queue.append(int(m.group(2)))
            step_c2e.append(int(m.group(3)))
            continue
        m = RE_STEP.search(line)
        if m:
            step_t.append(t)
            emb_queue.append(int(m.group(4)))
            step_c2e.append(int(m.group(5)))
            continue
        m = RE_DELIVER.search(line)
        if m and m.group(2).strip():
            deliv_rows.append(len(
                [x for x in m.group(2).split(",") if x.strip()]))

    # ① 相位构成与 decode 稀释
    n_by = {}
    for _, ph, _, _ in phases:
        n_by[ph] = n_by.get(ph, 0) + 1
    dec_ts = [t for t, ph, _, _ in phases if ph == "DECODE"]
    dec_gaps_ms = [
        (b - a) * 1000 for a, b in zip(dec_ts, dec_ts[1:]) if b > a]
    # 相邻 PREFILL 串(连续 prefill 步合并)之间的 decode 步数(穿插密度)
    interleave: list[int] = []
    run = 0
    seen_first = False
    in_prefill_run = False
    for _, ph, _, _ in phases:
        if ph == "PREFILL":
            if seen_first and not in_prefill_run:
                interleave.append(run)
            seen_first = True
            in_prefill_run = True
            run = 0
        else:
            in_prefill_run = False
            if ph == "DECODE":
                run += 1
    empty_starved = sum(
        1 for _, ph, _, pend in phases if ph == "EMPTY" and pend > 0)
    rep["phase"] = {
        "n": len(phases), "by": n_by,
        "decode_gaps_ms": dec_gaps_ms, "interleave": interleave,
        "empty_starved": empty_starved,
        "span_s": (phases[0][0], phases[-1][0]) if phases else None,
    }

    # ② embed 准入
    rep["admit"] = {
        "disp_rate": _rate(disp_t), "range_rate": _rate(range_t),
        "n_disp": len(disp_t), "n_range": len(range_t),
        "chunk_tokens_p50": _pct(disp_tokens, 50),
        "emb_queue_p95": _pct(emb_queue, 95),
        "c2e_pending_p95": _pct(step_c2e, 95),
    }

    # ③ 产出效率与接受率
    flat = [x for row in pub_tokens for x in row]
    spec_span = max((pub_t[-1] - pub_t[0]) for pub_t in [pub_t]) \
        if len(pub_t) > 1 else float("nan")
    rep["yield"] = {
        "steps": len(pub_tokens), "reqs_step_p50": _pct(
            [len(r) for r in pub_tokens], 50),
        "tok_flat": flat,
        "tok_per_req_p50": _pct(flat, 50), "tok_per_req_p95": _pct(flat, 95),
        "max_tok": max(flat) if flat else 0,
        "total_tokens": sum(flat) if flat else 0,
        "tok_per_s": (sum(flat) / spec_span) if spec_span == spec_span
        and spec_span > 0 else float("nan"),
        "decode_rate": _rate(dec_ts),
    }
    rep["exec"] = {
        "step_exec_p50": _pct(exec_ms, 50), "step_exec_p95": _pct(exec_ms, 95),
        "cexec_p50": _pct(cexec_ms, 50), "cexec_p95": _pct(cexec_ms, 95),
    }
    rep["deliver_rows_p50"] = _pct(deliv_rows, 50)
    return rep


def _f(v, nd=1):
    return f"{v:.{nd}f}" if v == v else "n/a"


def print_report(rep: dict) -> None:
    ph = rep["phase"]
    print("=" * 72)
    print("① 相位构成与 decode 稀释(相位调度器税)")
    if ph["n"]:
        share = {k: f"{v * 100 / ph['n']:.0f}%" for k, v in ph["by"].items()}
        print(f"   步数={ph['n']}  构成={share}  "
              f"EMPTY且pending>0(引擎有活没吃)={ph['empty_starved']} 步")
        g = ph["decode_gaps_ms"]
        it = ph["interleave"]
        print(f"   decode 间隔 ms: p50={_f(_pct(g, 50))} "
              f"p95={_f(_pct(g, 95))} max={_f(max(g)) if g else 'n/a'}")
        print(f"   相邻 PREFILL 串间 decode 步数: min={min(it) if it else 'n/a'} "
              f"p50={_f(_pct(it, 50))}")
    else:
        print("   无 cloud step 相位日志")
    a = rep["admit"]
    print("② embed 准入(prefill 输入绕行链)")
    print(f"   边 chunk 派发 {_f(a['disp_rate'], 2)}/s (n={a['n_disp']});"
          f" 云 RangeNotify 到达 {_f(a['range_rate'], 2)}/s (n={a['n_range']})")
    print(f"   chunk tokens p50={_f(a['chunk_tokens_p50'], 0)}  "
          f"emb 队列水位 p95={_f(a['emb_queue_p95'], 0)}  "
          f"c2e_pending p95={_f(a['c2e_pending_p95'], 0)}")
    y = rep["yield"]
    print("③ 产出效率与 MTP 接受率")
    if y["steps"]:
        k1 = y["max_tok"] or 1
        flat = y["tok_flat"]
        acc = (sum(flat) / len(flat)) / k1 if flat and k1 else float("nan")
        print(f"   步数={y['steps']}  每步请求数 p50={_f(y['reqs_step_p50'], 0)}  "
              f"每请求 token/步 均值={_f(sum(flat) / len(flat), 2)} "
              f"p50={_f(y['tok_per_req_p50'], 2)} "
              f"p95={_f(y['tok_per_req_p95'], 2)} (max={k1}→接受率≈"
              f"{_f(acc * 100, 0)}%)")
        print(f"   总 token={y['total_tokens']}  产出 {_f(y['tok_per_s'], 1)} "
              f"tok/s  decode 步频 {_f(y['decode_rate'], 2)}/s")
    e = rep["exec"]
    print("④ 步耗时")
    print(f"   引擎步 exec p50={_f(e['step_exec_p50'])}ms "
          f"p95={_f(e['step_exec_p95'])}ms  前向 cexec p50="
          f"{_f(e['cexec_p50'])}ms (exec≫cexec=引擎在等待/调度)")
    print("⑤ 判定")
    verdicts = []
    if ph["n"]:
        g95, g50 = _pct(ph["decode_gaps_ms"], 95), _pct(
            ph["decode_gaps_ms"], 50)
        prefill_share = ph["by"].get("PREFILL", 0) / ph["n"]
        if g95 - g50 > 200 or prefill_share > 0.3:
            verdicts.append(
                f"decode 间隔被拉长(p50 {_f(g50)}→p95 {_f(g95)}ms,"
                f"PREFILL 占比 {prefill_share * 100:.0f}%)→"
                f"【相位稀释显著】prefill 步在挤占 decode 步进")
        if ph["empty_starved"] > ph["n"] * 0.05:
            verdicts.append(
                f"EMPTY且pending>0 共 {ph['empty_starved']} 步→引擎有活没吃,"
                f"存在停摆/等待段")
    if a["disp_rate"] == a["disp_rate"] and a["range_rate"] == a[
            "range_rate"] and a["range_rate"] > a["disp_rate"] * 1.2:
        verdicts.append(
            f"RangeNotify 到达({a['range_rate']:.1f}/s)>边派发"
            f"({a['disp_rate']:.1f}/s)→【embed 准入积压】prefill 在边侧排队")
    if y["steps"] and y["max_tok"]:
        acc = (sum(y["tok_flat"]) / len(y["tok_flat"])) / y["max_tok"]
        if acc < 0.4:
            verdicts.append(
                f"接受率≈{acc * 100:.0f}% 偏低→MTP 在 bench 批组合下"
                f"产出打折,对照集中式接受率核实")
    if not verdicts:
        verdicts.append("三个嫌疑指标均不显著——扩大样本或贴原始日志人工复核")
    for v in verdicts:
        print(f"   • {v}")
    print("=" * 72)


def selftest() -> int:
    import tempfile
    from pathlib import Path
    fmt = "%Y-%m-%d %H:%M:%S,%f"

    def st(base, i):
        return (base + _dt.timedelta(milliseconds=i)).strftime(fmt)[:-3]

    t0 = _dt.datetime(2026, 9, 18, 12, 0, 0)
    cloud, edge = [], []
    t = 0.0
    sn = 0
    for cycle in range(60):          # 每 10 个 decode 步插 2 个 prefill 步
        for j in range(10):
            cloud.append(
                f"{st(t0, int(t * 1000))} [sched] [Lwd][sched] cloud "
                f"step={sn} phase=DECODE seqno={j} "
                f"reqs=[a,b] tokens=8 pending_notify=0 decode_ready=2")
            sn += 1
            cloud.append(
                f"{st(t0, int(t * 1000))} [perf] [Lwd][perf] "
                f"cloud-step exec=60.00ms")
            cloud.append(
                f"{st(t0, int(t * 1000))} [ctrl] [Lwd][cloud-ctrl] publish "
                f"c2e(token-id): reqs=2 tokens=[4, 1]")
            t += 0.06
        for j in range(2):
            cloud.append(
                f"{st(t0, int(t * 1000))} [sched] [Lwd][sched] cloud "
                f"step={sn} phase=PREFILL seqno=p{j} reqs=[c] tokens=4096 "
                f"pending_notify=1 decode_ready=2")
            sn += 1
            t += 0.25
        cloud.append(
            f"{st(t0, int(t * 1000))} [ctrl] [Lwd][cloud-ctrl] RangeNotify "
            f"req=c num=4096 seqno=1")
    for i in range(200):
        edge.append(
            f"{st(t0, i * 50)} [sched] [Lwd][sched] edge dispatch-embed "
            f"seqno={i} req=r tokens=2048 queue=3emb c2e_pending=0")
        edge.append(
            f"{st(t0, i * 50)} [sched] [Lwd][sched] edge step "
            f"harvest_emb=1 deliver_unemb=1 disp_emb=1 queue=3emb "
            f"c2e_pending=0")
    with tempfile.TemporaryDirectory() as d:
        cf, ef = Path(d) / "c.log", Path(d) / "e.log"
        cf.write_text("\n".join(cloud) + "\n")
        ef.write_text("\n".join(edge) + "\n")
        rep = analyze(str(cf), str(ef))
        assert rep["phase"]["by"].get("DECODE") == 600, rep["phase"]["by"]
        assert rep["phase"]["by"].get("PREFILL") == 120
        g95 = _pct(rep["phase"]["decode_gaps_ms"], 95)
        assert g95 > 400, f"prefill 穿刺应拉大 decode 间隔 p95={g95}"
        assert rep["phase"]["interleave"].count(10) == 59
        assert 0 not in rep["phase"]["interleave"]
        _flat = rep["yield"]["tok_flat"]
        assert sum(_flat) / len(_flat) == 2.5
        assert rep["admit"]["disp_rate"] > 15
        print_report(rep)
    print("SELFTEST PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--cloud")
    ap.add_argument("--edge")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.cloud or not args.edge:
        ap.error("--cloud 与 --edge 均需提供")
    print_report(analyze(args.cloud, args.edge))
    return 0


if __name__ == "__main__":
    sys.exit(main())
