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
  ③ MTP 接受率崩(重点核查 draft 首趟兜底污染):
     请求扰动下 draft 首趟 LWD embeds 路径塌回 token-id 兜底,prompt
     KV 被污染→接受率持续低迷;单请求正常而 bench 崩掉即指向此处。

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

流式解析:不缓存日志原文(GB 级日志不爆内存),进度打 stderr;
报告头部打印各锚点命中数,全零时提示日志版本不匹配。

用法:
  python lwd_bench_gap_probe.py --cloud 云日志 --edge 边日志
  python lwd_bench_gap_probe.py --selftest
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import sys

RE_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.](\d{3})")

RE_PHASE = re.compile(
    r"\[Lwd\]\[sched\] cloud step=\d+ phase=(\w+) seqno=\S+ "
    r"reqs=\[.*?\] tokens=\d+ pending_notify=(\d+) decode_ready=\d+")
RE_EXEC = re.compile(r"\[Lwd\]\[perf\] cloud-step exec=([\d.]+)ms")
RE_CEXEC = re.compile(r"\[Lwd\]\[perf\] cloud-exec total=([\d.]+)ms")
RE_PUB = re.compile(
    r"\[Lwd\]\[cloud-ctrl\] publish c2e\(token-id\): reqs=(\d+) "
    r"tokens=\[([\d, ]*)\]")
RE_RANGE = re.compile(r"\[Lwd\]\[cloud-ctrl\] RangeNotify req=")
RE_DISP_EMB = re.compile(
    r"\[Lwd\]\[sched\] edge dispatch-embed seqno=\d+ req=\S+ tokens=(\d+) "
    r"queue=(\d+)emb c2e_pending=(\d+)")
RE_STEP = re.compile(
    r"\[Lwd\]\[sched\] edge step harvest_emb=\d+ deliver_unemb=\d+ "
    r"disp_emb=\d+ queue=(\d+)emb c2e_pending=(\d+)")
RE_DELIVER = re.compile(
    r"\[Lwd\]\[sched\] edge deliver-unembed reqs=(\d+) tokens=\[([\d, ]*)\]")

_PROGRESS_EVERY = 5_000_000


def _sec(ts: str, ms: str) -> float:
    return _dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp() \
        + int(ms) / 1000.0


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


def _scan(path: str, which: str) -> dict:
    """单遍流式扫描:逐行提取字段后立即丢弃原文。"""
    phase: list[tuple[float, str, int]] = []    # (t, phase, pending_notify)
    exec_ms: list[float] = []
    cexec_ms: list[float] = []
    pub: list[tuple[float, list[int]]] = []
    range_t: list[float] = []
    disp: list[tuple[float, int, int, int]] = []  # (t, tokens, queue, c2e)
    step_q: list[tuple[int, int]] = []            # (queue, c2e)
    deliv_rows: list[int] = []
    n = hits = 0
    size = os.path.getsize(path) if os.path.exists(path) else 0
    print(f"[probe] 解析{which}日志 {path} ({size / 1e6:.1f} MB)...",
          file=sys.stderr, flush=True)
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n += 1
            if n % _PROGRESS_EVERY == 0:
                print(f"[probe]   {which}: 已扫描 {n:,} 行, 命中 {hits:,}",
                      file=sys.stderr, flush=True)
            if "[Lwd]" not in line:
                continue  # 快速预过滤:绝大多数行非锚点,跳过全部正则
            m = RE_TS.match(line)
            if not m:
                continue
            t = _sec(m.group(1), m.group(2))
            if which == "云":
                mm = RE_PHASE.search(line)
                if mm:
                    phase.append((t, mm.group(1), int(mm.group(2))))
                    hits += 1
                    continue
                mm = RE_EXEC.search(line)
                if mm:
                    exec_ms.append(float(mm.group(1)))
                    hits += 1
                    continue
                mm = RE_CEXEC.search(line)
                if mm:
                    cexec_ms.append(float(mm.group(1)))
                    hits += 1
                    continue
                mm = RE_PUB.search(line)
                if mm:
                    pub.append((t, [
                        int(x) for x in mm.group(2).split(",") if x.strip()]))
                    hits += 1
                    continue
                if RE_RANGE.search(line):
                    range_t.append(t)
                    hits += 1
            else:
                mm = RE_DISP_EMB.search(line)
                if mm:
                    disp.append((t, int(mm.group(1)), int(mm.group(2)),
                                 int(mm.group(3))))
                    hits += 1
                    continue
                mm = RE_STEP.search(line)
                if mm:
                    step_q.append((int(mm.group(1)), int(mm.group(2))))
                    hits += 1
                    continue
                mm = RE_DELIVER.search(line)
                if mm and mm.group(2).strip():
                    deliv_rows.append(len(
                        [x for x in mm.group(2).split(",") if x.strip()]))
                    hits += 1
    print(f"[probe]   {which}: 完成 {n:,} 行, 命中 {hits:,}",
          file=sys.stderr, flush=True)
    return {
        "phase": phase, "exec_ms": exec_ms, "cexec_ms": cexec_ms,
        "pub": pub, "range_t": range_t, "disp": disp, "step_q": step_q,
        "deliv_rows": deliv_rows,
        "counts": (
            {"cloud-phase": len(phase), "cloud-exec": len(exec_ms),
             "cloud-cexec": len(cexec_ms), "cloud-publish": len(pub),
             "cloud-RangeNotify": len(range_t)}
            if which == "云" else
            {"edge-dispatch": len(disp), "edge-step": len(step_q),
             "edge-deliver": len(deliv_rows)}
        ),
    }


def analyze(cloud_path: str, edge_path: str) -> dict:
    c = _scan(cloud_path, "云")
    e = _scan(edge_path, "边")
    rep: dict = {"counts": c["counts"] | e["counts"]}

    phases = c["phase"]
    n_by: dict[str, int] = {}
    for _, ph, _ in phases:
        n_by[ph] = n_by.get(ph, 0) + 1
    dec_ts = [t for t, ph, _ in phases if ph == "DECODE"]
    gaps = [(b - a) * 1000 for a, b in zip(dec_ts, dec_ts[1:]) if b > a]
    # 相邻 PREFILL 串(连续 prefill 步合并)之间的 decode 步数
    interleave: list[int] = []
    run = 0
    seen_first = False
    in_run = False
    for _, ph, _ in phases:
        if ph == "PREFILL":
            if seen_first and not in_run:
                interleave.append(run)
            seen_first = True
            in_run = True
            run = 0
        else:
            in_run = False
            if ph == "DECODE":
                run += 1
    empty_starved = sum(1 for _, ph, pend in phases
                        if ph == "EMPTY" and pend > 0)
    rep["phase"] = {
        "n": len(phases), "by": n_by, "gaps": gaps,
        "interleave": interleave, "empty_starved": empty_starved,
    }

    disp_t = [t for t, _, _, _ in e["disp"]]
    disp_tokens = [tk for _, tk, _, _ in e["disp"]]
    emb_q = [q for _, _, q, _ in e["disp"]] + [q for q, _ in e["step_q"]]
    c2e_q = [p for _, _, _, p in e["disp"]] + [p for _, p in e["step_q"]]
    rep["admit"] = {
        "disp_rate": _rate(disp_t), "range_rate": _rate(c["range_t"]),
        "n_disp": len(disp_t), "n_range": len(c["range_t"]),
        "chunk_tokens_p50": _pct(disp_tokens, 50),
        "emb_queue_p95": _pct(emb_q, 95), "c2e_p95": _pct(c2e_q, 95),
    }

    flat = [x for _, row in c["pub"] for x in row]
    span = (c["pub"][-1][0] - c["pub"][0][0]) if len(c["pub"]) > 1 \
        else float("nan")
    rep["yield"] = {
        "steps": len(c["pub"]),
        "reqs_step_p50": _pct([len(row) for _, row in c["pub"]], 50),
        "flat": flat, "p50": _pct(flat, 50), "p95": _pct(flat, 95),
        "max": max(flat) if flat else 0, "total": sum(flat) if flat else 0,
        "tok_per_s": (sum(flat) / span) if span == span and span > 0
        else float("nan"),
        "decode_rate": _rate(dec_ts),
    }
    rep["exec"] = {
        "p50": _pct(c["exec_ms"], 50), "p95": _pct(c["exec_ms"], 95),
        "cp50": _pct(c["cexec_ms"], 50), "cp95": _pct(c["cexec_ms"], 95),
    }
    return rep


def _f(v, nd=1):
    return f"{v:.{nd}f}" if v == v else "n/a"


def print_report(rep: dict) -> None:
    print("=" * 72, flush=True)
    print("锚点命中: " + "  ".join(
        f"{k}={v}" for k, v in rep["counts"].items()), flush=True)
    if not any(rep["counts"].values()):
        print("!! 两份日志没有任何锚点命中——确认日志来自 token_id 版分支"
              "(prefill_only_v0.1_mtp_perf_ww)的部署", flush=True)
        return
    ph = rep["phase"]
    print("① 相位构成与 decode 稀释(相位调度器税)")
    if ph["n"]:
        share = {k: f"{v * 100 / ph['n']:.0f}%" for k, v in ph["by"].items()}
        print(f"   步数={ph['n']}  构成={share}  "
              f"EMPTY且pending>0={ph['empty_starved']} 步")
        g, it = ph["gaps"], ph["interleave"]
        print(f"   decode 间隔 ms: p50={_f(_pct(g, 50))} "
              f"p95={_f(_pct(g, 95))} max={_f(max(g)) if g else 'n/a'}")
        print(f"   相邻 PREFILL 串间 decode 步数: "
              f"min={min(it) if it else 'n/a'} p50={_f(_pct(it, 50))}")
    else:
        print("   无 cloud step 相位日志")
    a = rep["admit"]
    print("② embed 准入(prefill 输入绕行链)")
    print(f"   边 chunk 派发 {_f(a['disp_rate'], 2)}/s (n={a['n_disp']});"
          f" 云 RangeNotify 到达 {_f(a['range_rate'], 2)}/s "
          f"(n={a['n_range']})")
    print(f"   chunk tokens p50={_f(a['chunk_tokens_p50'], 0)}  "
          f"emb 队列 p95={_f(a['emb_queue_p95'], 0)}  "
          f"c2e_pending p95={_f(a['c2e_p95'], 0)}")
    y = rep["yield"]
    print("③ 产出效率与 MTP 接受率(重点:单请求 vs bench 对比)")
    if y["steps"]:
        k1 = y["max"] or 1
        mean = sum(y["flat"]) / len(y["flat"])
        acc = mean / k1
        print(f"   步数={y['steps']}  每步请求数 p50={_f(y['reqs_step_p50'], 0)}"
              f"  每请求 token/步 均值={_f(mean, 2)} p50={_f(y['p50'], 2)}"
              f" p95={_f(y['p95'], 2)} (max={k1}→接受率≈{_f(acc * 100, 0)}%)")
        print(f"   总 token={y['total']}  产出 {_f(y['tok_per_s'], 1)} tok/s"
              f"  decode 步频 {_f(y['decode_rate'], 2)}/s")
    else:
        print("   无 publish c2e(token-id) 日志(token_id 版才有)")
    e = rep["exec"]
    print("④ 步耗时")
    print(f"   引擎步 exec p50={_f(e['p50'])}ms p95={_f(e['p95'])}ms  "
          f"前向 cexec p50={_f(e['cp50'])}ms (exec≫cexec=在等待不在计算)")
    print("⑤ 判定")
    vs = []
    if ph["n"]:
        g95, g50 = _pct(ph["gaps"], 95), _pct(ph["gaps"], 50)
        pre = ph["by"].get("PREFILL", 0) / ph["n"]
        if g95 - g50 > 200 or pre > 0.3:
            vs.append(f"decode 间隔被拉长(p50 {_f(g50)}→p95 {_f(g95)}ms,"
                      f"PREFILL 占 {pre * 100:.0f}%)→【相位稀释显著】")
        if ph["empty_starved"] > ph["n"] * 0.05:
            vs.append(f"EMPTY且pending>0 共 {ph['empty_starved']} 步→"
                      f"引擎有活没吃,存在停摆/等待段")
    if a["disp_rate"] == a["disp_rate"] and a["range_rate"] == a[
            "range_rate"] and a["range_rate"] > a["disp_rate"] * 1.2:
        vs.append(f"RangeNotify({a['range_rate']:.1f}/s)>边派发"
                  f"({a['disp_rate']:.1f}/s)→【embed 准入积压】")
    if y["steps"] and y["max"]:
        acc = (sum(y["flat"]) / len(y["flat"])) / y["max"]
        if acc < 0.4:
            vs.append(f"接受率≈{acc * 100:.0f}% 偏低→【MTP 接受率崩】"
                      f"重点核查 draft 首趟兜底污染(单请求对照)")
    if not vs:
        vs.append("三个嫌疑指标均不显著——拿单请求日志跑同工具对比接受率")
    for v in vs:
        print(f"   • {v}")
    print("=" * 72, flush=True)


def selftest() -> int:
    import tempfile
    from pathlib import Path
    fmt = "%Y-%m-%d %H:%M:%S,%f"
    t0 = _dt.datetime(2026, 9, 18, 12, 0, 0)

    def st(ms):
        return (t0 + _dt.timedelta(milliseconds=ms)).strftime(fmt)[:-3]

    cloud, edge = [], []
    t = 0.0
    sn = 0
    for _cycle in range(60):
        for _j in range(10):
            cloud.append(
                f"{st(int(t * 1000))} [Lwd][sched] cloud step={sn} "
                f"phase=DECODE seqno={sn} reqs=[a,b] tokens=8 "
                f"pending_notify=0 decode_ready=2")
            sn += 1
            cloud.append(f"{st(int(t * 1000))} [Lwd][perf] "
                         f"cloud-step exec=60.00ms")
            cloud.append(f"{st(int(t * 1000))} [Lwd][cloud-ctrl] publish "
                         f"c2e(token-id): reqs=2 tokens=[4, 1]")
            t += 0.06
        for j in range(2):
            cloud.append(
                f"{st(int(t * 1000))} [Lwd][sched] cloud step={sn} "
                f"phase=PREFILL seqno=p{j} reqs=[c] tokens=4096 "
                f"pending_notify=1 decode_ready=2")
            sn += 1
            t += 0.25
        cloud.append(f"{st(int(t * 1000))} [Lwd][cloud-ctrl] RangeNotify "
                     f"req=c num=4096 seqno=1")
    for i in range(200):
        edge.append(
            f"{st(i * 50)} [Lwd][sched] edge dispatch-embed seqno={i} "
            f"req=r tokens=2048 queue=3emb c2e_pending=0")
        edge.append(
            f"{st(i * 50)} [Lwd][sched] edge step harvest_emb=1 "
            f"deliver_unemb=1 disp_emb=1 queue=3emb c2e_pending=0")
    with tempfile.TemporaryDirectory() as d:
        cf, ef = Path(d) / "c.log", Path(d) / "e.log"
        cf.write_text("\n".join(cloud) + "\n")
        ef.write_text("\n".join(edge) + "\n")
        rep = analyze(str(cf), str(ef))
        assert rep["phase"]["by"].get("DECODE") == 600, rep["counts"]
        assert rep["phase"]["by"].get("PREFILL") == 120
        assert _pct(rep["phase"]["gaps"], 95) > 400
        assert rep["phase"]["interleave"].count(10) == 59
        assert 0 not in rep["phase"]["interleave"]
        flat = rep["yield"]["flat"]
        assert sum(flat) / len(flat) == 2.5
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
