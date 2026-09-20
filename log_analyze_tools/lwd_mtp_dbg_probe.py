#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""lwd_mtp_dbg_probe: MTP 接受率分桶 + 首步投机判定工具(v2)。

输入:云侧日志(单文件,编码自适应:UTF-8 / UTF-16 LE/BE,
带或不带 BOM——PowerShell 重定向日志常为 UTF-16)。接受数不读
c2e 报告值,而用 ``[lwd-finish-dbg]`` 的 out_len 台账差推导
(相邻步 out_len 之差 = 该步真实前进的 token 数),可独立校验
c2e 报告口径。

依赖的日志锚点(全部为现有日志,无新增):

  [Lwd][cloud-sched] decode reqs=[..]                       — 纯 decode 步
  [Lwd][cloud-sched] prefill notify req=R seqno=S num=N     — 纯 prefill 步
  [lwd-finish-dbg] req=R out_len=N last_tok=T finish=F      — 每请求每步台账
  (兼容旧版 [Lwd][sched] cloud step=N phase=P reqs=[..] tokens=T)

台账口径说明(finish-dbg 对"在批全体请求"记账,非仅本步调度者):
  - prefill 步 delta=0:请求在批但本步未推进 —— 正常;
  - delta > max_accept:该请求某步行缺失,下次记录两步并一笔 ——
    口径噪声,从分桶剔除并单独计数;
  - decode 步 delta<=0:真实异常,逐条列出。

输出:
  1. 判决表:first(from-prefill) 与 fresh/aged × gen-after-decode/
     gen-after-prefill,n/均值/p50/极值/直方/全满占比/退化告警
  2. 首个 decode 步专项:delta 全 1 ⇒ 首步不投机(账本实证);
     旧格式日志另有 tokens 贡献推算(1=没排草稿,4=排了被拒)
  3. 首个真验证步(第二个 decode 步)专项:delta 分布 + last_tok
     重复度——恒 4 且 last_tok 高度重复 ⇒ 样板句良性;恒 4 但
     last_tok 分散 ⇒ 疑点(草稿/目标共享错位,建议 token diff)
  4. 按 decode 位置的前 N 步接受率剖面(首块是否特殊一目了然)
  5. 账本质检(三类口径计数 + 真实异常明细)
  6. 判决汇总(自动,含退化桶告警)

用法:
  python lwd_mtp_dbg_probe.py cloud0.log [--top 3] [--max-accept 4]
  python lwd_mtp_dbg_probe.py cloud0.log --req chatcmpl-xxxx   # 单请求时间线
  python lwd_mtp_dbg_probe.py --selftest
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict

RE_DECODE = re.compile(r"\[Lwd\]\[cloud-sched\] decode reqs=\[([^\]]*)\]")
RE_PREFILL = re.compile(
    r"\[Lwd\]\[cloud-sched\] prefill notify req=(\S+) seqno=(\S+) num=(\d+)")
RE_STEP_OLD = re.compile(
    r"\[Lwd\]\[sched\] cloud step=\d+ phase=(PREFILL|DECODE|EMPTY) "
    r"seqno=\S+ reqs=\[([^\]]*)\].*?(?:tokens=(\d+))?")
RE_FINISH = re.compile(
    r"\[lwd-finish-dbg\] req=(\S+) out_len=(\d+) last_tok=(\S+) finish=(\S+)")

BUCKET_ORDER = (
    "first(from-prefill)",
    "fresh/gen-after-decode",
    "fresh/gen-after-prefill",
    "aged/gen-after-decode",
    "aged/gen-after-prefill",
)


def _parse_reqs(raw: str) -> list[str]:
    return [x.strip().strip("'\"") for x in raw.split(",") if x.strip()]


def detect_encoding(path: str) -> str:
    """按 BOM / 空字节密度自适应编码(PS 重定向日志常为 UTF-16)。"""
    with open(path, "rb") as f:
        head = f.read(8192)
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if b"\x00" in head:
        # 无 BOM 的 UTF-16:按可打印率选端序
        best, best_score = "utf-8", -1.0
        for enc in ("utf-16-le", "utf-16-be"):
            text = head.decode(enc, errors="ignore")
            printable = sum(
                ch.isprintable() or ch in "\r\n\t" for ch in text)
            score = printable / max(len(text), 1)
            if score > best_score:
                best, best_score = enc, score
        if best_score > 0.7:
            return best
    return "utf-8"


def parse_log(path: str, encoding: str | None = None) -> list[dict]:
    """流式解析:步事件(sched 行)+ 步末台账(finish 行挂到当前步)。"""
    if encoding is None:
        encoding = detect_encoding(path)
    steps: list[dict] = []
    cur: dict | None = None
    counts = Counter()
    with open(path, encoding=encoding, errors="replace") as f:
        for line in f:
            if "[Lwd]" not in line and "lwd-finish-dbg" not in line:
                continue
            m = RE_DECODE.search(line)
            if m:
                cur = {"kind": "DECODE", "reqs": _parse_reqs(m.group(1)),
                       "tokens": None, "finish": {}}
                steps.append(cur)
                counts["sched-decode"] += 1
                continue
            m = RE_PREFILL.search(line)
            if m:
                cur = {"kind": "PREFILL", "reqs": [m.group(1)],
                       "tokens": int(m.group(3)), "finish": {}}
                steps.append(cur)
                counts["sched-prefill"] += 1
                continue
            m = RE_STEP_OLD.search(line)
            if m:
                kind, reqs, tokens = m.group(1), m.group(2), m.group(3)
                cur = {"kind": kind, "reqs": _parse_reqs(reqs),
                       "tokens": int(tokens) if tokens else None, "finish": {}}
                steps.append(cur)
                counts[f"sched-old-{kind.lower()}"] += 1
                continue
            m = RE_FINISH.search(line)
            if m:
                counts["finish-dbg"] += 1
                if cur is not None:
                    cur["finish"][m.group(1)] = (
                        int(m.group(2)), m.group(3), m.group(4))
    if counts["sched-decode"] + counts["sched-prefill"] == 0 \
            and counts["sched-old-decode"] + counts["sched-old-prefill"] == 0:
        raise SystemExit(
            "[probe] 未匹配到任何调度步锚点。期望日志包含 "
            "[Lwd][cloud-sched] decode/prefill 或旧版 [Lwd][sched] cloud step= 行")
    return steps


def build_ledger(steps: list[dict]) -> dict[str, list[dict]]:
    """每请求事件序列(只含有台账的步),delta=相邻 out_len 差。"""
    events: dict[str, list[dict]] = defaultdict(list)
    for idx, st in enumerate(steps):
        for req, (out_len, last_tok, _fin) in st["finish"].items():
            events[req].append({
                "step": idx, "kind": st["kind"], "out_len": out_len,
                "last_tok": last_tok, "delta": None})
    for evs in events.values():
        prev = 0
        for ev in evs:
            ev["delta"] = ev["out_len"] - prev
            prev = ev["out_len"]
    return events


def bucketize(events: dict[str, list[dict]], steps: list[dict],
              max_accept: int) -> tuple[dict[str, list[int]], int]:
    """判决表分桶:position1=first;其余按 生成步位置 × 中间是否隔 prefill。

    delta>max_accept 是台账缺口(某步行缺失,下次把两步加在一起),
    不算单步接受数:剔除并计数返回。
    """
    prefill_steps = {i for i, st in enumerate(steps) if st["kind"] == "PREFILL"}
    buckets: dict[str, list[int]] = {name: [] for name in BUCKET_ORDER}
    gaps = 0
    for req, evs in events.items():
        dec = [e for e in evs if e["kind"] == "DECODE"]
        for pos, ev in enumerate(dec, start=1):
            if ev["delta"] > max_accept:
                gaps += 1
                continue
            if pos == 1:
                buckets["first(from-prefill)"].append(ev["delta"])
                continue
            gen_after_prefill = "gen-after-prefill" if pos == 2 \
                else "gen-after-decode"
            gen_step = dec[pos - 2]["step"]
            aged = any(gen_step < p < ev["step"] for p in prefill_steps)
            key = f"{'aged' if aged else 'fresh'}/{gen_after_prefill}"
            buckets[key].append(ev["delta"])
    return buckets, gaps


def _p50(vals: list[int]) -> int:
    s = sorted(vals)
    return s[len(s) // 2] if s else 0


def _hist(vals: list[int]) -> str:
    total = len(vals)
    return " ".join(
        f"{k}:{v}({v * 100.0 / total:.1f}%)"
        for k, v in sorted(Counter(vals).items()))


def print_buckets(buckets: dict[str, list[int]], top: int) -> None:
    print("分桶(每请求每步真实前进数=out_len 台账差)——判决表:")
    print("  fresh=生成与消费紧邻;aged=中间隔了其他请求的 prefill 步")
    print("  gen-after-prefill=草稿生成于该请求首个 decode 步(紧跟其 prefill)")
    for name in BUCKET_ORDER:
        vals = buckets[name]
        if not vals:
            print(f"{name:<28} n=    0")
            continue
        full = Counter(vals).most_common(1)[0]
        degen = "  <<< 退化分布(min==max)!" if min(vals) == max(vals) \
            and len(vals) >= 20 else ""
        print(f"{name:<28} n={len(vals):>6} 均值={sum(vals) / len(vals):.2f} "
              f"p50={_p50(vals)} min={min(vals)} max={max(vals)} "
              f"全{full[0]}占比={full[1] * 100.0 / len(vals):.1f}%{degen}")
        if top > 0:
            print(f"{'':>30}直方: {_hist(vals)}")


def probe_first_decode(events: dict[str, list[dict]], top: int,
                       max_accept: int) -> list[int]:
    """专项一:首个 decode 步。delta 全 1 ⇒ 首步无草稿(账本实证)。"""
    deltas = []
    for evs in events.values():
        dec = [e for e in evs if e["kind"] == "DECODE"]
        if dec and dec[0]["delta"] <= max_accept:
            deltas.append(dec[0]["delta"])
    print(f"\n专项一 首个 decode 步(prefill 后第一步) n={len(deltas)} "
          f"直方: {_hist(deltas)}")
    if deltas and set(deltas) == {1}:
        print("判决:全部请求首步只前进 1 token ⇒ 首步不参与投机"
              "(账本层面实证,与 c2e 报告口径一致)")
    else:
        print("判决:首步存在 >1 的前进 ⇒ 首步有草稿且被接受,与结构性 1 假设不符")
    return deltas


def probe_first_verify(events: dict[str, list[dict]], top: int,
                       max_accept: int) -> None:
    """专项二:首个真验证步(第二个 decode 步)恒满是否良性。"""
    deltas, toks = [], []
    for evs in events.values():
        dec = [e for e in evs if e["kind"] == "DECODE"]
        if len(dec) >= 2 and dec[1]["delta"] <= max_accept:
            deltas.append(dec[1]["delta"])
            toks.append(dec[1]["last_tok"])
    print(f"\n专项二 首个真验证步(第二个 decode 步) n={len(deltas)} "
          f"直方: {_hist(deltas)}")
    if not deltas:
        print("判决:样本不足")
        return
    tok_counter = Counter(toks)
    distinct = len(tok_counter)
    tops = ", ".join(f"{t!r}x{c}" for t, c in tok_counter.most_common(top))
    print(f"  last_tok 去重 {distinct}/{len(toks)}(top: {tops})")
    if set(deltas) == {4} and len(deltas) >= 20:
        if distinct <= max(3, len(toks) // 100):
            print("判决:恒 4 且 last_tok 高度重复 ⇒ 开头样板句+贪心的良性确定性"
                  "(建议抽查 [lwd-token-dbg] 文本通顺性收尾)")
        else:
            print("判决:恒 4 但 last_tok 分散 ⇒ 草稿逐一命中不可用样板解释,"
                  "疑点:草稿与目标共享同一错位上下文——跑 lwd_token_diff "
                  "对集中式基线定性")
    elif set(deltas) == {4}:
        print("判决:恒 4(样本少,弱判定)")
    else:
        print("判决:非退化分布,常态验证步")


def probe_positions(events: dict[str, list[dict]], max_accept: int) -> None:
    """按 decode 位置的接受率剖面:首块是否特殊一目了然。"""
    pos_sum: dict[int, list[int]] = defaultdict(list)
    for evs in events.values():
        for pos, ev in enumerate(
                (e for e in evs if e["kind"] == "DECODE"), start=1):
            if ev["delta"] <= max_accept:
                pos_sum[min(pos, 10)].append(ev["delta"])
    print("\n位置剖面(decode 第 N 步的均值,>=10 合并):")
    print("  " + "  ".join(
        f"{p}:{sum(v) / len(v):.2f}(n={len(v)})"
        for p, v in sorted(pos_sum.items())))


def probe_ledger_anomalies(events: dict[str, list[dict]],
                           max_accept: int) -> list[str]:
    """账本质检(三类口径 + 真实异常,口径见模块 docstring)。"""
    idle, gaps, bad = 0, 0, []
    for req, evs in events.items():
        for ev in evs:
            if ev["delta"] > max_accept:
                gaps += 1
            elif ev["kind"] == "PREFILL" and ev["delta"] == 0:
                idle += 1
            elif ev["kind"] == "DECODE" and ev["delta"] <= 0:
                bad.append(f"decode步delta={ev['delta']} req={req} "
                           f"step={ev['step']}")
    print(f"\n账本质检: 在批未推进(prefill步delta=0)={idle} "
          f"台账缺口(delta>{max_accept}两步并一笔)={gaps} "
          f"真实异常(decode步delta<=0)={len(bad)} 条")
    for line in bad[:5]:
        print(f"  {line}")
    return bad


def probe_token_contribution(steps: list[dict]) -> None:
    """旧格式 tokens=T 时:含首 decode 请求的步,新请求 token 贡献推算。

    稳态 decode 请求每步排 1+num_spec(默认4)行;新请求贡献 = T-4*(n-1),
    贡献 1=没排草稿(调度侧丢),4=排了 3 条被拒/未回填(落地侧丢)。
    """
    sample = [st for st in steps
              if st["kind"] == "DECODE" and st["tokens"]
              and len(st["finish"]) > 1][:200]
    if not sample:
        print("\n(tokens=T 不可用或无混合批,跳过 token 贡献推算)")
        return
    contribs = Counter()
    for st in sample:
        contribs[st["tokens"] - 4 * (len(st["finish"]) - 1)] += 1
    print(f"\n首混合 decode 步的新请求 token 贡献推算(n={len(sample)}): "
          f"{_hist(list(contribs.elements()))}")
    print("  1=没排草稿;4=排了 3 条草稿但全被拒(占位未回填/垃圾草稿)")


def print_timeline(events: dict[str, list[dict]], req: str) -> None:
    print(f"\n请求 {req} 时间线:")
    for ev in events.get(req, []):
        print(f"  step={ev['step']:<5} {ev['kind']:<7} out_len={ev['out_len']:<6} "
              f"delta={ev['delta']:<3} last_tok={ev['last_tok']}")
    if req not in events:
        print("  (未在日志中找到该请求)")


def analyze(steps: list[dict], top: int, max_accept: int) -> None:
    kind_counts = Counter(st["kind"] for st in steps)
    events = build_ledger(steps)
    print(f"步数={len(steps)} 构成={dict(kind_counts)} "
          f"有台账请求={len(events)}")
    print("接受数来源:相邻步 out_len 差([lwd-finish-dbg] 台账),"
          "非 c2e 报告值")
    buckets, gaps = bucketize(events, steps, max_accept)
    print(f"台账缺口(delta>{max_accept},两步并一笔,已剔除): {gaps} 条")
    print_buckets(buckets, top)
    fresh, aged = buckets["fresh/gen-after-decode"], \
        buckets["aged/gen-after-decode"]
    if fresh and aged:
        diff = abs(sum(fresh) / len(fresh) - sum(aged) / len(aged))
        verdict = "无显著差异" if diff < 0.1 else f"差异 {diff:.2f}(注意)"
        print(f"\n判决 fresh vs aged(gen-after-decode): {verdict}")
    probe_first_decode(events, top, max_accept)
    probe_first_verify(events, top, max_accept)
    probe_positions(events, max_accept)
    probe_ledger_anomalies(events, max_accept)
    probe_token_contribution(steps)


def _selftest() -> None:
    import tempfile
    lines = [
        "2026-09-20 INFO [Lwd][cloud-sched] prefill notify req=R1 seqno=1 num=10",
        "INFO [lwd-finish-dbg] req=R1 out_len=1 last_tok=100 finish=None",
        "INFO [Lwd][cloud-sched] decode reqs=['R1']",
        "INFO [lwd-finish-dbg] req=R1 out_len=2 last_tok=101 finish=None",
        "INFO [Lwd][cloud-sched] decode reqs=['R1']",
        "INFO [lwd-finish-dbg] req=R1 out_len=6 last_tok=100 finish=None",
        "INFO [Lwd][cloud-sched] prefill notify req=R2 seqno=2 num=8",
        "INFO [lwd-finish-dbg] req=R2 out_len=1 last_tok=50 finish=None",
        "INFO [Lwd][cloud-sched] decode reqs=['R1', 'R2']",
        "INFO [lwd-finish-dbg] req=R1 out_len=10 last_tok=102 finish=None",
        "INFO [lwd-finish-dbg] req=R2 out_len=2 last_tok=51 finish=None",
        "INFO [Lwd][cloud-sched] decode reqs=['R1', 'R2']",
        "INFO [lwd-finish-dbg] req=R2 out_len=6 last_tok=100 finish=None",
        # R1 本步行缺失(台账缺口):下次记录时两步并一笔(delta=7)
        "INFO [Lwd][cloud-sched] prefill notify req=R3 seqno=3 num=5",
        "INFO [lwd-finish-dbg] req=R3 out_len=1 last_tok=60 finish=None",
        "INFO [Lwd][cloud-sched] decode reqs=['R1', 'R2']",
        "INFO [lwd-finish-dbg] req=R1 out_len=17 last_tok=104 finish=None",
        "INFO [lwd-finish-dbg] req=R2 out_len=9 last_tok=52 finish=None",
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        path = f.name
    steps = parse_log(path)
    events = build_ledger(steps)
    buckets, gaps = bucketize(events, steps, max_accept=4)
    assert gaps == 1, f"缺口样本未识别: {gaps}"
    assert buckets["first(from-prefill)"] == [1, 1], buckets
    assert sorted(buckets["fresh/gen-after-prefill"]) == [4, 4], buckets
    assert buckets["fresh/gen-after-decode"] == [], buckets
    # R1@4(4) 与 R2@7(3):gen 步与消费步之间都隔了 prefill
    assert sorted(buckets["aged/gen-after-decode"]) == [3, 4], buckets
    assert buckets["aged/gen-after-prefill"] == [], buckets
    # UTF-16(带 BOM,PowerShell 重定向形态)解析须逐字节等价
    with tempfile.NamedTemporaryFile(
            "w", suffix=".log", delete=False, encoding="utf-16") as f16:
        f16.write("\n".join(lines) + "\n")
        path16 = f16.name
    assert detect_encoding(path16) == "utf-16", "BOM 检测失败"
    steps16 = parse_log(path16)
    assert bucketize(build_ledger(steps16), steps16, max_accept=4) == (
        buckets, gaps), "UTF-16 解析结果与 UTF-8 不一致"
    print("[selftest] OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="LWD MTP 接受率分桶探针 v2")
    ap.add_argument("log", nargs="?", help="云侧日志路径")
    ap.add_argument("--req", help="只打印该请求的完整时间线")
    ap.add_argument("--top", type=int, default=3, help="last_tok top-K (默认 3)")
    ap.add_argument("--max-accept", type=int, default=4,
                    help="单步最大接受数(=1+num_spec,默认 4)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.log:
        ap.error("需要日志路径(或 --selftest)")
    enc = detect_encoding(args.log)
    print(f"[probe] 解析 {args.log}(编码 {enc})...")
    steps = parse_log(args.log, enc)
    if args.req:
        events = build_ledger(steps)
        print_timeline(events, args.req)
        return
    analyze(steps, args.top, args.max_accept)


if __name__ == "__main__":
    sys.exit(main())
