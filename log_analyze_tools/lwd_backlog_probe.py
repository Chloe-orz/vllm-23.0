#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""lwd_backlog_probe: c2e 积压/背压/调度饿死定位工具(单请求或压测后分析)。

输入:云日志 + 边日志(两份文件路径;时间戳各自独立解析,跨机钟差
敏感的指标会自动跳过)。依赖的日志锚点(全部为现有日志,无新增):

  云: [Lwd][cloud-ctrl] handle_model_output: c2e_meta received
      reqs=N down_seqno=S                    — 每 notify 一条(生产侧)
      [Lwd][perf] cloud-step dt=Xms          — 云步间隔(背压探针)
      [Lwd][perf] bridge-wait op=send ...    — 发送桥接等待(锁步探针)
      [Lwd][cloud-ctrl] RangeNotify req=R num=N seqno=S — UP 预告到达
      [Lwd][sched] cloud step=N phase=P seqno=S reqs=[..] tokens=T
      pending_notify=K decode_ready=D        — 每步调度批(⑤段核心)
  边: [Lwd][edge-worker] UNEMBED seqno=S ... — 每条处理一次(消费侧)
      [Lwd][perf] unembed seqno=S post_recv=A wait_tensor=B
      lm_head=C select=D total=Ems           — 边单条成本分段
      [Lwd][perf] harvest dur=Xms            — 引擎收割时长
      [Lwd][edge-ctrl] C2eNotify reqs=N down_seqno=S     — c2e 到达
      [Lwd][sched] edge dispatch-embed/-unembed / harvest / step
                                                — 每批派发/收割/步汇总

输出:
  1. 生产/消费速率对比(时钟无关,两边各自算间隔)
  2. 边单条成本分段排名(谁在吃时间)
  3. 云侧每步 LWD 税分段(对账 cloud-step dt)
  4. 尾巴检测:云结束后边"背靠背清账"段的条数与时长(时钟无关)
  5. 调度批重建+饿死检测(各自单机时钟):
       云:相位构成 / RangeNotify 到达→PREFILL 步延迟 / 其间插入的
          decode 步数 / EMPTY 且 pending>0(引擎有活没吃=停摆)
       边:EMBED 派发时前方压着的 UNEMBED 数(ahead_unemb)/ 队列滞留
          wait / c2e 到达→派发滞后 c2e_wait / c2e 水位高值
  6. 判定结论(边瓶颈/背压触发/prefill 被 decode 饿死/EMBED 被
     UNEMBED 挤住/引擎停摆)

用法:
  python lwd_backlog_probe.py --cloud 云日志 --edge 边日志 [--verbose]
  python lwd_backlog_probe.py --cloud 云日志 --edge 边日志 --timeline
      [--req 请求id]      # 人工核对调度顺序(按各自时钟分侧打印)
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
# ch= 后是枚举 str(LwdChannelType.UP) 含点号,[\w.+] 才能吃下
RE_BRIDGE = re.compile(
    r"\[Lwd\]\[perf\] bridge-wait op=(\w+) ch=([\w.]+) dur=([\d.]+)ms")
RE_EDGE_UNEMBED = re.compile(r"\[Lwd\]\[edge-worker\] UNEMBED seqno=(\d+)")
RE_EDGE_PERF = re.compile(
    r"\[Lwd\]\[perf\] unembed seqno=(\S+) post_recv=([\d.]+) "
    r"wait_tensor=([\d.]+) lm_head=([\d.]+) select=([\d.]+) "
    r"total=([\d.]+)ms")
RE_EMBED_PERF = re.compile(
    r"\[Lwd\]\[perf\] embed seqno=(\S+) forward=([\d.]+) "
    r"submit_send=([\d.]+) total=([\d.]+)ms")
RE_UP_RECV = re.compile(
    r"\[Lwd\]\[perf\] up-recv seqno=(\S+) ready_at_take=(\w+)")
RE_RANK = re.compile(r"\[Lwd\]\[perf\] rank-calc rows=(\d+) dur=([\d.]+)ms")
RE_PACK = re.compile(
    r"\[Lwd\]\[perf\] step-pack collect=([\d.]+) pack=([\d.]+) rows=(\d+)")
RE_DOWN_SEND = re.compile(
    r"\[Lwd\]\[perf\] down-send seqno=(\d+) submit=([\d.]+)ms")
RE_PUBLISH = re.compile(
    r"\[Lwd\]\[perf\] publish reqs=(\d+) dur=([\d.]+)ms")
RE_CLOUD_SCHED = re.compile(r"\[Lwd\]\[perf\] cloud-sched dur=([\d.]+)ms")
RE_HARVEST = re.compile(r"\[Lwd\]\[perf\] harvest dur=([\d.]+)ms")

# ---- [Lwd][sched] 调度批锚点(⑤段) ----
RE_RANGE_NOTIFY = re.compile(
    r"\[Lwd\]\[cloud-ctrl\] RangeNotify req=(\S+) num=(\S+) seqno=(\d+)")
RE_CSCHED = re.compile(
    r"\[Lwd\]\[sched\] cloud step=(\d+) phase=(\w+) seqno=(\S*) "
    r"reqs=\[([^\]]*)\] tokens=(\d+) pending_notify=(\d+) "
    r"decode_ready=(\d+)")
RE_EDISP_EMB = re.compile(
    r"\[Lwd\]\[sched\] edge dispatch-embed seqno=(\d+) req=(\S+) "
    r"tokens=(\d+) ahead_unemb=(\d+) ahead_emb=(\d+) c2e_pending=(\d+)")
RE_EDISP_UNE = re.compile(
    r"\[Lwd\]\[sched\] edge dispatch-unembed seqno=(\S+) reqs=(\d+) "
    r"rows=(\d+) ahead_unemb=(\d+) ahead_emb=(\d+) "
    r"c2e_wait=([\d.]+)ms c2e_pending=(\d+)")
RE_EHARV = re.compile(
    r"\[Lwd\]\[sched\] edge harvest kind=(\w+) seqno=(\S+) "
    r"wait=([\d.]+)ms")
RE_ESTEP = re.compile(
    r"\[Lwd\]\[sched\] edge step harvest_emb=(\d+) harvest_unemb=(\d+) "
    r"disp_emb=(\d+) disp_unemb=(\d+) queue=(\d+)emb/(\d+)unemb "
    r"c2e_pending=(\d+)")
RE_C2E_ARRIVE = re.compile(
    r"\[Lwd\]\[edge-ctrl\] C2eNotify reqs=(\d+) down_seqno=(\S+)")

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


def fmt_ts(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]


def stats(vals: list[float]) -> str:
    if not vals:
        return "n=0"
    s = sorted(vals)

    def pct(p: float) -> float:
        return s[min(len(s) - 1, int(len(s) * p))]

    return (f"n={len(s)} 均值={sum(s)/len(s):.1f} "
            f"p50={pct(0.5):.1f} p90={pct(0.9):.1f} max={s[-1]:.1f}")


def p50(vals: list[float]) -> float:
    if not vals:
        return 0.0
    return sorted(vals)[len(vals) // 2]


def p90(vals: list[float]) -> float:
    if not vals:
        return 0.0
    return sorted(vals)[min(len(vals) - 1, int(len(vals) * 0.9))]


def diffs(vals: list[float]) -> list[float]:
    return [b - a for a, b in zip(vals, vals[1:]) if b > a]


def ms(v):
    """秒→毫秒,接受标量或列表。"""
    if isinstance(v, list):
        return [x * 1000.0 for x in v]
    return v * 1000.0


class SideEvents:
    """单侧事件流(调度时间线重建用):按各自时钟排序后逐条打印。"""

    def __init__(self, name: str):
        self.name = name
        self.events: list[tuple[float, str]] = []

    def add(self, ts: float | None, text: str) -> None:
        if ts is not None:
            self.events.append((ts, text))

    def print(self, req_filter: str | None) -> None:
        print("=" * 64)
        print(f"⓪ 调度时间线 · {self.name}(本机时钟;"
              f"{'过滤 req=' + req_filter if req_filter else '全量'})")
        for ts, text in sorted(self.events):
            if req_filter and req_filter not in text:
                continue
            print(f"   {fmt_ts(ts)}  {text}")


def analyze(cloud_log: str, edge_log: str, verbose: bool,
            timeline: bool = False, req_filter: str | None = None) -> None:
    cloud_ev = SideEvents("云侧")
    edge_ev = SideEvents("边侧")

    # ---- 云侧 ----
    cloud_ts: list[float] = []
    cloud_seq: list[int] = []
    step_dt: list[float] = []
    step_ts: list[float] = []
    up_recv_ready: list[int] = []  # 1=True 0=False(云 worker 的 up-recv 行)
    tax: dict[str, list[float]] = {}  # 云侧每步 LWD 税分段
    bridge: dict[tuple[str, str], list[float]] = {}
    # ⑤段:调度批
    range_arrive: dict[int, float] = {}      # UP seqno -> 到达时刻
    range_req: dict[int, str] = {}           # UP seqno -> req_id
    csched: list[dict] = []                  # 每步 {ts,phase,seqno,reqs,...}
    for line in read_lines(cloud_log):
        t = parse_ts(line)
        m = RE_CLOUD_NOTIFY.search(line)
        if m:
            if t is not None:
                cloud_ts.append(t)
                cloud_seq.append(int(m.group(2)))
            continue
        m = RE_RANGE_NOTIFY.search(line)
        if m:
            seq = int(m.group(3))
            range_arrive.setdefault(seq, t if t is not None else 0.0)
            range_req[seq] = m.group(1)
            cloud_ev.add(t, f"RANGENotify 到达 seqno={seq} req={m.group(1)} "
                            f"num={m.group(2)}")
            continue
        m = RE_CSCHED.search(line)
        if m:
            rec = {
                "ts": t, "phase": m.group(2),
                "seqno": int(m.group(3)) if m.group(3) else None,
                "reqs": m.group(4), "tokens": int(m.group(5)),
                "pending": int(m.group(6)), "ready": int(m.group(7)),
            }
            if t is not None:
                csched.append(rec)
                cloud_ev.add(
                    t,
                    f"STEP {m.group(1)} {rec['phase']}"
                    f"{' seqno=' + m.group(3) if m.group(3) else ''} "
                    f"reqs=[{rec['reqs']}] tokens={rec['tokens']} "
                    f"pend_notify={rec['pending']} ready={rec['ready']}")
            continue
        m = RE_UP_RECV.search(line)
        if m:
            up_recv_ready.append(1 if m.group(2) == "True" else 0)
            continue
        m = RE_RANK.search(line)
        if m:
            tax.setdefault("rank-calc(all_gather+求和)", []).append(
                float(m.group(2)))
            continue
        m = RE_PACK.search(line)
        if m:
            tax.setdefault("collect(采样收集)", []).append(float(m.group(1)))
            tax.setdefault("pack(hidden拼接)", []).append(float(m.group(2)))
            continue
        m = RE_DOWN_SEND.search(line)
        if m:
            tax.setdefault("down-send(提交)", []).append(float(m.group(2)))
            continue
        m = RE_PUBLISH.search(line)
        if m:
            tax.setdefault("publish(码+ZMQ)", []).append(float(m.group(2)))
            cloud_ev.add(t, f"PUBLISH dur={m.group(2)}ms "
                            f"(大≈50ms整数倍=队满小睡)")
            continue
        m = RE_CLOUD_SCHED.search(line)
        if m:
            tax.setdefault("cloud-sched(相位调度)", []).append(
                float(m.group(1)))
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
    embed_ts: list[float] = []
    seg: dict[str, list[float]] = {}
    embed_seg: dict[str, list[float]] = {}
    # ⑤段:派发/收割/步汇总
    emb_disp: dict[int, dict] = {}    # UP seqno -> {ts,ahead_unemb,pend,req}
    une_disp: dict[str, dict] = {}    # down_seqno -> {ts,c2e_wait,pend,rows}
    harv: dict[tuple[str, str], float] = {}  # (kind,seqno) -> wait ms
    estep_pend: list[int] = []
    estep_starve = 0  # disp_emb=0 且队列被 unemb 占满的步数
    # c2e 积压水位重建(旧日志可用:C2eNotify 到达 vs UNEMBED 执行,
    # 同机墙钟可比):水位若从未接近 1000,云侧 sleep 不可能触发
    c2e_arrive_ts: dict[str, float] = {}
    une_exec_ts: dict[str, float] = {}
    for line in read_lines(edge_log):
        t = parse_ts(line)
        m = RE_EDGE_UNEMBED.search(line)
        if m and t is not None:
            edge_ts.append(t)
            edge_seq.append(int(m.group(1)))
            une_exec_ts.setdefault(m.group(1), t)
            continue
        m = RE_C2E_ARRIVE.search(line)
        if m:
            if t is not None:
                c2e_arrive_ts.setdefault(m.group(2), t)
            edge_ev.add(t, f"C2E 到达 down_seqno={m.group(2)} "
                           f"reqs={m.group(1)}")
            continue
        m = RE_EDISP_EMB.search(line)
        if m:
            emb_disp[int(m.group(1))] = {
                "ts": t, "req": m.group(2), "ahead_unemb": int(m.group(4)),
                "ahead_emb": int(m.group(5)), "pend": int(m.group(6)),
            }
            edge_ev.add(
                t,
                f"DISP-EMB seqno={m.group(1)} req={m.group(2)} "
                f"ahead={m.group(4)}unemb/{m.group(5)}emb "
                f"c2e_pend={m.group(6)}")
            continue
        m = RE_EDISP_UNE.search(line)
        if m:
            une_disp[m.group(1)] = {
                "ts": t, "rows": int(m.group(3)),
                "ahead_unemb": int(m.group(4)), "c2e_wait": float(m.group(6)),
                "pend": int(m.group(7)),
            }
            edge_ev.add(
                t,
                f"DISP-UNEMB seqno={m.group(1)} rows={m.group(3)} "
                f"c2e_wait={m.group(6)}ms ahead={m.group(4)}unemb "
                f"c2e_pend={m.group(7)}")
            continue
        m = RE_EHARV.search(line)
        if m:
            harv[(m.group(1), m.group(2))] = float(m.group(3))
            edge_ev.add(t, f"HARV {m.group(1)} seqno={m.group(2)} "
                           f"wait={m.group(3)}ms")
            continue
        m = RE_ESTEP.search(line)
        if m:
            # 组:1=hv_emb 2=hv_unemb 3=disp_emb 4=disp_unemb
            #    5=q_emb 6=q_unemb 7=c2e_pending
            estep_pend.append(int(m.group(7)))
            if int(m.group(3)) == 0 and int(m.group(6)) >= 4:
                estep_starve += 1
            edge_ev.add(
                t,
                f"ESTEP 收割{m.group(1)}emb/{m.group(2)}unemb "
                f"派发{m.group(3)}emb/{m.group(4)}unemb "
                f"队列{m.group(5)}emb/{m.group(6)}unemb "
                f"c2e_pend={m.group(7)}")
            continue
        m = RE_EMBED_PERF.search(line)
        if m:
            if t is not None:
                embed_ts.append(t)
            for name, idx in (("forward", 2), ("submit_send", 3),
                              ("total", 4)):
                embed_seg.setdefault(name, []).append(float(m.group(idx)))
            continue
        m = RE_EDGE_PERF.search(line)
        if m:
            if t is not None and not edge_ts:
                edge_ts.append(t)
                try:
                    edge_seq.append(int(m.group(1)))
                except ValueError:
                    pass
            if t is not None:
                une_exec_ts.setdefault(m.group(1), t)
            for name, idx in (("post_recv", 2), ("wait_tensor", 3),
                              ("lm_head", 4), ("select", 5), ("total", 6)):
                seg.setdefault(name, []).append(float(m.group(idx)))
            continue

    print("=" * 64)
    print("① 速率对比(时钟无关:两边各自算事件间隔)")
    c_iv = ms(diffs(cloud_ts))
    e_iv = ms(diffs(edge_ts))
    emb_iv = ms(diffs(embed_ts))
    print(f"   云 notify 间隔    : {stats(c_iv)} ms")
    print(f"   边 unembed 间隔   : {stats(e_iv)} ms")
    if emb_iv:
        print(f"   边 embed 间隔     : {stats(emb_iv)} ms  (chunk 节奏,TTFT 相关)")
    if step_dt:
        print(f"   云步间隔(探针)    : {stats(step_dt)} ms")
    for k, v in sorted(bridge.items()):
        print(f"   bridge-wait {k[0]}/{k[1]}: {stats(v)} ms")

    print("=" * 64)
    print("② 边侧单条成本分段")
    print("   [unembed · decode 回程 · DOWN 收+lm_head+重推]")
    for name in ("post_recv", "wait_tensor", "lm_head", "select", "total"):
        print(f"   {name:<11}: {stats(seg.get(name, []))} ms")
    if embed_seg:
        print("   [embed · prefill 上行 · embed 前向+UP 广播提交]")
        for name in ("forward", "submit_send", "total"):
            print(f"   {name:<11}: {stats(embed_seg[name])} ms")
    if up_recv_ready:
        n_true = sum(up_recv_ready)
        print(f"   [up-recv · 云收 embed · 取用时已就绪] "
              f"{n_true}/{len(up_recv_ready)} "
              f"(False 多 = 边侧发送/传输慢,云在等数据;"
              f"True 多 = 云侧消费拖节奏)")

    print("=" * 64)
    print("③ 云侧每步 LWD 税分段(对账:各段之和 ≈ cloud-step dt − 纯计算)")
    for name in ("rank-calc(all_gather+求和)", "collect(采样收集)",
                 "pack(hidden拼接)", "down-send(提交)",
                 "publish(码+ZMQ)", "cloud-sched(相位调度)"):
        if name in tax:
            print(f"   {name:<26}: {stats(tax[name])} ms")

    print("=" * 64)
    print("④ 尾巴检测(时钟无关:边事件从'等云节奏'切到'背靠背清账')")
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

    # ---- ⑤ 调度批重建 + 饿死检测(各自单机时钟,跨机不比对绝对时刻) ----
    print("=" * 64)
    print("⑤ 调度批重建 + 饿死检测")
    # ⑤-0:c2e 积压水位重建(旧日志即可,无需 [Lwd][sched]):
    #     逐 seqno 到达→worker 执行的滞留 + 到达/执行事件计数的峰值水位。
    #     云 sleep 触发需水位越过 1000(容量),峰值远低于此即排除该路径。
    if c2e_arrive_ts:
        sojourns = [
            (t1 - t0) * 1000
            for seq, t1 in une_exec_ts.items()
            if (t0 := c2e_arrive_ts.get(seq)) is not None and t1 > t0
        ]
        events = ([(t, 1) for t in c2e_arrive_ts.values()]
                  + [(t, -1) for t in une_exec_ts.values()])
        events.sort()
        water = cur = 0
        for _, d in events:
            cur += d
            water = max(water, cur)
        print(f"   [边] c2e 积压水位(到达/执行重建): 峰值={water}"
              f"  欠账时长(到达→执行): {stats(sojourns)} ms"
              f"  (队列容量1000;峰值≪1000 → 云 sleep 不可能由此触发)")
    else:
        print("   [边] 无 C2eNotify 到达行(旧版边日志?),水位重建不可用")
    # ⑤-云:相位构成 / RangeNotify→PREFILL 延迟 / 其间 decode 步数 / 停摆
    prefill_delays: list[float] = []
    decode_between: list[int] = []
    never_scheduled: list[int] = []
    empty_stall = 0
    if csched:
        by_phase: dict[str, int] = {}
        for rec in csched:
            by_phase[rec["phase"]] = by_phase.get(rec["phase"], 0) + 1
            if rec["phase"] == "EMPTY" and rec["pending"] > 0:
                empty_stall += 1
        dec_tokens = [r["tokens"] for r in csched if r["phase"] == "DECODE"]
        mix = " / ".join(f"{k}={v}" for k, v in sorted(by_phase.items()))
        print(f"   [云] 步构成: {mix}")
        if dec_tokens:
            print(f"   [云] decode 批 token 数: {stats([float(x) for x in dec_tokens])}")
        scheduled_seq = {r["seqno"]: r for r in csched
                         if r["phase"] == "PREFILL" and r["seqno"] is not None}
        for seq, t0 in range_arrive.items():
            rec = scheduled_seq.get(seq)
            if rec is None:
                never_scheduled.append(seq)
            elif t0 and rec["ts"]:
                prefill_delays.append((rec["ts"] - t0) * 1000)
                decode_between.append(sum(
                    1 for r in csched
                    if r["phase"] == "DECODE" and t0 < r["ts"] < rec["ts"]))
        if prefill_delays:
            print(f"   [云] RangeNotify到达→PREFILL步: {stats(prefill_delays)} ms")
            print(f"   [云] 其间插入 decode 步数    : {stats([float(x) for x in decode_between])}")
        if empty_stall:
            print(f"   [云] EMPTY步且pending_notify>0: {empty_stall} 次"
                  "(有活没吃:引擎线程被 publish 小睡/收割阻塞)")
        if never_scheduled:
            print(f"   [云] 从未被 PREFILL 步消费的 seqno: {len(never_scheduled)} 个"
                  f"(abort/丢通告){never_scheduled[:8]}")
    else:
        print("   [云] 无 [Lwd][sched] 行(旧日志?新增调度批日志后重采)")
    # ⑤-边:EMBED 被挤程度 / 队列滞留 / c2e 滞后
    if emb_disp or une_disp:
        ahead = [v["ahead_unemb"] for v in emb_disp.values()]
        if ahead:
            n_starved = sum(1 for a in ahead if a > 0)
            print(f"   [边] EMBED派发 ahead_unemb: {stats([float(a) for a in ahead])} "
                  f"(>0 占 {n_starved}/{len(ahead)}:EMBED 前方压着 decode 回程批)")
        waits_emb = {k[1]: v for k, v in harv.items() if k[0] == "embed"}
        waits_une = {k[1]: v for k, v in harv.items() if k[0] == "unembed"}
        if waits_emb:
            w_free = [v for s, v in waits_emb.items()
                      if s in emb_disp and emb_disp[s]["ahead_unemb"] == 0]
            w_blk = [v for s, v in waits_emb.items()
                     if s in emb_disp and emb_disp[s]["ahead_unemb"] > 0]
            print(f"   [边] EMBED 派发→收割 wait: 全部 {stats(list(waits_emb.values()))} ms")
            if w_blk:
                print(f"        前方无unemb: {stats(w_free)} ms / "
                      f"前方有unemb: {stats(w_blk)} ms"
                      "  ← 差值即被 decode 回程挤住的时长")
        if waits_une:
            print(f"   [边] UNEMBED 派发→收割 wait: {stats(list(waits_une.values()))} ms")
        c2e_waits = [v["c2e_wait"] for v in une_disp.values()]
        if c2e_waits:
            print(f"   [边] c2e 到达→派发 c2e_wait: {stats(c2e_waits)} ms"
                  "(消费滞后:持续偏大 → 云 publisher 队满小睡在即)")
        pend_series = estep_pend or [v["pend"] for v in une_disp.values()]
        if pend_series:
            print(f"   [边] c2e_pending 水位: max={max(pend_series)} "
                  f"p90={p90([float(x) for x in pend_series]):.0f}"
                  f"(容量1000,云侧小睡阈值)")
        if estep_starve:
            print(f"   [边] disp_emb=0 且队列被 unembed 占满的步数: {estep_starve}"
                  "(EMBED 想派派不进去)")
    else:
        print("   [边] 无 [Lwd][sched] 行(旧日志?新增调度批日志后重采)")

    # ---- ⑥ 判定 ----
    print("=" * 64)
    print("⑥ 判定")
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
    if embed_seg.get("submit_send"):
        ss = sorted(embed_seg["submit_send"])
        if p50(ss) > 5.0:
            verdicts.append(
                f"embed submit_send p50={p50(ss):.1f}ms 偏大:"
                "UP 广播提交阻塞(对照 up-recv:云若在等数据=边侧/传输慢;"
                "首块大后续小=一次性建链)")
    if up_recv_ready and sum(up_recv_ready) < len(up_recv_ready) * 0.3:
        verdicts.append(
            f"up-recv 就绪率仅 {sum(up_recv_ready)}/{len(up_recv_ready)}:"
            "云先挂收在等数据——瓶颈在边侧发送/传输"
            "(对照 embed submit_send 时长)")
    if step_dt and c_iv:
        dt_mean = sum(step_dt) / len(step_dt)
        pace = sum(c_iv) / len(c_iv)
        if dt_mean > pace * 1.3:
            verdicts.append(
                f"云步间隔({dt_mean:.1f}ms)明显大于 notify 节奏"
                f"({pace:.1f}ms):队列满背压已生效")
    pub = tax.get("publish(码+ZMQ)", [])
    if pub and p90(pub) >= 45.0:
        n_slept = sum(1 for d in pub if d >= 45.0)
        verdicts.append(
            f"云 publish p90={p90(pub):.1f}ms(≈50ms档=队满小睡):"
            f"sleep 实际触发 {n_slept}/{len(pub)} 次,最大 {max(pub):.1f}ms"
            "——云引擎主线程被 c2e 背压停摆,根因在边侧消费速率(对照②)")
    if prefill_delays and (p90(prefill_delays) > 100
                           or p50([float(x) for x in decode_between]) > 8):
        verdicts.append(
            f"云 prefill 被 decode 饿死倾向:RangeNotify 到达后 p90 等 "
            f"{p90(prefill_delays):.0f}ms、其间插 "
            f"{p50([float(x) for x in decode_between]):.0f} 个 decode 步")
    if empty_stall:
        verdicts.append(
            f"云引擎停摆证据:EMPTY 步且 pending_notify>0 共 {empty_stall} 次"
            "(通告已到却没被调度——引擎线程被阻塞,非调度策略问题)")
    if emb_disp:
        ahead = [v["ahead_unemb"] for v in emb_disp.values()]
        n_starved = sum(1 for a in ahead if a > 0)
        waits_emb = {k[1]: v for k, v in harv.items() if k[0] == "embed"}
        w_free = [v for s, v in waits_emb.items()
                  if s in emb_disp and emb_disp[s]["ahead_unemb"] == 0]
        w_blk = [v for s, v in waits_emb.items()
                 if s in emb_disp and emb_disp[s]["ahead_unemb"] > 0]
        if (n_starved > len(ahead) * 0.3 and w_free and w_blk
                and p50(w_blk) - p50(w_free) > 3.0):
            verdicts.append(
                f"边 EMBED 被 UNEMBED 挤住:{n_starved}/{len(ahead)} 次 EMBED "
                f"前方压着 unembed,wait 从 {p50(w_free):.1f}ms 涨到 "
                f"{p50(w_blk):.1f}ms → UP 迟到,云 prefill 等数据")
    c2e_waits = [v["c2e_wait"] for v in une_disp.values()] if une_disp else []
    if c2e_waits and p90(c2e_waits) > 20.0:
        verdicts.append(
            f"边 c2e 消费滞后:c2e_wait p90={p90(c2e_waits):.1f}ms"
            "(引擎步循环被收割/派发占住 → 积压 → 云小睡闭环)")
    if c2e_arrive_ts and pub and max(pub) < 45.0:
        events = ([(t, 1) for t in c2e_arrive_ts.values()]
                  + [(t, -1) for t in une_exec_ts.values()])
        events.sort()
        water = cur = 0
        for _, d in events:
            cur += d
            water = max(water, cur)
        if water < 900:
            verdicts.append(
                f"排除 c2e 背压路径:水位峰值仅 {water}(容量1000)且 publish "
                f"最大 {max(pub):.1f}ms<45ms,sleep 从未触发——'几秒等待'"
                "另有原因(查 EMBED 被 UNEMBED 挤住 / UP 链 / 云调度)")
    for v in verdicts or ["样本不足或无明显异常,看上面原始数字"]:
        print(f"   • {v}")

    if verbose:
        print("=" * 64)
        print("原始计数:",
              f"云 notify={len(cloud_seq)}(seqno {cloud_seq[0] if cloud_seq else '-'}"
              f"~{cloud_seq[-1] if cloud_seq else '-'})",
              f"边 UNEMBED={len(edge_seq)}(seqno {edge_seq[0] if edge_seq else '-'}"
              f"~{edge_seq[-1] if edge_seq else '-'})",
              f"云 sched步={len(csched)}",
              f"边 EMBED派发={len(emb_disp)} UNEMBED派发={len(une_disp)}")

    if timeline:
        cloud_ev.print(req_filter)
        edge_ev.print(req_filter)


def selftest() -> None:
    import tempfile
    cloud = "\n".join([
        "2026-09-12 10:00:00,100 INFO [Lwd][cloud-ctrl] RangeNotify "
        "req=req-1 num=512 seqno=7",
        "2026-09-12 10:00:00,100 INFO [Lwd][cloud-ctrl] handle_model_output: "
        "c2e_meta received reqs=1 down_seqno=0",
        "2026-09-12 10:00:00,110 INFO [Lwd][sched] cloud step=1 phase=DECODE "
        "seqno= reqs=[req-1] tokens=1 pending_notify=1 decode_ready=1",
        "2026-09-12 10:00:00,115 INFO [Lwd][perf] cloud-step dt=15.0ms",
        "2026-09-12 10:00:00,120 INFO [Lwd][sched] cloud step=2 phase=DECODE "
        "seqno= reqs=[req-1] tokens=1 pending_notify=1 decode_ready=1",
        "2026-09-12 10:00:00,130 INFO [Lwd][sched] cloud step=3 phase=PREFILL "
        "seqno=7 reqs=[req-1] tokens=512 pending_notify=0 decode_ready=1",
        "2026-09-12 10:00:00,131 INFO [Lwd][perf] publish reqs=1 dur=50.2ms",
        "2026-09-12 10:00:00,140 INFO [Lwd][cloud-ctrl] handle_model_output: "
        "c2e_meta received reqs=1 down_seqno=1",
        "2026-09-12 10:00:00,145 INFO [Lwd][perf] cloud-step dt=15.0ms",
    ])
    edge = "\n".join([
        "2026-09-12 10:00:00,104 INFO [Lwd][edge-ctrl] C2eNotify reqs=1 "
        "down_seqno=0",
        "2026-09-12 10:00:00,105 INFO [Lwd][edge-worker] UNEMBED seqno=0 reqs=1",
        "2026-09-12 10:00:00,105 INFO [Lwd][perf] unembed seqno=0 post_recv=0.5 "
        "wait_tensor=1.0 lm_head=2.0 select=10.0 total=14.0ms reqs=1",
        "2026-09-12 10:00:00,106 INFO [Lwd][sched] edge dispatch-unembed "
        "seqno=0 reqs=1 rows=1 ahead_unemb=0 ahead_emb=0 c2e_wait=1.0ms "
        "c2e_pending=0",
        "2026-09-12 10:00:00,108 INFO [Lwd][sched] edge dispatch-embed "
        "seqno=7 req=req-1 tokens=512 ahead_unemb=2 ahead_emb=0 "
        "c2e_pending=3",
        "2026-09-12 10:00:00,124 INFO [Lwd][edge-worker] UNEMBED seqno=1 reqs=1",
        "2026-09-12 10:00:00,124 INFO [Lwd][perf] unembed seqno=1 post_recv=0.5 "
        "wait_tensor=1.0 lm_head=2.0 select=10.0 total=14.0ms reqs=1",
        "2026-09-12 10:00:00,130 INFO [Lwd][sched] edge harvest kind=unembed "
        "seqno=0 wait=12.0ms",
        "2026-09-12 10:00:00,131 INFO [Lwd][sched] edge harvest kind=embed "
        "seqno=7 wait=25.0ms",
        "2026-09-12 10:00:00,131 INFO [Lwd][sched] edge step harvest_emb=1 "
        "harvest_unemb=1 disp_emb=1 disp_unemb=1 queue=0emb/1unemb "
        "c2e_pending=2",
    ])
    with tempfile.TemporaryDirectory() as d:
        for name, body in (("c.log", cloud), ("e.log", edge)):
            Path(d, name).write_text(body, encoding="utf-8")
        analyze(str(Path(d, "c.log")), str(Path(d, "e.log")), True,
                timeline=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="lwd c2e backlog probe")
    ap.add_argument("--cloud", help="云侧日志路径")
    ap.add_argument("--edge", help="边侧日志路径")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--timeline", action="store_true",
                    help="打印边/云各自时钟的调度时间线(人工核对顺序)")
    ap.add_argument("--req", help="时间线按请求 id 过滤(unembed 批只带 "
                    "seqno,不过滤该类行)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.cloud and args.edge):
        ap.error("需要 --cloud 与 --edge(或 --selftest)")
    analyze(args.cloud, args.edge, args.verbose,
            timeline=args.timeline, req_filter=args.req)


if __name__ == "__main__":
    main()
