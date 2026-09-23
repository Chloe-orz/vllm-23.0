#!/usr/bin/env python3
"""lwd_verify_router_dealer: ROUTER/DEALER 控制面换型后的日志验收。

从边/云两侧启动与请求日志中提取既有 [Lwd]/[LWD] 锚点,分六个阶段核对
换型后的行为是否符合预期;错误信号(注册被拒/丢帧/旧 socket 残留等)
出现即 FAIL。

日志锚点(均为现有日志,无需改代码):
  两侧 [LWD][config][parsed]           — 拓扑快照(取 digest 做两侧一致性)
       [LWD][config][transport]        — 投影端点(router_bind/dealers/identity)
       [LWD][config] path=... TP=.. PP=.. world=.. digest=..
       [LWD] distributed init method from lwd_config: tcp://ip:29600
       [Lwd][zmq] communicator up: type=ROUTER|DEALER mode=bind|connect
  云   [Lwd][cloud-ctrl] edge registered: identity=.. edge=E dp=D
       [Lwd][cloud-ctrl] RequestNotify req=R edge=E dp=D
       [Lwd] cloud request E#D#R admitted via gate        — rid 前缀形态
       [Lwd][cloud-ctrl] RangeNotify req=R num=N seqno=S edge=E dp=D
       [Lwd][cloud-sched] prefill notify req=E#D#R
       [Lwd][cloud-ctrl] publish C2eNotify->b'..' reqs=N down_seqno=K
  边   [Lwd] edge engine assembled: N DEALER link(s) registered, identity=..
       [Lwd][edge-notify] req=R offset=.. num=.. seqno=S — UP 序列边侧发送序
       [Lwd][edge-ctrl] C2eNotify reqs=N down_seqno=K edge=E dp=D

用法:
  python3 lwd_verify_router_dealer.py --edge edge.log --cloud cloud.log
  python3 lwd_verify_router_dealer.py --edge e.log --cloud c.log --strict

判定:
  * FAIL 任一出现退出码 1(可直接接 CI);--strict 下请求链路无流量也 FAIL
    (默认无流量记 SKIP,只验启动面)。
  * 阶段 E 的序列核对:边侧序列必须是云侧序列的顺序前缀(允许尾部
    在途未到),出现丢号/重号/乱序即 FAIL。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field

# --------------------------------------------------------------------- #
# 锚点正则(命名字段与日志格式一一对应)                                  #
# --------------------------------------------------------------------- #
RE_PARSED = re.compile(r"\[LWD\]\[config\]\[parsed\] (\{.*\})")
RE_TRANSPORT = re.compile(r"\[LWD\]\[config\]\[transport\] role=(\w+) instance_id=(\d+) wire=(\d+) digest=(\S+) values=(\{.*\})")
RE_CONFIG_LINE = re.compile(r"\[LWD\]\[config\] path=\S+ role=(\w+) instance_id=\d+ .*TP=(\d+) PP=(\d+) world=(\d+) digest=(\S+)")
RE_STORE = re.compile(r"\[LWD\] distributed init method from lwd_config: (\S+)")
RE_COMM = re.compile(r"\[Lwd\]\[zmq\] communicator up: type=(\w+) mode=(\w+) endpoint=(\S+)")
RE_REGISTERED = re.compile(r"\[Lwd\]\[cloud-ctrl\] edge registered: identity=(b'[^']*') edge=(\d+) dp=(\d+)")
RE_ASSEMBLED = re.compile(r"\[Lwd\] edge engine assembled: (\d+) DEALER link\(s\) registered, identity=(b'[^']*')")
RE_REQ_NOTIFY = re.compile(r"\[Lwd\]\[cloud-ctrl\] RequestNotify req=(\S+) edge=(-?\d+) dp=(-?\d+)")
RE_GATE = re.compile(r"\[Lwd\] cloud request (\S+) admitted via gate")
RE_RANGE_CLOUD = re.compile(r"\[Lwd\]\[cloud-ctrl\] RangeNotify req=(\S+) num=\d+ seqno=(\d+) edge=(-?\d+) dp=(-?\d+)")
RE_RANGE_EDGE = re.compile(r"\[Lwd\]\[edge-notify\] req=(\S+) offset=\d+ num=\d+ seqno=(\d+)")
RE_C2E_CLOUD = re.compile(r"\[Lwd\]\[cloud-ctrl\] publish C2eNotify->(b'[^']*') reqs=(\d+) down_seqno=(-?\d+)")
RE_C2E_EDGE = re.compile(r"\[Lwd\]\[edge-ctrl\] C2eNotify reqs=\d+ down_seqno=(-?\d+) edge=(-?\d+) dp=(-?\d+)")

# 负向信号:出现即判 FAIL(lwd 换型相关),行样例计数入报告
NEGATIVE_PATTERNS = {
    "register rejected": re.compile(r"register rejected from .*?(topology digest mismatch|NPU count mismatch|wire version mismatch)"),
    "no register-ack": re.compile(r"no register-ack within \d+s for links"),
    "unhashable": re.compile(r"TypeError: unhashable type"),
    "drop unregistered": re.compile(r"drop frame from unregistered edge"),
    "outbound dropped": re.compile(r"outbound frame dropped"),
    "executor failed": re.compile(r"EXECUTOR_FAILED"),
}
# 旧 PUSH/PULL 控制面残留(新代码不应再出现这两类 socket)
RE_OLD_SOCKET = re.compile(r"\[Lwd\]\[zmq\] communicator up: type=(PUSH|PULL) ")


@dataclass
class Side:
    """单侧日志的提取结果。"""

    name: str
    lines: list[str] = field(default_factory=list)
    parsed_digest: str | None = None
    transport_role: str | None = None
    transport_wire: int | None = None
    transport_digest: str | None = None
    transport_links: list[list[int]] = field(default_factory=list)
    router_bind: str | None = None
    dealer_endpoints: dict[str, str] = field(default_factory=dict)
    dealer_identity: str | None = None
    config_role: str | None = None
    config_tp: int | None = None
    config_pp: int | None = None
    config_world: int | None = None
    store: str | None = None
    sockets: list[tuple[str, str, str]] = field(default_factory=list)
    registered: list[tuple[str, int, int]] = field(default_factory=list)
    assembled: tuple[int, str] | None = None
    req_notify: list[tuple[str, int, int]] = field(default_factory=list)
    gate: list[str] = field(default_factory=list)
    range_cloud: list[tuple[str, int, int, int]] = field(default_factory=list)
    range_edge: list[tuple[str, int]] = field(default_factory=list)
    c2e_cloud: list[tuple[str, int, int]] = field(default_factory=list)
    c2e_edge: list[tuple[int, int, int]] = field(default_factory=list)
    negatives: dict[str, int] = field(default_factory=dict)
    old_sockets: list[str] = field(default_factory=list)


def open_log(path: str):
    """按 BOM 探测编码打开日志:板上经 PowerShell/部分重定向产出的日志
    是 UTF-16(带 BOM),直接按 utf-8 读会整体乱码、锚点全部失配。
    无 BOM 默认 utf-8(与原行为一致);解码错误 replace 不中断。"""
    with open(path, "rb") as probe:
        head = probe.read(4)
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        encoding = "utf-16"  # 解码器按 BOM 区分 LE/BE
    elif head.startswith(b"\xef\xbb\xbf"):
        encoding = "utf-8-sig"
    else:
        encoding = "utf-8"
    return open(path, encoding=encoding, errors="replace")


def parse_side(name: str, path: str) -> Side:
    side = Side(name=name)
    with open_log(path) as stream:
        side.lines = stream.readlines()
    for line in side.lines:
        if m := RE_PARSED.search(line):
            try:
                side.parsed_digest = json.loads(m.group(1)).get("topology", {}).get("digest") or None
            except json.JSONDecodeError:
                pass
        if m := RE_TRANSPORT.search(line):
            side.transport_role = m.group(1)
            side.transport_wire = int(m.group(3))
            side.transport_digest = m.group(4) if m.group(4) != "-" else ""
            try:
                values = json.loads(m.group(5))
                side.transport_links = values.get("links", [])
                side.router_bind = values.get("router_bind")
                side.dealer_endpoints = values.get("dealers", {})
                identity = values.get("identity")
                side.dealer_identity = identity or None
            except json.JSONDecodeError:
                pass
        if m := RE_CONFIG_LINE.search(line):
            side.config_role, side.config_tp, side.config_pp = m.group(1), int(m.group(2)), int(m.group(3))
            side.config_world = int(m.group(4))
        if m := RE_STORE.search(line):
            side.store = m.group(1)
        if m := RE_COMM.search(line):
            side.sockets.append((m.group(1), m.group(2), m.group(3)))
        if m := RE_REGISTERED.search(line):
            side.registered.append((m.group(1), int(m.group(2)), int(m.group(3))))
        if m := RE_ASSEMBLED.search(line):
            side.assembled = (int(m.group(1)), m.group(2))
        if m := RE_REQ_NOTIFY.search(line):
            side.req_notify.append((m.group(1), int(m.group(2)), int(m.group(3))))
        if m := RE_GATE.search(line):
            side.gate.append(m.group(1))
        if m := RE_RANGE_CLOUD.search(line):
            side.range_cloud.append((m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))))
        if m := RE_RANGE_EDGE.search(line):
            side.range_edge.append((m.group(1), int(m.group(2))))
        if m := RE_C2E_CLOUD.search(line):
            side.c2e_cloud.append((m.group(1), int(m.group(2)), int(m.group(3))))
        if m := RE_C2E_EDGE.search(line):
            side.c2e_edge.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
        for label, pattern in NEGATIVE_PATTERNS.items():
            if pattern.search(line):
                side.negatives[label] = side.negatives.get(label, 0) + 1
        if m := RE_OLD_SOCKET.search(line):
            side.old_sockets.append(m.group(0))
    # 无 BOM 的 UTF-16 按 utf-8 读会剩大量 NUL:锚点全失配前先提醒
    nul = sum(1 for line in side.lines[:500] if "\x00" in line)
    if side.lines and nul > len(side.lines[:500]) // 2:
        print(f"[WARN] {name} 日志疑似 UTF-16 但无 BOM(过半行含 NUL),"
              f"请转存为 UTF-8/带 BOM 的 UTF-16 后重试", file=sys.stderr)
    return side


# --------------------------------------------------------------------- #
# 检查框架                                                               #
# --------------------------------------------------------------------- #
class Report:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, str]] = []  # (verdict, check, evidence)

    def add(self, verdict: str, check: str, evidence: str = "") -> None:
        self.results.append((verdict, check, evidence))

    @property
    def failed(self) -> bool:
        return any(v == "FAIL" for v, _, _ in self.results)

    def dump(self) -> None:
        width = max(len(c) for _, c, _ in self.results) if self.results else 0
        counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
        for verdict, check, evidence in self.results:
            counts[verdict] = counts.get(verdict, 0) + 1
            print(f"[{verdict}] {check.ljust(width)}  {evidence}")
        print(f"\n合计: PASS={counts['PASS']} FAIL={counts['FAIL']} "
              f"WARN={counts['WARN']} SKIP={counts['SKIP']}")
        print("结论: " + ("存在不符合预期的行为,按上方 FAIL 项定位" if self.failed
                          else "控制面换型行为符合预期"))


def check_prefix_order(
    sent: list, received: list, label: str, report: Report,
    sender: str, receiver: str,
) -> None:
    """received 必须是 sent 的顺序前缀(尾部允许在途);丢/重/乱序即 FAIL。"""
    if not sent and not received:
        report.add("SKIP", f"{label}: 双侧无流量", "")
        return
    n = len(received)
    if n > len(sent):
        report.add("FAIL", f"{label}: {receiver} 多于 {sender}",
                   f"{sender}={len(sent)} {receiver}={n}")
        return
    mismatch = next((i for i in range(n) if sent[i] != received[i]), None)
    if mismatch is not None:
        report.add("FAIL", f"{label}: 序列发散(丢号/重号/乱序)",
                   f"第 {mismatch} 条 {sender}={sent[mismatch]} {receiver}={received[mismatch]}")
        return
    if n < len(sent):
        report.add("PASS", f"{label}: 前缀一致(尾部在途)",
                   f"{n}/{len(sent)}")
    else:
        report.add("PASS", f"{label}: 完全一致", f"{n} 条")


def verify(edge: Side, cloud: Side, strict: bool) -> Report:
    report = Report()

    # ---- A 配置一致性 ----
    digests = {s.name: s.transport_digest or s.parsed_digest for s in (edge, cloud)}
    if all(digests.values()):
        verdict = "PASS" if len(set(digests.values())) == 1 else "FAIL"
        report.add(verdict, "A1 两侧拓扑 digest 一致",
                   f"edge={digests['edge'][:16]} cloud={digests['cloud'][:16]}")
    else:
        report.add("FAIL", "A1 两侧拓扑 digest", f"缺失: {digests}")
    for s in (edge, cloud):
        if s.transport_wire is not None:
            report.add("PASS" if s.transport_wire == 2 else "FAIL",
                       f"A2 {s.name} wire_version=2", f"实际 {s.transport_wire}")
    report.add("PASS" if cloud.router_bind else "FAIL",
               "A3 云侧 ROUTER bind 端点", str(cloud.router_bind))
    report.add("PASS" if edge.dealer_endpoints and not edge.router_bind else "FAIL",
               "A4 边侧仅 DEALER 端点(无 bind)", f"dealers={list(edge.dealer_endpoints.values())}")
    if edge.dealer_identity:
        report.add("PASS", "A5 边侧稳定 identity", str(edge.dealer_identity))
    else:
        report.add("FAIL", "A5 边侧稳定 identity", "缺失")
    if cloud.router_bind and edge.dealer_endpoints:
        cloud_port = cloud.router_bind.rsplit(":", 1)[-1]
        dealer_ports = {ep.rsplit(":", 1)[-1] for ep in edge.dealer_endpoints.values()}
        report.add("PASS" if dealer_ports == {cloud_port} else "FAIL",
                   "A6 边 DEALER 端口 == 云 ctrl_port",
                   f"cloud={cloud_port} dealers={sorted(dealer_ports)}")
    if edge.config_tp is not None and cloud.config_tp is not None:
        ok = (edge.config_tp, cloud.config_tp, cloud.config_world) == (1, 8, 9)
        report.add("PASS" if ok else "FAIL", "A7 TP/world(1+8=9)",
                   f"edge TP={edge.config_tp} cloud TP={cloud.config_tp} world={cloud.config_world}")
    if cloud.config_pp is not None:
        report.add("PASS" if cloud.config_pp == 1 else "FAIL",
                   "A8 云侧 PP=1(no-PP)", f"PP={cloud.config_pp}")

    # ---- B 共享世界 store ----
    if edge.store and cloud.store:
        same = edge.store == cloud.store
        port_ok = edge.store.endswith(":29600")
        report.add("PASS" if same and port_ok else "FAIL", "B1 两侧 store 一致",
                   f"edge={edge.store} cloud={cloud.store}")
    else:
        report.add("FAIL", "B1 两侧 store", "日志缺 distributed init method 行")

    # ---- C 通信面换型 ----
    cloud_router = [s for s in cloud.sockets if s[0] == "ROUTER" and s[1] == "bind"]
    edge_dealer = [s for s in edge.sockets if s[0] == "DEALER" and s[1] == "connect"]
    report.add("PASS" if cloud_router else "FAIL", "C1 云侧 ROUTER bind",
               f"{len(cloud_router)} 个")
    report.add("PASS" if edge_dealer else "FAIL", "C2 边侧 DEALER connect",
               f"{len(edge_dealer)} 个")
    old = edge.old_sockets + cloud.old_sockets
    report.add("FAIL" if old else "PASS", "C3 无 PUSH/PULL 旧 socket 残留",
               f"{len(old)} 处" + (f" 如 {old[0]}" if old else ""))

    # ---- D 注册握手 ----
    if cloud.registered:
        identity, e_id, dp = cloud.registered[0]
        report.add("PASS", "D1 云侧 edge registered",
                   f"{identity} edge={e_id} dp={dp} (共 {len(cloud.registered)} 次)")
    else:
        report.add("FAIL", "D1 云侧 edge registered", "未出现(边未注册或日志不全)")
    if edge.assembled:
        n_links, identity = edge.assembled
        report.add("PASS", "D2 边侧 engine assembled",
                   f"{n_links} DEALER, {identity}")
    else:
        report.add("FAIL", "D2 边侧 engine assembled", "未出现(构造失败或日志不全)")
    if cloud.registered and edge.assembled:
        same = cloud.registered[0][0] == edge.assembled[1]
        report.add("PASS" if same else "FAIL", "D3 两侧 identity 一致",
                   f"cloud={cloud.registered[0][0]} edge={edge.assembled[1]}")
    if edge.dealer_endpoints and edge.assembled:
        report.add("PASS" if edge.assembled[0] == len(edge.dealer_endpoints) else "FAIL",
                   "D4 assembled 连接数 == dealer 端点数",
                   f"{edge.assembled[0]} vs {len(edge.dealer_endpoints)}")

    # ---- E 请求链路(rid 前缀 + 两条序列) ----
    has_traffic = bool(edge.range_edge or cloud.c2e_cloud)
    if not has_traffic:
        report.add("SKIP" if not strict else "FAIL", "E* 请求链路",
                   "日志中无请求流量(--strict 下记 FAIL)")
    else:
        if cloud.req_notify and cloud.gate:
            sample_req, e_id, dp = cloud.req_notify[0]
            expected_key = f"{e_id}#{dp}#{sample_req}"
            hit = expected_key in cloud.gate
            report.add("PASS" if hit else "FAIL", "E1 rid 前缀形态(E#D#R)",
                       f"gate 收到 {expected_key}" if hit
                       else f"gate 首条 {cloud.gate[0]} 期望 {expected_key}")
        else:
            report.add("WARN", "E1 rid 前缀形态", "RequestNotify/gate 日志不全")
        # E2 UP seqno: 边发送序 vs 云接收序(单链路下值序一致)
        sent = [s for _, s in edge.range_edge]
        received = [s for _, s, _, _ in cloud.range_cloud]
        check_prefix_order(sent, received, "E2 UP seqno 序列", report,
                           sender="边发送", receiver="云接收")
        # E3 DOWN seqno: 云发布序 vs 边接收序
        sent_down = [k for _, _, k in cloud.c2e_cloud]
        received_down = [k for k, _, _ in edge.c2e_edge]
        check_prefix_order(sent_down, received_down, "E3 DOWN seqno 序列", report,
                           sender="云发布", receiver="边接收")
        # E4 C2e 定向目标必须是已注册 identity
        if cloud.c2e_cloud and cloud.registered:
            targets = {t for t, _, _ in cloud.c2e_cloud}
            known = {r[0] for r in cloud.registered}
            bad = targets - known
            report.add("FAIL" if bad else "PASS", "E4 C2e 定向目标已注册",
                       f"异常目标 {sorted(bad)}" if bad else f"{len(targets)} 个 identity")

    # ---- F 负向信号 ----
    for s in (edge, cloud):
        for label, count in s.negatives.items():
            report.add("FAIL", f"F {s.name}: {label}", f"{count} 次")
    for s in (edge, cloud):
        if not s.negatives:
            report.add("PASS", f"F {s.name} 无负向信号", "")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--edge", required=True, help="边侧日志文件")
    parser.add_argument("--cloud", required=True, help="云侧日志文件")
    parser.add_argument("--strict", action="store_true",
                        help="请求链路无流量时按 FAIL 处理(默认 SKIP)")
    args = parser.parse_args()
    edge = parse_side("edge", args.edge)
    cloud = parse_side("cloud", args.cloud)
    print(f"edge 日志 {len(edge.lines)} 行, cloud 日志 {len(cloud.lines)} 行\n")
    report = verify(edge, cloud, args.strict)
    report.dump()
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
