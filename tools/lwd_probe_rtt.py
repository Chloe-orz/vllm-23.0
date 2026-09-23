#!/usr/bin/env python3
"""lwd_probe_rtt: 边云控制面链路 RTT 抖动探测(纯 pyzmq,无 vllm 依赖)。

用与线上完全相同的 ROUTER/DEALER 模式(云 bind、边 connect+identity)
测一来一回的 RTT 分布,隔离"网络抖还是 ZMQ 排队抖"。空闲机器上跑出的
是链路基线;与 vllm 同机跑可看负载下的退化。服务端原样回显(带请求
时间戳),客户端算 RTT 分位。

用法:
  云侧: python3 tools/lwd_probe_rtt.py --server --port 7200
  边侧: python3 tools/lwd_probe_rtt.py --client --addr 76.76.26.233:7200 \
             --count 2000 --interval 0.01
输出: 发送数/回显数/丢失、RTT p50/p90/p99/max(ms)、>10ms 抖动占比。
退出码 0;p99 超 5ms 或丢失非零时打印异常提示。
"""

from __future__ import annotations

import argparse
import struct
import sys
import time

import zmq


def run_server(port: int) -> None:
    ctx = zmq.Context()
    router = ctx.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    router.bind(f"tcp://*:{port}")
    print(f"server: ROUTER bind *:{port}, echoing")
    count = 0
    try:
        while True:
            identity, payload = router.recv_multipart()
            router.send_multipart([identity, payload])
            count += 1
            if count % 1000 == 0:
                print(f"echoed {count}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        router.close(0)
        ctx.term()


def run_client(addr: str, count: int, interval: float) -> int:
    ctx = zmq.Context()
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.LINGER, 0)
    dealer.setsockopt(zmq.IDENTITY, b"probe-edge-0")
    dealer.connect(f"tcp://{addr}")
    # 序号 + 客户端发送时刻(monotonic 秒, double)
    rtts: dict[int, float] = {}
    sent_at: dict[int, float] = {}
    outstanding = 0
    poller = zmq.Poller()
    poller.register(dealer, zmq.POLLIN)
    seq = 0
    deadline = time.monotonic() + 30  # 连接建立等待上限
    while (seq < count or outstanding) and time.monotonic() < deadline + 60:
        if seq < count:
            frame = struct.pack("!qd", seq, time.monotonic())
            dealer.send(frame)
            sent_at[seq] = frame[8:]
            seq += 1
            outstanding += 1
            if seq % 500 == 0:
                print(f"sent {seq}/{count}", flush=True)
        events = dict(poller.poll(timeout=int(interval * 1000)))
        if events.get(dealer) == zmq.POLLIN:
            frame = dealer.recv()
            rseq, t0 = struct.unpack("!qd", frame)
            if rseq in sent_at:
                rtts[rseq] = (time.monotonic() - t0) * 1000.0
                outstanding -= 1
        # 发送节奏:允许积压在途(outstanding 上限防打爆队列)
        while outstanding > 256 and seq < count:
            break
        time.sleep(interval if outstanding < 256 else 0)
    dealer.close(0)
    ctx.term()

    values = sorted(rtts.values())
    lost = count - len(values)
    if not values:
        print("no replies; check server/firewall")
        return 1

    def pct(p: float) -> float:
        return values[min(len(values) - 1, int(len(values) * p))]

    janky = sum(1 for v in values if v > 10.0)
    print(
        f"sent={count} replied={len(values)} lost={lost}\n"
        f"RTT ms: p50={pct(0.50):.3f} p90={pct(0.90):.3f} "
        f"p99={pct(0.99):.3f} max={values[-1]:.3f} min={values[0]:.3f}\n"
        f">10ms 占比: {janky / len(values) * 100:.2f}%"
    )
    if lost or pct(0.99) > 5.0:
        print("异常: p99>5ms 或存在丢失——链路抖动坐实,查网卡/路由/负载")
        return 1
    print("链路基线正常(亚毫秒级)——抖动若仍出现,查本机负载/排队而非网络")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LWD 控制面 RTT 探测")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--server", action="store_true", help="云侧: ROUTER bind 回显")
    group.add_argument("--client", action="store_true", help="边侧: DEALER 测 RTT")
    parser.add_argument("--port", type=int, default=7200)
    parser.add_argument("--addr", help="客户端目标 addr:port")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--interval", type=float, default=0.01, help="发送间隔秒")
    args = parser.parse_args()
    if args.server:
        run_server(args.port)
        return 0
    if not args.addr:
        parser.error("--client requires --addr host:port")
    return run_client(args.addr, args.count, args.interval)


if __name__ == "__main__":
    sys.exit(main())
