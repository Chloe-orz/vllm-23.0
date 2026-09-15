#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""lwd_pickle_race_demo: 复现 pinned 元数据跨进程 pickle 旧值问题的最小实验。

背景(prefill_only LWD 无同步版本的 bug):
  云 worker 把 int32 元数据 [ranks|counts|seg_lens] 经
  ``copy_(meta_dev, non_blocking=True)`` 异步落到 pinned 缓冲(设备上
  排队,尚未执行),随后把 ``(pinned, req_ids, ...)`` 挂到输出返回引擎。
  worker 与引擎是两个进程(MultiprocExecutor + MQ),对象出进程必须
  pickle——而 pickle 张量/缓冲 = 把【当前字节】抄进字节流,发生在
  worker 返回后一瞬间 = 设备拷贝落地之前。引擎拿到的因此是一份
  "过早冻结的快照"(旧一步的值或未初始化垃圾),它在同步点之后解析
  得再晚也读不到新值。下游即 tokens=0 / 强制 ERROR / 僵尸解码。

本脚本无需 NPU/torch,两个实验:
  1. 快照语义(确定性):pickle 捕获序列化时刻的数据,此后原对象
     怎么改都与反序列化结果无关;
  2. 竞态模拟(概率性):后台线程模拟设备 DMA 慢速写 pinned(等效
     异步 D2H),主线程立即 pickle 模拟 worker 返回→mq.put——统计
     快照里新/旧/混合数据的占比,并演示"压力越大错得越多"。

用法:
  python3 lwd_pickle_race_demo.py            # 跑全部实验
  python3 lwd_pickle_race_demo.py --race-only
  python3 lwd_pickle_race_demo.py --torch    # 追加 torch 张量版(需 torch)
"""

from __future__ import annotations

import argparse
import pickle
import random
import threading
import time
from dataclasses import dataclass, field

# 模拟 pinned 元数据缓冲:一步 16 个 int32,拍成字节
META_ELEMS = 16
OLD_STEP = 0x11  # "旧一步"的填充值(torch.empty 轮换环的残留)
NEW_STEP = 0x22  # "本步"的真值


def _meta_bytes(value: int) -> bytes:
    return bytes([value] * META_ELEMS)


@dataclass
class Carrier:
    """worker 挂到输出上的载荷(无同步版本的原样抽象)。"""

    pinned: bytearray = field(default_factory=lambda: bytearray(_meta_bytes(OLD_STEP)))
    req_ids: list[str] = field(default_factory=lambda: ["req-A"])


def demo_snapshot_semantics() -> None:
    """实验 1:pickle = 拍照,不是传引用。"""
    print("=" * 60)
    print("实验 1:pickle 快照语义(确定性)")
    print("=" * 60)
    carrier = Carrier()
    snapshot = pickle.dumps(carrier)  # worker: mq.put() 时刻拍照
    carrier.pinned[:] = _meta_bytes(NEW_STEP)  # 设备:拍照后 DMA 才落地
    restored = pickle.loads(snapshot)  # 引擎:收到的是照片
    stale = all(b == OLD_STEP for b in restored.pinned)
    print(f"  worker 侧活缓冲   : {[hex(b) for b in carrier.pinned[:4]]} ... (新值 0x22)")
    print(f"  引擎侧反序列化结果 : {[hex(b) for b in restored.pinned[:4]]} ... "
          f"({'旧值 0x11 —— 照片不会更新' if stale else '新值?'})")
    print("  结论:跨进程 pickle 捕获的是序列化那一刻的字节;\n"
          "        引擎侧'同步点之后再读'保护的是活缓冲,不是这份快照。\n")


def _device_writer(
    carrier: Carrier,
    tail_delay_s: float,
    step_start: threading.Event,
    step_done: threading.Event,
    stop: threading.Event,
) -> None:
    """模拟设备侧一步:先排"计算尾部"(tail_delay,负载越大越长),
    然后执行 D2H。小 int32 元数据的 DMA 近乎瞬时,一次性写入。"""
    while not stop.is_set():
        step_start.wait()
        if stop.is_set():
            return
        step_start.clear()
        time.sleep(tail_delay_s)  # non_blocking 拷贝在设备上排队等计算尾部
        carrier.pinned[:] = _meta_bytes(NEW_STEP)
        step_done.set()


# worker 返回→组包→mq.put(pickle)的窗口(与负载无关)。
# 真实管线为 µs~ms 级;demo 统一放大到 ms 标度便于跨平台稳定复现,
# 比例关系(pickle 窗口 vs 各档计算尾部)与真实一致。
PICKLE_DELAY_S = 3e-3
TAIL_PER_LOAD_S = 2e-3
TRIALS = 150


def demo_race(load_level: int, trials: int = TRIALS) -> None:
    """实验 2:异步 D2H vs 立即 pickle 的竞态,统计快照质量。

    真实时序:worker 入队拷贝后立即返回→mq.put(pickle 窗口固定);
    设备要先跑完本步计算尾部才执行拷贝——负载越大尾部越长,
    pickle 越容易落在拷贝【之前】= 快照全是旧值(错乱元数据)。
    """
    print("=" * 60)
    print(f"实验 2:竞态模拟(计算尾部 = {load_level} 档 ≈ "
          f"{load_level * TAIL_PER_LOAD_S * 1e3:.1f}ms,"
          f"pickle 窗口 = {PICKLE_DELAY_S * 1e3:.1f}ms;"
          f"时标已放大,比例同真实)")
    print("=" * 60)
    carrier = Carrier()
    step_start, step_done, stop = threading.Event(), threading.Event(), threading.Event()
    writer = threading.Thread(
        target=_device_writer,
        args=(carrier, load_level * TAIL_PER_LOAD_S,
              step_start, step_done, stop),
        daemon=True,
    )
    writer.start()

    stale, mixed, fresh = 0, 0, 0
    for _ in range(trials):
        # pinned 环轮转:本步开始前缓冲里是"上一步的残留"
        carrier.pinned[:] = _meta_bytes(OLD_STEP)
        step_start.set()                     # worker:copy_ 入队,立即返回
        time.sleep(PICKLE_DELAY_S)           # 返回→组包→mq.put(pickle)
        snap = pickle.loads(pickle.dumps(carrier))  # 引擎收到的快照
        step_done.wait(); step_done.clear()  # 等设备本步写完再进下一轮
        n_new = sum(1 for b in snap.pinned if b == NEW_STEP)
        if n_new == 0:
            stale += 1      # 全旧值:拷贝未落地,counts/seg_lens 整体错位
        elif n_new == META_ELEMS:
            fresh += 1      # 全新值:尾部短于 pickle 窗口,侥幸正确
        else:
            mixed += 1      # 拍在 DMA 写入中:窄窗口,偶发
    stop.set()
    step_start.set()  # 唤醒写线程退出
    writer.join(timeout=1)

    print(f"  {trials} 步 '入队→立即pickle' 中:")
    print(f"    全新值(侥幸正确) : {fresh:4d}  ({fresh / trials:6.1%})")
    print(f"    全旧值(整体错位) : {stale:4d}  ({stale / trials:6.1%})")
    print(f"    混合值(部分垃圾) : {mixed:4d}  ({mixed / trials:6.1%})")
    wrong = stale + mixed
    verdict = ("计算尾部短于 pickle 窗口,基本不踩 —— ≈单请求/低并发,验证通过"
               if wrong / trials < 0.05 else
               "计算尾部超过 pickle 窗口,高概率踩中 —— ≈高并发满批,快照不可信")
    print(f"  结论:{verdict}。\n"
          f"  可自行对比:--load 1(≈单请求)vs --load 4(≈满批并发)。\n")


def demo_torch_variant() -> None:
    """实验 3(可选):torch 张量版,与管线里的 pinned 张量同型。"""
    try:
        import torch
    except ImportError:
        print("[skip] 未安装 torch,跳过实验 3")
        return
    print("=" * 60)
    print("实验 3:torch 张量版(与 pinned 张量同型)")
    print("=" * 60)
    t = torch.full((META_ELEMS,), OLD_STEP, dtype=torch.int32)
    snapshot = pickle.dumps(t)  # mq.put():抄走当前存储字节
    t.fill_(NEW_STEP)           # 设备:此后 DMA 才落地
    t2 = pickle.loads(snapshot)
    print(f"  活张量 t  = {t[:4].tolist()} ... (0x22={NEW_STEP},新值)")
    print(f"  快照张量 t2 = {t2[:4].tolist()} ... (0x11={OLD_STEP},旧值)"
          if (t2 == OLD_STEP).all() else f"  快照张量 t2 = {t2.tolist()}")
    print("  结论:张量跨进程同样只带走序列化时刻的字节。\n")


def print_mapping() -> None:
    print("=" * 60)
    print("映射回 LWD 管线")
    print("=" * 60)
    print("""  白板(被写)        = pinned 元数据缓冲 [ranks|counts|seg_lens]
  写白板的人(慢)    = NPU 设备,non_blocking D2H 拷贝排在整步计算之后
  拍照               = worker 返回后 mq.put() 触发的 pickle(抄当前字节)
  收照片的人         = 云引擎进程,拿到的永远是过早的快照
  快照错乱的下游行   = counts/seg_lens 错位 → 边侧空行 → tokens=0
                        → 强制 ERROR 埋请求 → 云侧僵尸解码 → stale warning
  修复(同步版,当前) = 拍照前先 synchronize 等 DMA 落地(单请求 +3ms 税)
  修复(数据面版,stash)= 这份数据不过云侧主机,设备直发边侧""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--race-only", action="store_true", help="只跑竞态实验")
    parser.add_argument("--torch", action="store_true", help="追加 torch 版实验")
    parser.add_argument("--load", type=int, default=8,
                        help="竞态实验的负载档位(计算尾部倍数,默认 8;"
                             "1≈单请求,4+≈满批并发)")
    args = parser.parse_args()

    random.seed(0)
    if not args.race_only:
        demo_snapshot_semantics()
    demo_race(load_level=max(1, args.load))
    if not args.race_only:
        if args.torch:
            demo_torch_variant()
        print_mapping()


if __name__ == "__main__":
    main()
