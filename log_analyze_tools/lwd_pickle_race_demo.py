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


def _device_writer(carrier: Carrier, write_delay_s: float, stop: threading.Event) -> None:
    """模拟设备 DMA:慢速逐字节写入(等效 non_blocking D2H 在设备上排队后执行)。"""
    while not stop.is_set():
        for i in range(META_ELEMS):
            if stop.is_set():
                return
            carrier.pinned[i] = NEW_STEP
            time.sleep(write_delay_s)
        # 回到"旧值"再写一轮,模拟 pinned 环轮转后下一步复用
        for i in range(META_ELEMS):
            if stop.is_set():
                return
            carrier.pinned[i] = OLD_STEP
            time.sleep(write_delay_s)


def demo_race(load_level: int) -> None:
    """实验 2:异步写 vs 立即 pickle 的竞态,统计快照质量。

    load_level 越大 = 设备尾巴越长(DMA 延迟越大),对应并发越高。
    """
    print("=" * 60)
    print(f"实验 2:竞态模拟(设备写延迟 = {load_level} x 基准)")
    print("=" * 60)
    carrier = Carrier()
    stop = threading.Event()
    # 主线程 pickle 一次的耗时约为几十 µs;写延迟按同量级放大
    writer = threading.Thread(
        target=_device_writer,
        args=(carrier, load_level * 20e-6, stop),
        daemon=True,
    )
    writer.start()
    time.sleep(0.01)  # 让设备线程先进入"写新值"阶段

    trials, stale, mixed, fresh = 2000, 0, 0, 0
    for _ in range(trials):
        snap = pickle.loads(pickle.dumps(carrier))  # mq.put + 引擎 loads
        n_new = sum(1 for b in snap.pinned if b == NEW_STEP)
        if n_new == 0:
            stale += 1      # 全旧值:counts/seg_lens 整体错位
        elif n_new == META_ELEMS:
            fresh += 1      # 全新值:本步侥幸正确
        else:
            mixed += 1      # 半新半旧:更隐蔽的垃圾
    stop.set()
    writer.join(timeout=1)

    print(f"  {trials} 次 'worker返回→立即pickle' 中:")
    print(f"    全新值(侥幸正确) : {fresh:5d}  ({fresh / trials:6.1%})")
    print(f"    全旧值(整体错位) : {stale:5d}  ({stale / trials:6.1%})")
    print(f"    混合值(部分垃圾) : {mixed:5d}  ({mixed / trials:6.1%})")
    verdict = "竞态被踩中,快照不可信" if (stale + mixed) > 0 else "本组延迟下窗口未命中"
    print(f"  结论:{verdict}。\n")


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
                        help="竞态实验的负载档位(设备写延迟倍数,默认 8)")
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
