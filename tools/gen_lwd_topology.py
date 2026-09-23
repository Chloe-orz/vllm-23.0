#!/usr/bin/env python3
"""gen_lwd_topology: LWD 拓扑文件生成器(设计文档 lwd_topology_config_design §5.2)。

六类输入推导整份 YAML(id/ranks/端口/配对全算,不手写),生成即校验
(先跑 LwdTopology.from_dict 全量校验再落盘,产物可直接被两侧认领)。

  1. --scene          场景(与边机/云机数互推,§4.3):single_instance 1/1,
                      edge_share 1/N,cloud_share N/1,lwd_cluster N/N
  2. --edge-machines  边机地址清单(逗号分隔,按 rank 顺序)
  3. --cloud-machines 云机地址清单(逗号分隔,按 rank 顺序)
  4. --dp             每实例 dp 数(可逗号分隔批量,如 1,2;必须整除本机卡数)
  5. --port-base      云侧控制端口基址(默认 5550;ctrl_port = base + dp_idx,
                      同机错开,跨机可重开)
  6. --enable-early-recv/--enable-scramble  特性开关(默认 false)

每机卡数当前是常量(边 1 / 云 8,§5.2 第 7 类输入,异构机器时再提升)。

rank 分配规则(§4.2):先边后云;边机每台 1 个 rank(同机所有实例/dp 复用);
云机每台占连续 8 个 rank,机内按 dp_idx 升序均分。

用法:
  python3 tools/gen_lwd_topology.py --scene single_instance \\
      --edge-machines 76.76.26.17 --cloud-machines 76.76.26.233 \\
      --dp 1 --port-base 7200 -o etc/lwd/lwd_config.yaml
  # 批量: --dp 1,2 一次产多份(-o 给目录或名不含 DP 时自动加后缀)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 自洽导入:工具可能被任意 cwd 调用,把仓库根(本文件上两级)挂进 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 边机/云机固定卡数(§5.2 第 7 类输入;异构机器时提升为 CLI)
EDGE_CARDS = 1
CLOUD_CARDS = 8

_SCENES = ("single_instance", "edge_share", "cloud_share", "lwd_cluster")
# 场景 -> (边机数, 云机数);机器清单长度必须与之匹配(§4.3)
_SCENE_SHAPE = {
    "single_instance": (1, 1),
    "edge_share": (1, None),      # 1 边机,N 云机
    "cloud_share": (None, 1),     # N 边机,1 云机
    "lwd_cluster": (None, None),  # N/M 全连接
}


def _die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(2)


def build_topology(scene: str, edge_machines: list[str], cloud_machines: list[str],
                   dp: int, port_base: int,
                   early_recv: bool, scramble: bool) -> dict:
    """六类输入 -> 5 大块 dict(与加载器逐字段对齐)。"""
    want_edge, want_cloud = _SCENE_SHAPE[scene]
    for role, machines in (("edge", edge_machines), ("cloud", cloud_machines)):
        if not machines or len(set(machines)) != len(machines):
            _die(f"{role} machine addresses must be non-empty and unique")
    if want_edge is not None and len(edge_machines) != want_edge:
        _die(f"scene {scene} requires {want_edge} edge machine(s), got {len(edge_machines)}")
    if want_cloud is not None and len(cloud_machines) != want_cloud:
        _die(f"scene {scene} requires {want_cloud} cloud machine(s), got {len(cloud_machines)}")
    if dp < 1 or CLOUD_CARDS % dp:
        _die(f"dp={dp} must divide {CLOUD_CARDS} cloud cards; edge DPs share one rank")

    # 边侧实例:edge_share = 每朵云配一个边实例(共享同一台边机,rank 同为 0);
    # 其余场景 = 每台边机一个实例(实例 i 持 rank i)。边机 1 卡不可分,
    # 每个 dp 的 ranks 恒为 [rank](长度 1,各 dp 共用同一 rank,§4.2)
    edge_instances = len(cloud_machines) if scene == "edge_share" else len(edge_machines)
    edges = []
    for i in range(edge_instances):
        addr = edge_machines[0] if scene == "edge_share" else edge_machines[i]
        rank = 0 if scene == "edge_share" else i
        edges.append({
            "id": i,
            "dp": [{"dp_idx": d, "addr": addr, "ranks": [rank]}
                   for d in range(dp)],
        })

    # 云侧:每台云机占连续 CLOUD_CARDS 个 rank,机内按 dp_idx 均分
    cloud_start = len(edge_machines) * EDGE_CARDS
    clouds = []
    for i, addr in enumerate(cloud_machines):
        base = cloud_start + i * CLOUD_CARDS
        per_dp = CLOUD_CARDS // dp
        clouds.append({
            "id": i,
            "dp": [{"dp_idx": d, "addr": addr,
                    "ranks": list(range(base + d * per_dp, base + (d + 1) * per_dp)),
                    "ctrl_port": port_base + d}
                   for d in range(dp)],
        })

    world = cloud_start + len(cloud_machines) * CLOUD_CARDS
    if scene == "edge_share":
        links = [{"edge": i, "cloud": i} for i in range(len(cloud_machines))]
    else:
        links = [{"edge": e, "cloud": c}
                 for e in range(len(edges)) for c in range(len(clouds))]

    return {
        "deployment": {
            "mode": 0,
            "scene": scene,
            "hccl_world_size": world,
            "edges_num": len(edges),
            "clouds_num": len(clouds),
        },
        "feature_ctrl": {
            "enable_early_recv": early_recv,
            "enable_scramble": scramble,
        },
        "edges": edges,
        "clouds": clouds,
        "instance_links": links,
    }


_HEADER = """\
# =============================================================================
# LWD prefill_only 拓扑 · 场景 {scene} · 由 tools/gen_lwd_topology.py 生成
# -----------------------------------------------------------------------------
#   scene        = {scene}(机器 {n_edge} 边 × {n_cloud} 云,配对规则:{pairing})
#   world        = {world}       dp 数 = {dp}(边机 {edge_cards} 卡 / 云机 {cloud_cards} 卡,
#                               dp 只切分/共享本机卡,不新增 rank)
#   控制面       云 dp ctrl_port: {ports}
#               (同设备多 dp 错开 = port_base+dp_idx,异设备可重新从 {port_base} 开始)
#   生成命令     {cmdline}
#   数据面       每条连接各有独立 UP/DOWN 组;广播域按云 DP 去重
{connections}
#   校验         生成器已按 LwdTopology.from_dict 全量校验(结构/rank 铺满/
#               端口错开/配对一致);两侧使用内容必须完全一致
# =============================================================================
"""


def output_path(output: str, dp: int, *, batch: bool) -> Path:
    """Keep the canonical single-profile filename when a directory is given."""
    path = Path(output)
    if path.is_dir() or output.endswith(("/", "\\")):
        name = f"lwd_config_{dp}dp.yaml" if batch else "lwd_config.yaml"
        return path / name
    if batch:
        return path.with_name(f"{path.stem}_{dp}dp{path.suffix or '.yaml'}")
    return path


def connection_comments(raw: dict) -> str:
    """Show the same global connection order used to create data-plane groups."""
    edges = {item["id"]: item for item in raw["edges"]}
    clouds = {item["id"]: item for item in raw["clouds"]}
    lines = []
    for link in sorted(raw["instance_links"], key=lambda item: (item["edge"], item["cloud"])):
        edge_id, cloud_id = link["edge"], link["cloud"]
        cloud_dps = {item["dp_idx"]: item for item in clouds[cloud_id]["dp"]}
        for edge in edges[edge_id]["dp"]:
            dp_idx = edge["dp_idx"]
            cloud = cloud_dps[dp_idx]
            pair = [edge["ranks"][0], cloud["ranks"][0]]
            lines.append(
                f"#   link=({edge_id},{cloud_id},{dp_idx}) "
                f"UP={pair} DOWN={pair} cloud_dp_broadcast={cloud['ranks']}"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="LWD topology generator (design doc §5.2)")
    parser.add_argument("--scene", required=True, choices=_SCENES)
    parser.add_argument("--edge-machines", required=True,
                        help="边机地址清单,逗号分隔(按 rank 顺序)")
    parser.add_argument("--cloud-machines", required=True,
                        help="云机地址清单,逗号分隔(按 rank 顺序)")
    parser.add_argument("--dp", default="1",
                        help="dp 数,可逗号分隔批量(如 1,2);默认 1")
    parser.add_argument("--port-base", type=int, default=5550)
    parser.add_argument("--enable-early-recv", action="store_true")
    parser.add_argument("--enable-scramble", action="store_true")
    parser.add_argument("-o", "--output", required=True,
                        help="输出路径;--dp 批量时为不含后缀的前缀或目录")
    args = parser.parse_args()

    edge_machines = [m.strip() for m in args.edge_machines.split(",") if m.strip()]
    cloud_machines = [m.strip() for m in args.cloud_machines.split(",") if m.strip()]
    try:
        dps = [int(d) for d in args.dp.split(",")]
    except ValueError:
        parser.error("--dp must contain comma-separated integers, e.g. 1,2")
    if len(set(dps)) != len(dps):
        parser.error("--dp must not contain duplicate values")
    if not edge_machines or not cloud_machines:
        _die("machine lists must not be empty")

    # 校验器与加载器同源:生成即校验(§5.1 "生成器必须先跑校验再落盘")。
    # lwd_topology 只依赖 yaml/stdlib:完整环境走包导入;无 torch 的
    # 环境(本机/CI 轻量跑)按文件路径独立加载,绕过 vllm/__init__
    def _load_validator():
        try:
            from vllm.config.lwd_topology import LwdTopology
            return LwdTopology
        except Exception:
            from importlib import util as iutil
            path = Path(__file__).resolve().parents[1] / "vllm/config/lwd_topology.py"
            spec = iutil.spec_from_file_location("lwd_topology_standalone", path)
            module = iutil.module_from_spec(spec)
            sys.modules[spec.name] = module  # dataclasses 注解处理需可查模块
            spec.loader.exec_module(module)
            return module.LwdTopology

    LwdTopology = _load_validator()

    import yaml
    outputs: list[tuple[Path, str]] = []
    for dp in dps:
        raw = build_topology(args.scene, edge_machines, cloud_machines, dp,
                             args.port_base, args.enable_early_recv, args.enable_scramble)
        try:
            LwdTopology.from_dict(raw)  # 结构/rank/端口/配对全量校验
        except ValueError as exc:
            _die(f"generated topology failed validation (dp={dp}): {exc}")
        out = output_path(args.output, dp, batch=len(dps) > 1)
        ports = " ".join(
            f"c{c['id']}.dp{d['dp_idx']}={d['ctrl_port']}"
            for c in raw["clouds"] for d in c["dp"])
        pairing = ("一一对应 eᵢ↔cᵢ" if args.scene == "edge_share" else "全连接")
        header = _HEADER.format(
            scene=args.scene, n_edge=len(edge_machines), n_cloud=len(cloud_machines),
            pairing=pairing, world=raw["deployment"]["hccl_world_size"], dp=dp,
            ports=ports, port_base=args.port_base,
            edge_cards=EDGE_CARDS, cloud_cards=CLOUD_CARDS,
            connections=connection_comments(raw),
            cmdline=" ".join(sys.argv[1:]))
        body = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
        outputs.append((out, header + body))

    for out, content in outputs:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
        print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
