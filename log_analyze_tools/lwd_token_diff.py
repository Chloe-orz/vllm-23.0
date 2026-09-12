#!/usr/bin/env python3
"""lwd_token_diff: 边云逐 token 对账。

从边/云两侧日志提取每个请求逐步产出的 token_id,拼接成完整序列,
对比两边是否一致,定位首个发散点与全部错位。

日志锚点(均为现有 [Lwd]/[lwd-*] 探针,无需改代码):
  边: [lwd-token-dbg] req=R tokens=[ids] text='..'     — 每步交付的 id
  云: [lwd-finish-dbg] req=R out_len=N last_tok=T       — 每步最后一个 id
      [Lwd][cloud-sample] intended sampled ids=[[ids]]  — 每步全部采样 id
                                                        (spec 步含多 id,-1 为无效位)

用法:
  python3 lwd_token_diff.py --edge edge.log --cloud cloud.log
  python3 lwd_token_diff.py --edge e.log --cloud c.log --req chatcmpl-xxx
  python3 lwd_token_diff.py --edge e.log --cloud c.log --tokenizer /path/to/model

说明:
  * 云侧序列优先用 intended(单请求日志时,spec 步也不丢 token);
    多请求或无 intended 时用 finish-dbg 的 last_tok 链(非 spec 每步
    恰一个 token,准确;spec 步只保留每步最后一个,会漏)。
  * 退出码: 全一致 0,有差异/缺数据 1,可直接接 CI。
"""

from __future__ import annotations

import argparse
import ast
import re
import sys

RE_EDGE_TOKEN = re.compile(
    r"\[lwd-token-dbg\] req=(\S+) tokens=\[([^\]]*)\]"
)
RE_CLOUD_FINISH = re.compile(
    r"\[lwd-finish-dbg\] req=(\S+) out_len=(\d+) last_tok=(-?\d+)"
)
RE_CLOUD_INTENDED = re.compile(
    r"intended sampled ids=(\[\[.*?\]\])"
)
RE_EDGE_RECOVERED = re.compile(
    r"\[seqno=(\d+)\] RECV DOWN recovered token_ids=(\[\[.*?\]\])"
)


def parse_ids(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def read_lines(path: str) -> list[str]:
    """读日志行:自动识别编码(UTF-16 BOM / UTF-8 BOM / UTF-8 / GB18030),
    并剥离 ANSI 颜色码(终端重定向的日志常带,会干扰肉眼看不影响正则,
    但统一剥掉最干净)。"""
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    elif raw.startswith(b"\xef\xbb\xbf"):
        text = raw.decode("utf-8-sig")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("gb18030", errors="replace")
    return [ANSI_RE.sub("", l) for l in text.splitlines()]


def diagnose(path: str, side: str) -> None:
    """无匹配时的现场诊断:这份文件里到底有没有探针行。"""
    lines = read_lines(path)
    n_lwd = sum(1 for l in lines if "[Lwd]" in l or "[lwd-" in l)
    n_tok = sum(1 for l in lines if "token-dbg" in l)
    n_fin = sum(1 for l in lines if "finish-dbg" in l)
    n_int = sum(1 for l in lines if "intended sampled ids" in l)
    print(f"  [{side}] {path}: 共 {len(lines)} 行 | Lwd 探针行 {n_lwd} | "
          f"token-dbg {n_tok} | finish-dbg {n_fin} | intended {n_int}")
    if n_lwd == 0 and len(lines) > 0:
        print(f"    → 文件里没有任何 Lwd 探针输出:确认拿的是 EngineCore "
              "所在进程的日志(不是别的文件),且运行的代码带 lwd_debug 探针")
    if n_tok > 0 and side == "cloud":
        print("    → 这份文件含边侧锚点(token-dbg),可能传反了或是合并粘贴件")


def extract_edge(path: str) -> dict[str, list[int]]:
    """边侧: token-dbg 按出现顺序拼接(即交付顺序)。"""
    seqs: dict[str, list[int]] = {}
    for line in read_lines(path):
        m = RE_EDGE_TOKEN.search(line)
        if m:
            ids = parse_ids(m.group(2))
            if -1 in ids:  # spec 无效位不计交付
                ids = [t for t in ids if t != -1]
            seqs.setdefault(m.group(1), []).extend(ids)
    return seqs


def extract_cloud(path: str) -> tuple[dict[str, list[int]], list[int]]:
    """云侧: finish-dbg 的 last_tok 链(按 out_len 排序) + intended 全量。"""
    steps: dict[str, dict[int, int]] = {}
    intended: list[int] = []
    for line in read_lines(path):
        m = RE_CLOUD_FINISH.search(line)
        if m:
            req, out_len, tok = m.group(1), int(m.group(2)), int(m.group(3))
            steps.setdefault(req, {})[out_len] = tok
            continue
        m = RE_CLOUD_INTENDED.search(line)
        if m:
            try:
                rows = ast.literal_eval(m.group(1))
                intended.extend(
                    t for row in rows for t in row if t != -1
                )
            except (ValueError, SyntaxError):
                pass
    seqs = {
        req: [toks[i] for i in sorted(toks)] for req, toks in steps.items()
    }
    return seqs, intended


def compare(
    edge: dict[str, list[int]],
    cloud: dict[str, list[int]],
    req_filter: str | None,
    edge_path: str | None = None,
    cloud_path: str | None = None,
) -> bool:
    reqs = sorted(set(edge) | set(cloud))
    if req_filter:
        reqs = [r for r in reqs if req_filter in r]
    if not reqs:
        print("没有提取到任何请求。现场诊断:")
        if req_filter:
            print(f"  [--req {req_filter}] 过滤后为空——先去掉 --req 跑一次,"
                  "看日志里实际有哪些 req_id。")
        if edge_path:
            diagnose(edge_path, "edge")
        if cloud_path:
            diagnose(cloud_path, "cloud")
        return False

    all_match = True
    for req in reqs:
        e, c = edge.get(req), cloud.get(req)
        print(f"\n== req {req} ==")
        if e is None:
            print("  边侧日志缺此请求")
            all_match = False
            continue
        if c is None:
            print("  云侧日志缺此请求")
            all_match = False
            continue
        print(f"  cloud n={len(c)}: {c}")
        print(f"  edge  n={len(e)}: {e}")
        if c == e:
            print("  MATCH ✅")
            continue
        all_match = False
        diffs = [
            (i, a, b) for i, (a, b) in enumerate(zip(c, e)) if a != b
        ]
        first = diffs[0][0] if diffs else min(len(c), len(e))
        print(f"  MISMATCH ❌ 首个发散 idx={first} "
              f"长度 cloud={len(c)} edge={len(e)} 错位 {len(diffs)} 处")
        for i, a, b in diffs[:10]:
            print(f"    idx={i}: cloud={a}  edge={b}")
    return all_match


def decode(tokenizer_path: str, seqs: list[list[int]]) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("(未安装 transformers,跳过文本解码)")
        return
    tok = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )
    for label, ids in seqs:
        if ids:
            print(f"  {label} decoded: {tok.decode(ids)!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--edge", required=True, help="边侧日志文件")
    ap.add_argument("--cloud", required=True, help="云侧日志文件")
    ap.add_argument("--req", default=None, help="只对比 req_id 含此子串的请求")
    ap.add_argument("--tokenizer", default=None,
                    help="模型目录,给出则附文本解码")
    args = ap.parse_args()

    edge = extract_edge(args.edge)
    cloud_steps, intended = extract_cloud(args.cloud)

    # 云侧序列选择: 单请求且有 intended → 用 intended(spec 准确);
    # 否则用 finish-dbg last_tok 链。
    if len(cloud_steps) == 1 and intended:
        req = next(iter(cloud_steps))
        cloud = {req: intended}
        print(f"[i] 云侧使用 intended 全量序列(req={req}, n={len(intended)})")
    else:
        cloud = cloud_steps
        print("[i] 云侧使用 finish-dbg last_tok 链"
              "(spec 步仅保留最后一个 token)")

    ok = compare(edge, cloud, args.req, args.edge, args.cloud)

    if args.tokenizer:
        print("\n== 文本解码 ==")
        for req in sorted(set(edge) & set(cloud)):
            decode(args.tokenizer, [
                (f"cloud {req}:", cloud[req]),
                (f"edge  {req}:", edge[req]),
            ])

    print(f"\n结论: {'边云 token 序列完全一致 ✅' if ok else '存在差异 ❌'}")
    return 0 if ok else 1


# ------------------------------------------------------------------ #
# 自检: 用真实日志行验证正则; --selftest 运行
# ------------------------------------------------------------------ #
SAMPLE_EDGE = (
    "(EngineCore pid=14587) INFO 09-12 07:15:13 [lwd_debug.py:28] "
    "[lwd-token-dbg] req=chatcmpl-t tokens=[220] text=' ' finish=None\n"
    "(EngineCore pid=14587) INFO 09-12 07:15:13 [lwd_debug.py:28] "
    "[lwd-token-dbg] req=chatcmpl-t tokens=[96181] text='求' finish=None\n"
)
SAMPLE_CLOUD = (
    "(EngineCore pid=1) INFO [lwd_debug.py:28] [lwd-finish-dbg] "
    "req=chatcmpl-t out_len=1 last_tok=220 finish=-1\n"
    "(EngineCore pid=1) INFO [lwd_debug.py:28] [lwd-finish-dbg] "
    "req=chatcmpl-t out_len=2 last_tok=97900 finish=-1\n"
    "(EngineCore pid=1) INFO [lwd_cloud_model_runner.py:300] "
    "[Lwd][cloud-sample] intended sampled ids=[[220]]\n"
    "(EngineCore pid=1) INFO [lwd_cloud_model_runner.py:300] "
    "[Lwd][cloud-sample] intended sampled ids=[[97900]]\n"
)


def selftest() -> int:
    import tempfile, os  # noqa: E401
    with tempfile.TemporaryDirectory() as d:
        e = os.path.join(d, "e.log")
        c = os.path.join(d, "c.log")
        open(e, "w").write(SAMPLE_EDGE)
        open(c, "w").write(SAMPLE_CLOUD)
        edge = extract_edge(e)
        cloud, intended = extract_cloud(c)
        assert edge == {"chatcmpl-t": [220, 96181]}, edge
        assert intended == [220, 97900], intended
        # last_tok 链与边侧不一致(97900 vs 96181) → compare 应报 False
        assert not compare(edge, cloud_steps_to_map(cloud), None)
    print("selftest OK: 正则匹配真实日志格式,差异能被检出")
    return 0


def cloud_steps_to_map(cloud: dict) -> dict:
    return cloud


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main())
