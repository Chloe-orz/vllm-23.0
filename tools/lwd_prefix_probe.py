"""云侧前缀协商 HTTP 探针(替代手工 curl)。

对运行中的云(1E1C 下 `lwd_coordination.enabled=true`)发真实 HTTP 探测,
解析响应头里的 ``X-Edge-Cloud-Prefix-Hit-Tokens``;不等于 ``--expect-hit``
(默认 0)即非零退出,可挂进脚本/CI。

用法示例:

  python tools/lwd_prefix_probe.py \
    --control-url http://127.0.0.1:8100/v1/chat/completions \
    --tenant-key-file /path/tenant.key \
    --edge-id 0 --block-size 16 --prompt-tokens 32 \
    --request-id probe-1 --expect-hit 0
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import aiohttp

from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_client import (
    LwdEdgePrefixClient,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-url", required=True,
                        help="云控制面 URL(http://<cloud>:8100/v1/chat/completions)")
    parser.add_argument("--tenant-key-file", required=True,
                        help="租户密钥文件(≥16B,仅边持有)")
    parser.add_argument("--edge-id", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=16,
                        help="manifest 块粒度(须为云 hash 块粒度的整数倍)")
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--request-id", default="probe")
    parser.add_argument("--expect-hit", type=int, default=0,
                        help="期望 hit_tokens;不符则非零退出")
    return parser.parse_args()


async def _probe(args: argparse.Namespace) -> int:
    with open(args.tenant_key_file, "rb") as f:
        tenant_key = f.read()
    client = LwdEdgePrefixClient(
        args.control_url, args.block_size, tenant_key, edge_id=args.edge_id
    )
    prompt_token_ids = list(range(args.prompt_tokens))
    async with aiohttp.ClientSession() as session:
        result = await client.negotiate(session, args.request_id, prompt_token_ids)
    print(
        "[lwd-probe] request_id=%s instance=%s block_size=%d "
        "hit_blocks=%d hit_tokens=%d",
        result.request_id, result.instance_id, result.block_size,
        result.hit_blocks, result.hit_tokens,
    )
    return 0 if result.hit_tokens == args.expect_hit else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_probe(_parse_args())))