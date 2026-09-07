"""云侧数据面:tag 匹配按需接收 + embeds 缓存/拼接/消费释放(仅此文件用 dist,§8.3/§9.9)。

fill 完全对齐上游实现(gpu_model_runner.py:1949-1982 的 req_prompt_embeds
填充循环):该循环只依赖 ``.shape[0]`` 与切片访问,LwdCloudRemoteEmbeds 按
同款鸭子类型提供惰性视图,切片触达即按需接收,上游 fill 代码零改动。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from vllm.v1.lwd.lwd_config import LwdConfig
    from vllm.v1.lwd.lwd_message import LwdEmbedNotify


class LwdCloudRemoteEmbeds:
    """fill 点惰性拼接视图(对齐上游 prompt_embeds 访问面:shape/切片/len)。

    被抢占重 prefill 时从头再切片即可命中保留缓存(重计算不发回边侧,§9.2)。
    """

    def __init__(
        self, store: "LwdCloudEmbedStore", request_id: str, start: int, stop: int
    ) -> None:
        ...

    @property
    def shape(self) -> "tuple[int, ...]":
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, key):
        ...

    def numel(self) -> int:
        ...


class LwdCloudEmbedStore:
    """云侧 embeds 仓:预告登记 -> 按需 recv(tag)-> 拼接 -> 消费释放/保留/abort 丢弃。

    范围由原生调度决策界定(notify 携带 offset/num_tokens);
    retain 集合是被抢占待重算的请求:其已收 embeds 保留在缓存,重 prefill
    直接重放;abort 请求整仓丢弃(在途 recv 由 tag 无人认领作废)。
    """

    def __init__(self, config: LwdConfig) -> None:
        ...

    def lwd_register_request(self, request_id: str, num_prompt_tokens: int) -> None:
        ...

    def lwd_on_embed_notify(self, notify: LwdEmbedNotify) -> None:
        """预告登记:seqno->request 映射 + recv 范围与张量形状推导。"""
        ...

    def lwd_ensure_range(
        self, request_id: str, offset: int, num_tokens: int
    ) -> "torch.Tensor":
        """缓存命中直接返回;未命中同步 recv(tag=BASE+seqno,阻塞语义为 §8.3 显式接受)。"""
        ...

    def _lwd_recv_range(self, request_id: str, offset: int) -> "torch.Tensor":
        ...

    def lwd_gather_range(
        self, request_id: str, start: int, stop: int
    ) -> LwdCloudRemoteEmbeds:
        """fill 点数据源入口(挂在请求 prompt_embeds 位置,上游 fill 循环直读)。"""
        ...

    def lwd_release_consumed(self, request_id: str, consumed_upto: int) -> None:
        """按消费位置释放已消费 embeds;retain 集合内请求跳过(重计算保留,§9.2)。"""
        ...

    def lwd_set_retain_requests(self, retain: "set[str]") -> None:
        """设置保留集合(被抢占待重 prefill 的请求);重算完成或 abort 后解除。"""
        ...

    def lwd_drop_request(self, request_id: str) -> None:
        """abort 路径:清请求全部登记与缓存。"""
        ...

    def lwd_num_live_ranges(self, request_id: str) -> int:
        ...

    def lwd_stats(self) -> dict:
        """只读统计接口([PO-MEM] 收敛;单向无水位,这是云侧内存唯一观测口)。"""
        ...
