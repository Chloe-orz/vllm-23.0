"""LwdConfig:内核唯一配置入口,env/additional_config 只允许在本文件解析(§7.3-C3)。"""

from __future__ import annotations

from dataclasses import dataclass

# 控制面 ZMQ 端口(单向边->云;env 可覆盖,§3/§9)
LWD_PRE_OUT_PORT = 5558

LWD_DEBUG_WIRE_ENV = "VLLM_LWD_DEBUG_WIRE"


@dataclass(frozen=True)
class LwdConfig:
    """装配期一次成型,内核各模块只收 plain 值(零 env/零 getattr)。

    分块大小不自造(§9.9):沿用上游 scheduler_config 的 chunked prefill
    配置(max_num_batched_tokens);本配置只保留通道与调试开关。
    """

    pre_out_endpoint: str
    debug_wire: bool

    @classmethod
    def from_env_and_config(cls, vllm_config) -> LwdConfig:
        """工厂方法:additional_config.edge_cloud_config + env 覆盖一次解析成型。"""
        ...

    def validation_problems(self, vllm_config) -> list[str]:
        """互斥/约束校验(如边侧 TP 必须 =1,§8.3);返回问题清单,不抛异常。"""
        ...
