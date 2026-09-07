"""诊断外观:全部 Lwd 日志/统计经 LwdLog 收敛,业务路径不直打日志(§2.5/W1/W7/W15)。"""

from __future__ import annotations


class LwdLog:
    """日志/统计外观(Facade):phase/admission/zombie/memory 四类事件统一出口。"""

    def phase(self, event: str, **fields) -> None:
        """相位推进类事件(原 [PO-*] phases)。"""
        ...

    def admission(self, event: str, **fields) -> None:
        """准入决策类事件(原 [PO-ADM],含云侧僵尸检测)。"""
        ...

    def zombie(self, request_id: str, **fields) -> None:
        """云侧长时无进展请求的可观测告警(§8.3 代价 2 的替代观测)。"""
        ...

    def memory(self, stats: dict) -> None:
        """内存/积压统计(原 [PO-MEM];数据面统计由其落位侧提供,§9.12)。"""
        ...


def lwd_log() -> LwdLog:
    """共享日志外观实例。"""
    ...
