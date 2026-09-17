"""边/云引擎公共基类:调度器注入时序与收发两面生命周期。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
    LwdControlPublisher,
)
from vllm.v1.lwd_control.control_communication.lwd_control_subscriber import (
    LwdControlSubscriber,
)

if TYPE_CHECKING:
    from vllm.config.lwd import LwdConfig


class LwdBaseEngineCore(EngineCoreProc):
    """边/云 EngineCore 公共基类。

    子类经 lwd_scheduler_cls 声明调度器类;注入必须发生在 super() 之前
    (super 构建 scheduler 时一次性消费 scheduler_cls,后设无效)。收发
    两面构造前为 None,关停容忍未建。"""

    lwd_scheduler_cls: type

    def __init__(self, *args, **kwargs) -> None:
        self._publisher: LwdControlPublisher | None = None
        self._subscriber: LwdControlSubscriber | None = None
        kwargs["vllm_config"].scheduler_config.scheduler_cls = self.lwd_scheduler_cls
        super().__init__(*args, **kwargs)
        self.lwd_config: LwdConfig = kwargs["vllm_config"].lwd_config

    def _lwd_setup_planes(self) -> None:
        """建收发两面,发布面回填调度器,起接收线程;端点/编解码由
        子类的两个构建钩子指定。"""
        self._subscriber = self._lwd_build_subscriber()
        self._publisher = self._lwd_build_publisher()
        self.scheduler.lwd_publisher = self._publisher
        self._subscriber.start(self._lwd_on_message)

    def _lwd_build_subscriber(self) -> LwdControlSubscriber:
        raise NotImplementedError

    def _lwd_build_publisher(self) -> LwdControlPublisher:
        raise NotImplementedError

    def _lwd_shutdown_planes(self) -> None:
        """收发两面关停(幂等;装配失败路径两面可能未建)。"""
        if self._subscriber is not None:
            self._subscriber.shutdown()
        if self._publisher is not None:
            self._publisher.shutdown()

    def shutdown(self) -> None:
        """收发两面关停后走原生(幂等)。"""
        self._lwd_shutdown_planes()
        super().shutdown()

    def _lwd_on_message(self, msg) -> None:
        """接收线程消息路由钩子(循环骨架与线程在 Subscriber.start)。"""
        raise NotImplementedError
