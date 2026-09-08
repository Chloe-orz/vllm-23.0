"""控制面通信层(传输层,side-agnostic):通知协议 + 纯收发句柄 + 方向原语。

构成:lwd_notify(通知定义与编解码,纯协议,零 zmq/线程)、
lwd_control_communicator(ZMQ socket 收发句柄,无线程)、
lwd_control_publisher(OUTBOUND,自有线程+有界队列)、
lwd_control_subscriber(INBOUND,自有线程+桥接队列)。传输层只认方向,
不认边/云;侧别与 bind/connect 是装配期 wiring(control_scheduler 侧
assemble 文件决定),新增平面/角色互换不触碰本包。

依赖方向:本包只依赖 stdlib/msgspec/zmq/vllm.logger,被
control_scheduler 装配层消费;不 import 任何调度/执行类。
"""
