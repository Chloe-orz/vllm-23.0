"""控制面通信层(传输层):通知协议 + 收发句柄 + 方向原语 + 云侧复用单通道。

1E1C(单边一云)沿用 Publisher/Subscriber 双单向通道 + HELLO 发现;
云侧复用(registry 非空)走 LwdControlRouterChannel——每进程单
ROUTER socket(云 bind 单端口、边 connect 全部云),双向靠 identity
首帧寻址,来源由信封携带。侧别与 bind/connect 由装配层决定,本层
不 import 调度/执行类。"""
