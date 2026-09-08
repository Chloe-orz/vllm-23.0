# control_communication 组合化重构方案(communicator 去线程化)

> 范围:`vllm/v1/lwd_control/control_communication/` 包内封闭重排(4 文件)
> 标准:refactor-code skill(≤50 有效行/函数、嵌套≤4、Lwd 命名、注释精简、直改仓)
> 上游关系:本包为 fork-only(上游 vLLM 无 lwd_control 目录),零上游同步面;
> 消费面(control_scheduler 装配层)零改动,不触碰 `tools/lwd_check_budget.py` 任何预算项

## 1. 现状问题(量化)

| # | 问题 | 证据 |
|---|------|------|
| P1 | 倒置构造:基类 `__init__` 启动线程,线程体读派生类状态;"派生类自有状态须先于 super().__init__ 初始化"仅由 docstring 约束,无机制保护,publisher 现有初始化顺序是唯一防线,违反即与已启动线程构成数据竞争 | 旧 communicator.py:29-45(线程启动)、旧 publisher.py:43-44、docstring 第 31 行 |
| P2 | 基类混合两个变化轴:ZMQ socket 句柄 + 线程生命周期焊死,无法获得不带线程的纯收发器 | 旧 communicator.py:18-63 |
| P3 | 纯复用继承:全仓无任何以 `LwdControlCommunicator` 类型的多态调用点(仅两个子类继承声明),Template Method 只买到代码复用 | 全仓 grep 仅 publisher.py:29 / subscriber.py:28 两处 |
| P4 | 收发不可独立测试:send/recv 与线程机制耦合,本包 0 测试 | tests 缺失 |
| P5 | 关停声明失实(冒烟实测发现的存量缺陷):docstring 声称 "close(0) 使阻塞 recv 以 ZMQError 退出",但实测(macOS/pyzmq)跨线程 close **不唤醒**阻塞 recv——subscriber 的"线程唯一退出路径"从未生效,每次关停泄漏一个永久阻塞线程、context 永不 term;旧代码无测试故从未暴露 | 旧 subscriber.py:58-60;冒烟输出 join(2s) 超时 |

## 2. 目标与非目标

目标:

- `LwdControlCommunicator` 瘦身为纯收发 socket 句柄:`send`/`recv`/`close`/`terminate`,无线程、单线程亲和;类名与文件名不变(沿用代码库既有词汇,不引入新概念)
- `LwdControlPublisher` / `LwdControlSubscriber` 脱离继承,各自拥有线程(关停时序在本类内编排),以成员组合 communicator;线程在全部自有状态就绪后启动(结构性消除 P1)
- 公开 API 与构造签名不变:`publish`/`drain`/`shutdown`、`(endpoint, *, bind[, queue_max])`

非目标:

- 不改 assemble/调度层任何调用点(edge_assemble / edge_scheduler / cloud_assemble / cloud_core 4 文件零改动)
- 不抽 ABC/Protocol(无可预见的第二传输实现;需要测试替身时再以 `typing.Protocol` 升级)
- 不改背压/关停语义(关停机制后按 P5 冒烟结论修正,见 §4/§6)

> 追记(同日二次小步):词根统一 wire/message → notify,`lwd_message.py` 经 git mv
> 改名 `lwd_notify.py`,`LwdWireMessage`→`LwdNotify`,encode/decode_wire→notify,
> 消费 import 路径与预算工具例外键随迁;详见主台账 §10.9。原"不动 lwd_message.py"
> 非目标仅约束组合化这一步。

## 3. 重构后结构与职责

| 类 | 职责 | 线程 |
|----|------|------|
| `LwdControlCommunicator` | ZMQ socket 句柄:send/recv/close/terminate;单线程亲和,close 是唯一允许跨线程的调用(打断阻塞收发) | 无 |
| `LwdControlPublisher` | 有界队列 + 自有线程:队列 → 编码 → send;publish 队满返回 False 背压 | 1(lwd-publisher) |
| `LwdControlSubscriber` | deque+锁 + 自有线程:recv → 解码 → 入队;drain 非阻塞取走 | 1(lwd-subscriber) |

## 4. 行为不变量(重构义务)

1. 关停有界且幂等(关停机制按 P5 修正):subscriber 为 close(释放 fd,不保证唤醒)→ `terminate()`(term 使阻塞 recv 以 ETERM 返回,可靠打断)→ `join(2s)`;publisher 为哨兵入队 → `join(2s)` → 仍存活(线程卡死在阻塞 send)则以 term 兜底再 join(1s)。不再保留旧"线程已死才 term"时序——term 正是 ZMQ 认可的跨线程打断手段,term 返回即线程收尾完毕
2. `LINGER=2000`;socket 只被所属线程或关停路径关闭;pyzmq 重复 close 安全
3. publish 队满返回 False(可预期失败,不抛异常不丢已发消息);PUSH 阻塞传导对端背压
4. 坏包(msgspec DecodeError/ValidationError)丢弃 + warning,不中断接收
5. drain 保持到达序;两侧 shutdown 幂等
6. publisher 关停令 `queue.put(timeout=1.0)` + suppress `Full`(线程卡死时随进程退出)

## 5. 职责映射(旧 → 新)

| 旧 | 新 | 变化 |
|----|----|------|
| `LwdControlCommunicator.__init__`(建 socket+启动线程) | `LwdControlCommunicator.__init__`(仅建 socket)+ 两侧 `__init__` 各自建线程并最后启动 | 拆分 |
| `LwdControlCommunicator.shutdown`(统一关停) | `Publisher.shutdown` / `Subscriber.shutdown`(各自幂等实现;P5 修正:term 作可靠打断) | 拆分+修正 |
| `LwdControlCommunicator._communicator_thread` | `Publisher._send_thread` / `Subscriber._receive_thread` | 拆分 |
| `LwdControlCommunicator._request_stop` | 哨兵入队并入 `Publisher.shutdown`;`close()` 并入 `Subscriber.shutdown` | 合并 |
| `Publisher.publish` / `Subscriber.drain` | 原样保留 | 不变 |

## 6. 迁移步骤与落地状态

- [x] 精简设计文档归档(本文档)
- [x] communicator 瘦身为纯收发句柄(无线程)
- [x] publisher / subscriber 脱离继承,组合 communicator + 自有线程
- [x] 包文档(`__init__.py`)同步更新
- [x] 自查:`check_functions.py` 50 行/4 层全过;`ruff check` + `ruff format` 全过;
      `lwd_check_budget.py` 达标;消费面构造签名核对一致;无 `_communicator_thread`/
      `_request_stop` 残留引用
- [x] 冒烟(ipc PUSH/PULL,stub vllm.logger):203 条消息 FIFO 保序、drain 取空、
      双侧关停毫秒级且幂等、无线程泄漏;**该冒烟暴露并修复 P5**(修正前 sub 关停
      2.01s 超时,线程泄漏)
- [x] 词根统一二次小步:wire/message → notify(模块 git mv + 符号 + 消费 import +
      预算工具键),改名后 ruff/check_functions/budget/冒烟复跑全绿

后续建议(独立小步,不混入本次):把冒烟固化为仓内回归测试(放点需与仓内测试布局
约定对齐),防止关停语义回退。
