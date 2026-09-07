# prefill_only 弱耦合迁移重构方案(按 refactor-code 标准)

> 目标仓:`/Users/wangwei/prefill_only_base_br`(`vllm` + `vllm-ascend`)
> 标准:refactor-code skill(≤50 有效行/函数、嵌套≤4、Lwd 命名前缀、注释精简、可预期失败不抛异常、直改仓 additive)
> 输入文档:`prefill_only_control_plane_changes.md`(结构)、`prefill_only_whitebox_refactor_notes.md`(W1-W20 观察点)、`prefill_only_migration_assessment.md`(依赖与冲突评估)
> 本文档为设计文档底稿;动代码前按标准流程精简归档到目标仓 `docs/refactor/prefill_only_migration.md`。

> **⛔ 硬约束(后补,优先级最高):只允许修改 vllm 仓,`vllm-ascend` 保持零改动。**
> 该约束推翻 §3 的 ascend 侧设计与 §7 的部分收敛手段,替代方案见 **§8(仅改 vllm 仓的约束变体)**——模块树迁至 `vllm/v1/lwd/`,数据面弃用 `edge_cloud_comm` 改用 torch.distributed 带 tag 直传,所有 monkey-patch 降级为 in-tree 守卫分支。§8 与前文冲突处,以 §8 为准。

---

## 0. 总体策略:迁移即重构,结构重写、语义保真

- **一步到位**:不做"先原样平移再规范化"的两段式(两轮改动、两轮风险、两轮评审)。落位时直接按 refactor-code 标准写。
- **语义保真**:函数拆分、命名、错误处理按标准重写;**功能不删不减**(如 seqnos 旁路、retain 门控等行为全部保留)。功能瘦身(砍优化旁路、批量派发)是迁移完成后的独立小步,不混入本次。
- **弱耦合的操作化定义**(可量化验收):
  1. PO 内核对 PD/passive/其他 edge_cloud 模式符号的 import 数 = **0**;
  2. 主仓 additive 契约面 = **2 个 config 字段 + 2 个线上结构(LwdChunkPlan / EdgeEmbedChunk)+ 4 个可选字段(EngineCoreRequest 1 个 + SchedulerOutput 3 个)+ 1 个枚举值 + 3 个空默认钩子**(逐项台账见 §7);
  3. 模式判定函数全仓只有 **1 个**实现(现状 5 处,W9);
  4. 对上游既有文件的改动全部 **additive**(新增可选字段/空默认钩子/守卫分支),不改任何原有行为行;
  5. 内核模块的外部交互受 **import 白名单 + 交互预算** 约束(§7),超标即 CI fail。

---

## 1. 分层架构(弱耦合骨架)

```
┌─ L3 挂钩层(唯一允许触碰既有文件的位置)─────────────────────┐
│  ascend: lwd_edge_engine_hooks / lwd_cloud_launch(精简重写,  │
│          仅 PO 挂钩;原 patch_engine_core/serve_headless 的  │
│          PD 主体不迁)                                        │
├─ L2 PO 内核(全新目录 vllm_ascend/lwd/,全 Lwd 命名)─────────┤
│  lwd_message.py / lwd_edge_channel.py / lwd_cloud_channel.py │
│  lwd_edge_dispatcher.py / lwd_cloud_engine.py                │
│  lwd_cloud_phase_scheduler.py / lwd_cloud_recv_manager.py    │
│  只依赖 L1 显式接口,不 import 任何 PD/passive 符号           │
├─ L1 底座最小集 ──────────────────────────────────────────────┤
│  vllm 主仓(additive):ParallelConfig 2 字段;                │
│    SchedulerOutput 3 可选字段 + BatchType.EDGE_EMBED;        │
│    multiproc_executor 本地环分支;worker_base 索引修正;       │
│    parallel_state 世界布局最小化(见 §3);utils 判定函数      │
│  ascend:edge_cloud_comm 整迁(8 文件 1608 行,原样平移);     │
│    端口常量(替代 pd_separation_config)                      │
├─ L0 目标仓原生 v0.23.0(不动)────────────────────────────────┤
│  AsyncScheduler / step_with_batch_queue / prompt_embeds /    │
│  make_empty / run_headless(已核实目标仓齐备且与源基准一致)  │
└──────────────────────────────────────────────────────────────┘
```

依赖方向严格单向:L3 → L2 → L1 → L0。L2 内部同样单向:
`lwd_message`(纯数据)← `lwd_*_channel` ← `dispatcher/engine` ← `scheduler/recv_manager`。

---

## 2. 按标准重写要点(逐条消化白盒观察点)

### 2.1 消解 200 行拷贝与三层 step 覆盖(W10 + W8)

源方案拷贝整个 `step_with_batch_queue`(~200 行)再注入 4 处增强;迁移方案改为**在目标仓 core.py 上 additive 加 3 个空默认钩子**,云引擎只写差异:

```python
# vllm/v1/engine/core.py  (additive,默认空实现,上游行为不变)
def _make_empty_batch_future(self) -> Future | None:
    """0-token batch 的 future 工厂;None=走原生 worker 往返。"""
    return None

def _before_execute_model(self, scheduler_output: SchedulerOutput) -> None:
    """execute_model 前的元数据附加点(默认 no-op)。"""

def _check_batch_output_consistency(self, scheduler_output, model_output) -> None:
    """update_from_output 前的一致性预检(默认 no-op)。"""
```

`LwdCloudEngine(EngineCore)` 覆写这三个钩子(空 batch 预完成 future、附加 seqnos/retain、一致性预检),step 主体**零拷贝**。已核实:目标仓 core.py 与源仓的该函数无漂移(42 行 diff 全在日志/kv-config),钩子方案安全。
注:core.py 的 3 个钩子属于"上游暂不可能接收的自研扩展",按标准 §5 收敛记录在设计文档例外表中。

### 2.2 请求清理收敛(W11)

三处重复的 4-5 项登记表清理 → `LwdCloudEngine._lwd_release_request(request_id)` 唯一实现,abort/finished_requests/finish_reason 三条路径全部调它。

### 2.3 模式判定收敛(W9)

```python
# vllm_ascend/lwd/lwd_mode.py(唯一实现)
def is_lwd_prefill_only(vllm_config) -> bool: ...
```

主仓侧不 import ascend 代码(方向不许反),主仓内的分支判断只读 `parallel_config.enable_edge_cloud` 这一个 bool(L1 字段),不重复解析 `additional_config`。

### 2.4 通道背压对称化(W16,按"可预期失败不抛异常")

`LwdEdgeControlPublisher.publish` 队满不再"ERROR+丢弃":返回 `False`,dispatcher 将该步视为未派发(credit 不扣),下一步重试——与 subscriber 侧"元数据绝不丢"的阻塞语义对称,且天然并入现有步进节奏,不新增异常路径。

### 2.5 诊断收敛与同步点(W1 / W7 / W15)

- 所有 `[PO-*]` 日志与统计(phases/admission/zombie/memory)收敛到 `lwd_diagnostics.py`,业务路径只留 `LwdLog.phase(...)` 一类调用;
- worker 侧每 chunk 的 `absmean().item()` / `head8.tolist()` 采样默认关闭(`LWD_DEBUG_WIRE` 开关),消除每 chunk 两次 NPU→CPU 同步;
- `[PO-MEM]` 不再直戳 recv_manager 私有字段,由 `lwd_cloud_recv_manager` 暴露 `lwd_stats()` 只读接口。

### 2.6 函数拆分清单(迁移时直接按 ≤50 有效行落位)

| 源函数 | 行数 | 拆分后 |
|---|---|---|
| `_native_step_bq_prefill_only` | ~200 | 钩子化后消失(§2.1) |
| `run_active_engine_core` | ~180 | `_lwd_cloud_init_process` / `_lwd_cloud_connect_planes` / `_lwd_cloud_build_engine` + 主函数纯编排 |
| `try_dispatch_embed` | ~88 | `_lwd_edge_pick_candidate`(两道门)/ `_lwd_edge_notify_chunk` / `_lwd_edge_submit_chunk` |
| `setup_prefill_only_engine_core` | ~75 | `_lwd_edge_build_planes` / `_lwd_edge_read_credit` + 装配主体 |
| `_process_engine_step`(云) | ~66 | `_lwd_cloud_publish_results` / `_lwd_cloud_publish_consumed` / `_lwd_release_request` |
| `_apply_scheduling_policy` / `_publish_consumed_watermarks` / `_prefill_only_step` | 45-60 | 各拆 2-3 个私有子函数 |
| `_execute_model_embed_prefill_only`(worker) | ~65 | `_lwd_edge_embed_one_chunk` / 批处理编排 |

### 2.7 命名映射表(Lwd 判定表落地)

| 源名 | 新名 | 判定 |
|---|---|---|
| `EdgePrefillDispatcher` / `EdgePrefillRequestState` / `PrefillStatus` | `LwdEdgeDispatcher` / `LwdEdgeRequestState` / `LwdEdgeStatus` | 仅边侧 |
| `PrefillOnlyControlPublisher` / `CloudOutputReceiver` | `LwdEdgeControlPublisher` / `LwdEdgeResultReceiver` | 仅边侧 |
| `PrefillOnlyControlSubscriber` / `CloudResultPublisher` | `LwdCloudControlSubscriber` / `LwdCloudResultPublisher` | 仅云侧 |
| `ActiveEdgeCloudEngineCore` | `LwdCloudEngine` | 仅云侧 |
| `PurePhaseSchedulerBase` / `PrefillFirst*` / `DecodeFirst*` | `LwdCloudPhaseScheduler` / `LwdCloudPrefillFirstScheduler` / `LwdCloudDecodeFirstScheduler` | 仅云侧(scheduler_cls 只装在云) |
| `SeparatePhasesPolicy` / `ImmediateAdmissionPolicy` | `LwdCloudSeparatePhasesPolicy` / `LwdCloudImmediatePolicy` | 仅云侧 |
| `PrefillOnlyRecvManager` / `PrefillOnlyRemoteEmbeds` | `LwdCloudRecvManager` / `LwdCloudRemoteEmbeds` | 仅云侧 |
| `PrefillOnlyChunkPlan` | `LwdChunkPlan` | 两侧共用(线上结构) |
| `EdgeEmbedChunkNotify` / `EdgePrefillAbort` / `EdgeEmbedConsumed` | `LwdChunkNotify` / `LwdAbortSignal` / `LwdConsumedSignal` | 两侧共用 |
| `EdgeEmbedChunkAck` / `EdgeEmbedBatchAck` | `LwdEdgeChunkAck` / `LwdEdgeBatchAck` | 仅边侧进程内 |
| `is_prefill_only_edge_cloud` 等 5 处 | `is_lwd_prefill_only`(唯一) | 通用 |

模块文件同理:`vllm_ascend/lwd/` 下 `lwd_message.py`、`lwd_edge_channel.py`、`lwd_cloud_channel.py`、`lwd_edge_dispatcher.py`、`lwd_cloud_engine.py`、`lwd_cloud_phase_scheduler.py`、`lwd_cloud_recv_manager.py`、`lwd_diagnostics.py`、`lwd_mode.py`。

**例外(设计文档记录)**:
- `edge_cloud_comm/` 8 文件**原样平移不改名**——它是独立设计的数据面框架(有自己的设计文档),改名收益低、与源仓后续同步成本高;标准"前缀只约束自研新增代码",平移存量记为例外;
- 环境变量保留 `VLLM_ASCEND_` 生态前缀(`VLLM_ASCEND_LWD_CREDIT_PER_REQ` 等),与仓内其他 env 一致。

---

## 3. L1 底座最小集(弱耦合的边界工程)

### vllm 主仓(additive 清单,预计触碰 ~13 文件,全部守卫激活)

| 文件 | 增量 | 性质 |
|---|---|---|
| config/parallel.py、config/vllm.py、engine/arg_utils.py | `enable_edge_cloud` / `is_edge_node` 字段 + CLI/env 接线 | 新增字段 |
| v1/core/sched/output.py | `BatchType.EDGE_EMBED` + `edge_embed_chunks/seqnos/retain` 3 可选字段 | 新增枚举值/字段 |
| v1/executor/multiproc_executor.py | 本地环 MQ 分支(PO 判定下) | 新增分支 |
| v1/worker/worker_base.py | KV config 按 rpc_rank 索引(PO 判定下) | 新增分支 |
| v1/worker/gpu_model_runner.py | PO 云端 `is_first_rank=True`(PO 判定下) | 新增分支 |
| v1/engine/core.py | §2.1 的 3 个空默认钩子 | 新增钩子 |
| distributed/parallel_state.py | 世界布局**最小化版**:角色翻转 + `is_lwd_prefill_only()` 对应的主仓侧标志(不带 head_tail/embedding_only/PD 全家) | 新增分支/函数 |
| utils/__init__.py、v1/engine/utils.py、executor/abstract.py、ray_executor.py、multimodal/registry.py、shm_broadcast.py、kv_cache_utils.py | 各 7-19 行的 enable_edge_cloud 判断接线 | 新增分支 |

### vllm-ascend 仓

| 内容 | 方式 |
|---|---|
| `distributed/edge_cloud_comm/`(1608 行) | 整体平移,不改名 |
| 端口配置 | `LWD_PRE_OUT_PORT=5558 / LWD_POST_OUT_PORT=5559` 常量 + env 覆盖(替代 89 行的 pd_separation_config 依赖) |
| `EdgeCloudConfig` | **不迁三模式版**;ascend_config 新增仅 `prefill_only` 的最小配置段(mode 字符串 + enabled + 互斥校验),不引入 head_tail/embedding_only/PDSeparationConfig |
| patch/platform 注册 | 目标仓 `__init__.py` 增加 Lwd 挂钩模块导入(适配目标仓的条件注册风格) |
| worker.py / model_runner_v1.py | PO 增量按 hunk 摘迁并按标准改写(comm loop 的 PO 路由、EDGE_EMBED 执行、fill 点、虚拟 KV 内存) |

### 迁移方向(规避冲突,沿用评估结论)

- **vllm 主仓**:目标 = 源直接祖先 → 按文件 `git checkout 源 -- <L1 文件>` 后**人工剥离**非 PO 的 LWD/PD 增量(这 13 个文件的源版本含底座全家,只留 §3 清单所列增量);
- **vllm-ascend 仓**:以源分支为基底新开 `lwd-migration` 分支,把目标仓独有修复(kv_transfer/ascend_store 系列,仅 platform.py 重叠)rebase 上来,再按本方案做剥离与重写。

---

## 4. 阶段化执行(每阶段独立可验证、可暂停)

| 阶段 | 内容 | 出口判据 |
|---|---|---|
| **S0 设计文档门禁** | 本文档精简归档至目标仓 `docs/refactor/prefill_only_migration.md`(含例外表、迁移计划勾选框) | 文档评审通过 |
| **S1 L1 底座** | §3 清单逐文件落位;`edge_cloud_comm` 平移;世界布局层 scoping(先精确圈定 parallel_state 最小集再动手) | 目标仓原生测试全绿;PO 未接入时行为与迁移前逐字节一致 |
| **S2 L2 内核** | 按依赖序落文件:`lwd_message` → `lwd_*_channel` → `lwd_edge_dispatcher` → `lwd_cloud_phase_scheduler` → `lwd_cloud_engine` → `lwd_cloud_recv_manager`;逐文件跑 `check_functions.py` | 全部函数 ≤50 行/4 层;模块级单测(import 图断言:无 PD/passive 依赖) |
| **S3 L3 挂钩** | `lwd_edge_engine_hooks` / `lwd_cloud_launch`(精简版,仅 PO);worker/model_runner 的 PO 路由 hunk | 双进程拉起冒烟(edge+cloud);非 PO 路径行为不变 |
| **S4 自查回写** | check_functions.py 全量、ruff、mypy(如启用)、`tools/lwd_check_budget.py`(§7.4 交互预算)、目标仓既有 ut;文档回写 | 自查清单全绿(含交互预算专项);patch 清零点验(未新增任何 .patch) |

S2 内部建议的提交粒度:每文件一个 commit,commit message 引用设计文档章节,方便回溯与源仓 diff 对照。

---

## 5. 验收清单(合并标准自查 + 弱耦合专项)

**refactor-code 标准**:
- [ ] 新增/修改函数 ≤50 有效行、嵌套 ≤4(`check_functions.py` 全绿)
- [ ] Lwd 前缀判定表核对无专误用;同一概念词根唯一(transfer/notify/consume 各一个)
- [ ] 注释只留"为什么";50 行函数内行内注释 ≤3 条
- [ ] 无新增可预期 raise;error 日志只在最终处理点各一次
- [ ] 直改仓零 .patch;对上游文件的改动全部 additive 并在设计文档例外表登记

**弱耦合专项**:
- [ ] `grep -r "passive\|PDSeparated\|pd_separation" vllm_ascend/lwd/` = 0
- [ ] 模式判定实现数 = 1(`lwd_mode.is_lwd_prefill_only`)
- [ ] 主仓 additive 契约面与 §7-A 台账逐项一致(2 config 字段 + 2 结构 + 4 字段 + 1 枚举 + 3 钩子)
- [ ] 主仓在 `enable_edge_cloud=False` 时行为与迁移前 diff 为空(守卫验证)
- [ ] 目标仓 ascend 独有修复(kv_transfer 系列)在迁移后仍在

**交互预算专项(§7,CI 脚本强制;约束变体下按 §8.4 修订)**:
- [ ] 内核(lwd/ 除白名单文件)import 白名单违规 = 0
- [ ] 内核 getattr 防御访问 = 0(预算全部让给适配层,≤10 且逐处注释理由)
- [ ] 内核 env/os.environ/config 解析 = 0(唯一入口 `LwdConfig.from_env`,L3 调用)
- [ ] 云引擎对 EngineCore 的触达仅经 2 个端口协议(LwdCloudEnginePort 5 方法 + LwdCloudSchedulerView 3 方法)
- [ ] ~~edge patch 点 ≤ 6~~ →(仅改 vllm 仓时)monkey-patch = 0,in-tree 守卫分支 = 8 个固定点位,静态可 grep
- [ ] ~~`edge_cloud_comm` import 仅 2 文件~~ →(仅改 vllm 仓时)`torch.distributed` 使用仅 `lwd_edge_embed.py` + `lwd_cloud_chunks.py` 2 文件
- [ ] **`git -C vllm-ascend diff` = 0(硬门禁,CI 强制)**

---

## 6. 风险与决策点

1. **世界布局层是最大不确定项**(源仓 parallel_state 28 处符号、目标 0 处):S1 必须先 scoping 再动手——圈出 prefill_only 实际走到的路径(伪 PP 对、`is_cloud_device`、tensor meta),剥离 head_tail/embedding_only 分支。若发现与 PD 共享过深,降级方案是整块平移该层并在例外表登记(弱耦合让位于可行性)。
2. **语义保真的验证手段**:迁移前后用 `[PO-*]`→`Lwd` 日志做行为 diff(源仓联调日志作为 golden);关键不变量(notify 序==isend 序==recv 序、credit 公式、chunk-0 门)写成断言留在代码里。
3. **`current_wave`/coordinator**:目标仓有该字段;挂钩层返回 `(request, request.current_wave)` 依赖其存在,已在 L0 核实,S3 冒烟再验。
4. **后续与源仓的双向同步**:命名映射表(§2.7)是同步锚点;建议源仓 PO 侧冻结功能改动期间完成迁移,之后以目标仓为主。

---

## 7. 外部交互最小化:交互预算与端口-适配器

> 目标:内核(`vllm_ascend/lwd/`)与外部模块(vllm 主仓、edge_cloud_comm、ascend 配置/patch 框架)的**每一个**交互点都进台账、都有预算、都受 CI 检查。原则:**内核纯逻辑,外部触达全部收进 L3 适配层或 L1 additive hunk**。

### 7.0 现状交互审计(迁移基线,源仓实测)

| 交互形态 | 现状计数 | 位置示例 |
|---|---|---|
| 内核对外部模块的 import(vllm/vllm_ascend) | ~20 条 | dispatcher 直接 import `SchedulerOutput/BatchType/EdgeEmbedChunk`;active_engine_core import `Request`、`EMPTY_MODEL_RUNNER_OUTPUT`、executor、tracing、envs、`pd_separation_config` |
| `getattr(x, "attr", default)` 防御式访问(**隐性交互**) | **28 处** | active_engine_core 19、edge hooks 7、dispatcher 2 |
| 云引擎触达 EngineCore 的属性/方法 | **10 个** | scheduler、step_fn、add_request、abort_requests、post_step、request_block_hasher、vllm_config、model_executor、`_po_chunk_seqnos`(伪装成 engine 属性)、shutdown |
| edge 侧 monkey-patch 点 | 6 个 | preprocess_add_request、add_request、abort_requests、has_work、`__init__` 尾、shutdown |
| 内核内 env/config 解析 | 5 处 | dispatcher 读 credit env ×1、setup ×2、active ×2 |

### 7.1 主仓契约台账(A 类:数据/字段面,不可消除,只可精确)

| # | 契约项 | 落点 | 备注 |
|---|---|---|---|
| A1 | `enable_edge_cloud: bool` | config/parallel.py | 模式总开关,主仓分支只读它,不解析 additional_config |
| A2 | `is_edge_node: bool` | config/parallel.py | 角色 |
| A3 | `LwdChunkPlan` struct + `EngineCoreRequest.lwd_chunk_plan` 字段 | v1/engine/__init__.py | 结构必须住主仓(EngineCoreRequest 是它的宿主);默认 None |
| A4 | `EdgeEmbedChunk` struct | v1/core/sched/output.py | 同上(SchedulerOutput 字段的类型);仅 edge 进程内使用,永不上跨节点平面 |
| A5 | `SchedulerOutput.lwd_edge_chunks / lwd_edge_seqnos / lwd_edge_retain` 3 可选字段 | v1/core/sched/output.py | 默认 None,非 PO 路径不触 |
| A6 | `BatchType.LWD_EDGE_EMBED` 枚举值 | v1/core/sched/output.py | 不迁 PD 系列枚举 |
| A7 | 3 个空默认钩子(`_make_empty_batch_future` / `_before_execute_model` / `_check_batch_output_consistency`) | v1/engine/core.py | 见 §2.1,默认 no-op |

### 7.2 内核 import 白名单(B 类:CI 强制)

`vllm_ascend/lwd/` 内文件**只允许** import:

```
stdlib + msgspec + zmq
vllm.logger
vllm.v1.engine                # 仅 lwd_message.py / lwd_*_channel.py(线上类型)
vllm.v1.core.sched.output     # 仅 lwd_edge_executor_adapter.py(EDGE_EMBED batch 工厂)
vllm.v1.core.sched.async_scheduler + request_queue  # 仅 lwd_cloud_phase_scheduler.py(唯一继承点)
本目录(lwd.*)模块
```

**明确禁止**(现状有、迁移后归零):`pd_separation_config`、`edge_cloud_comm.*`、`ascend_config`、`vllm.v1.engine.core`(EngineCore/EngineCoreProc)、`vllm.v1.executor.*`、`vllm.v1.request`、`vllm.v1.outputs`、`vllm.envs`、tracing/system_utils。这些的使用点全部下沉到:
- `lwd_edge_engine_hooks.py` / `lwd_cloud_launch.py`(L3,本来就是贴 EngineCore 的胶水);
- `lwd_edge_executor_adapter.py`(新,见 7.3-C);
- `lwd_cloud_engine.py` 允许一个例外:进程入口 `lwd_cloud_main()`(原 run_active_engine_core 的装配壳)可以 import executor/tracing——但必须放在文件底部的独立装配区,与内核类(LwdCloudEngine)物理隔开,类本体零违禁 import。

### 7.3 交互收敛手段(C 类:结构性削减)

**C1. EngineCore 触达 10 个属性 → 2 个窄端口协议。** 内核类只依赖:

```python
class LwdCloudEnginePort(Protocol):      # 稳定公共 API,5 方法
    def lwd_step(self) -> tuple[dict | None, bool]: ...
    def lwd_add_request(self, request) -> None: ...
    def lwd_abort_requests(self, request_ids) -> None: ...
    def lwd_post_step(self, model_executed: bool) -> None: ...
    def lwd_shutdown(self) -> None: ...

class LwdCloudSchedulerView(Protocol):   # 只读快照,3 方法
    def lwd_unfinished_count(self) -> int: ...
    def lwd_waiting_count(self) -> int: ...
    def lwd_request_progress(self) -> Iterable[tuple[str, int, int]]: ...  # (rid, num_computed, num_prompt)
```

适配器(L3)包住 `engine_core` / `engine_core.scheduler` 实现。`request_block_hasher`、`Request.from_engine_core_request`、marker 挂载全部收进**一个**函数 `_lwd_cloud_admit_request(port, request, cfg)` —— 一次准入 = 一个交互点。
`_po_chunk_seqnos` 不再伪装成 engine 属性:seqno 注册表是 `LwdCloudEngine` 自己的状态,`_before_execute_model` 钩子实现直接读 self,`getattr(engine_core, "_po_chunk_seqnos", ...)` 这类访问(现状 5 处)全部消失。

**C2. EDGE_EMBED 运载收进一个工厂。** dispatcher 不再 import `SchedulerOutput/BatchType/EdgeEmbedChunk`;它产出内核自己的 `LwdChunk`(纯数据),经 `LwdEdgeExecutorPort.lwd_submit_chunk(chunk) -> LwdFuture` 提交。SchedulerOutput 构造、`make_empty()`、batch_type 赋值、`non_block=True` 语义,全部封在 `lwd_edge_executor_adapter.py` 一个函数里(W12 的消除,交互从"散布 4 行"变"集中 1 点")。

**C3. 配置解析一次成型。** `LwdConfig.from_env_and_config(vllm_config) -> LwdConfig`(dataclass:端口、credit、outstanding、chunk 对齐、debug 开关)——内核所有模块收 plain 值;内核 env/os.environ 读取 = 0,装配期断言替代 getattr 默认值。

**C4. edge patch 点预算 ≤ 6,含两个 removal spike。** S3 期间验证:
- spike-1:`preprocess_add_request` patch 是否可省(若原始实现对透传请求无副作用,则并入 add_request 单点拦截);
- spike-2:`has_work` patch 是否可省(PO 步进驱动自持忙循环时,EngineCoreProc 原逻辑是否已足够)。
结论(保留/移除)写入设计文档,最终 patch 点计数固定。

**C5. getattr 预算:内核 0,适配层 ≤10。** getattr 是"对上游版本边界的容忍",只允许出现在真正的边界文件(hooks/launch/adapter),每处注释容忍的版本差异。内核内 28 处 → 0。

**C6. edge_cloud_comm 封锁在 2 个文件。** `lwd_cloud_recv_manager.py` + worker 侧 `_lwd_edge_embed_one_chunk` 适配;其余内核文件 import 它即 CI fail。

**C7. hint MQ 经端口注入。** 云引擎不读 `model_executor.cloud_recv_hint_mq` 属性——装配期由 L3 取出并以 `enqueue_hint` 回调注入(`Callable`,与 dispatcher 的 publish 同款依赖注入风格)。

### 7.4 预算汇总与执行机制

| 交互类别 | 现状 | 迁移后预算 | 检查方式 |
|---|---|---|---|
| 主仓契约项(A) | 未受控 | **11 项台账**(7.1) | 文档台账 + 主仓 diff 审计 |
| 内核违禁 import | ~10 条 | **0** | CI grep 白名单 |
| 内核 getattr | 28 | **0**(适配层 ≤10) | CI grep + 注释审计 |
| 内核 env/config 解析 | 5 处 | **0**(`LwdConfig` 唯一入口) | CI grep `os.environ|environ\[` in lwd/ |
| EngineCore 属性触达 | 10 | **2 个端口协议(8 方法)** | 代码评审 + grep `engine_core\.` |
| monkey-patch 点 | 6 | **≤6(两个 spike 尽量减)** | 安装函数内集中列举 |
| edge_cloud_comm import | 3 文件 | **2 文件** | CI grep |

执行机制:白名单与预算写进 `lwd/__init__.py` 模块注释,配一个 `tools/lwd_check_budget.py`(grep 组合脚本,可挂 pre-commit/CI);预算变更必须先改台账(设计文档)再改代码——**台账是唯一事实源**。

### 7.5 交互设计的两条取舍(记录在案)

1. **继承 vs 组合(phase scheduler)**:`AsyncScheduler` 继承是内核与上游调度器最紧的一根线。备选是"只用准入策略、砍相位调度器"(交互归零但丢 prefill_first 优先级,行为不保真)。按"语义保真"原则保留继承,但它是内核**唯一**继承点,单文件隔离,若上游 AsyncScheduler 内部结构变化,爆炸半径锁定在这一个文件。
2. **结构住主仓 vs 弱类型减契约**:A3/A4 两个 struct 住在主仓才能让 SchedulerOutput/EngineCoreRequest 字段有类型(主仓不能 import ascend)。备选是字段类型放宽为 `Any`、struct 下沉内核(契约面 -2,丢类型安全与 mypy)。取前者:契约可数、类型完整,比"少两行主仓代码"更符合弱耦合本意。

---

## 8. 约束变体:仅改 vllm 仓(vllm-repo-only)

> 本节为硬约束下的**替代设计**,与前文冲突处以本节为准。约束:`prefill_only_base_br/vllm` 可改,`prefill_only_base_br/vllm-ascend` **diff 必须为 0**。

### 8.1 可行性结论(锚点已在目标仓核实)

| 能力 | 落点(全部 vllm 仓) | 核实锚点 |
|---|---|---|
| 控制面全套(ZMQ/dispatcher/云引擎/相位调度/准入) | `vllm/v1/lwd/` 新目录(纯 Python,无 NPU 依赖) | — |
| 云侧 fill(prompt_embeds 拼接) | gpu_model_runner 现有 `req_prompt_embeds` 分支(L1949)+ Lwd 守卫分支换数据源 | ✅ 目标仓已有该分支 |
| 边侧嵌入前向 | 通用 API:`model.get_input_embeddings()`(adapters.py 已提供)——不依赖 ascend 的 `embed_chunk_forward` | ✅ |
| 数据面传输 | `torch.distributed` isend/irecv,**tag=seqno** 匹配(hccl 后端支持 tag;edge rank0 ↔ cloud TP rank0 + 组内广播) | ✅ |
| EDGE_EMBED worker 分发 | `WorkerWrapperBase.execute_model`(worker_base.py:340)加守卫分支 → Lwd 处理器 | ✅ |
| 配置 | `additional_config` 本就是 vllm 仓字段:`vllm/v1/lwd/lwd_config.py` 解析 `edge_cloud_config` dict | ✅ |
| 进程入口 | serve.py `run_headless`(目标仓 L173)加守卫分支调 Lwd 云入口 | ✅ |
| 世界布局(边云伪 PP 对/角色) | parallel_state.py 本就是 vllm 仓文件 | ✅ |
| 边侧 KV 内存虚拟化 | gpu_worker `determine_num_available_blocks` 加守卫分支(原 ascend worker.py 的 1TiB hack 迁来) | vllm 仓文件 |

**结论:可行。代价集中在数据面(§8.3)——放弃 edge_cloud_comm 的预发 irecv 重叠,换 torch.distributed 直传。**

### 8.2 架构调整(相对 §1/§3 的增量)

- **模块树迁移**:`vllm_ascend/lwd/` → **`vllm/v1/lwd/`**(命名、分层、依赖方向不变;L2 内核反而更"名正"——它服务的 EngineCore/SchedulerOutput 本就住在这里)。
- **§3 的 ascend 侧整表作废**:edge_cloud_comm 不平移、pd_separation_config 不迁、EdgeCloudConfig 不进 ascend_config、patch/platform 不注册。
- **迁移方向简化**:ascend 仓的反向 rebase 整个取消(vllm-ascend 不动);vllm 仓仍按 §3 主仓路径(checkout 后人工剥离非 PO 增量)。
- **一个意外收益:patch 面真正归零**。原设计里 edge 侧 6 个 monkey-patch(patch_engine_core)与 serve 包装,在"可改 vllm 仓"前提下全部变成 **in-tree 守卫分支**(core.py 的 `add_request/abort_requests/…` 里加 `if lwd_active(): …` 分支)——比 monkey-patch 更符合标准 §5,交互点从"运行时替换"降级为"静态可见分支",可 grep、可评审。
- **装配点**:`EngineCore.__init__` 尾部加一个守卫调用 `lwd_edge_try_assemble(self)`(additive,非 PO 时立即返回)——取代原 patch 挂钩,同款单点开关。

### 8.3 数据面重设计(最大的实质变更)

原设计(源仓):edge_cloud_comm 通信服务(1608 行)+ hint MQ 快路径 + 云 recv manager 预发 irecv + 30s 等待门。
约束变体:**全部弃用**,改为最小直传——

```
边侧 LwdEdgeEmbedHandler(vllm/v1/lwd/lwd_edge_embed.py):
  token_ids → get_input_embeddings()(ids) → hidden
  → dist.isend(hidden, dst=cloud_first_rank, tag=LWD_WIRE_TAG_BASE + seqno)
云侧 LwdCloudChunkStore(vllm/v1/lwd/lwd_cloud_chunks.py):
  fill 点(gpu_model_runner 守卫分支)按需 dist.recv(tag=…)
  → 按 request 组织 chunk 缓存 → 拼接进 inputs_embeds → 消费即释放(retain 语义保留)
```

关键简化:**tag=seqno 让"通知序==isend 序==recv 序"的顺序契约(§6 风险 4/W20)整体消失**——torch.distributed 的 tag 匹配天然乱序安全,内核不再需要 seqno 注册表旁路(`edge_embed_seqnos` 字段与 A5 台账可减 1)。
云端 TP>1:cloud TP rank0 接收 + 组内广播(复用 vllm 既有 broadcast 原语);边侧 v1 限 TP=1(lwd_config 校验)。

**代价(设计文档显式记录,需用户确认接受)**:
1. 无 irecv 预发重叠:云 fill 在消费点同步 recv,首 chunk 延迟增加(预估每 chunk 一次 RTT 级);credit/水位背压机制不变,edge 不会跑在云消费前面太多,缓冲压力反而更小;
2. 无 30s 等待门:hccl 的 dist.recv 无超时参数。v1 语义 = 阻塞等待 + 云侧僵尸检测日志([PO-ADM] 同款)可观测、不自动恢复;后续可用 ZMQ 边带完成通知或看门狗线程补门(记入 backlog,不进本次);
3. 放弃 [PO-MEM] 的 NPU 私有统计(收敛为 chunk store 自身的 Python 侧计数)。

### 8.4 交互预算修订(§7 的约束变体版)

| §7 原条目 | 约束变体下 |
|---|---|
| C4 edge patch 点 ≤6 + 2 spike | **作废**:monkey-patch 归零,变为 core.py/serve.py/worker_base 的 **in-tree 守卫分支**(计数:core.py 装配 1 + 输入路径 4 + has_work 1 + serve 入口 1 + worker_base 分发 1 = 8 个守卫分支,全部静态可 grep) |
| C6 edge_cloud_comm 封锁 2 文件 | **作废**:edge_cloud_comm 不存在;torch.distributed 的使用封锁在 `lwd_edge_embed.py` + `lwd_cloud_chunks.py` 2 文件 |
| C7 hint MQ 依赖注入 | **作废**:无 hint MQ |
| A5 台账 3 字段 | 减为 2(`edge_embed_seqnos` 取消,tag 匹配取代;`edge_embed_retain` 保留——被抢占重 prefill 仍需云侧保留张量) |
| 内核 import 白名单 | 不变,但"外部"定义收窄为**上游 vllm 模块**(内核与上游同仓,白名单仍防内核蔓延);`vllm.distributed` 加入白名单(仅上述 2 个数据面文件) |
| 新增硬门禁 | **`git -C vllm-ascend diff` = 0**(CI 强制;这是本约束的验收底线) |

### 8.5 阶段修订

| 原阶段 | 变体下 |
|---|---|
| S1 L1 底座 | 仅 vllm 仓文件(§3 主仓表 + gpu_worker KV 分支);**无 ascend 工作** |
| S2 L2 内核 | 落 `vllm/v1/lwd/`;数据面 2 文件(8.3)按"边侧发送 → 云侧 chunk store"顺序加在通道/引擎之后 |
| S3 L3 挂钩 | patch 安装全部变为 in-tree 守卫分支落位(8 个);无 patch/platform 注册 |
| S4 自查 | 增加 `vllm-ascend diff=0` 校验;行为基线对比改为与源仓联调日志(控制面部分)+ 数据面新路径单测(tag 匹配、retain、消费释放) |

### 8.6 约束变体下的新增风险

1. **数据面保真度**:直传路径无真机联调历史(源仓的修复都发生在 edge_cloud_comm 路径上)。缓解:tag 匹配消掉了顺序类缺陷的一半面;credit/水位/准入等控制面语义逐字节保真;数据面单测 + 双机冒烟先行。
2. **dist.recv 阻塞**:见 8.3 代价 2,v1 显式接受并文档化。
3. **边侧加载完整模型**:无 ascend 定制时边侧载全模型只为 embedding(内存浪费,功能正确);优化(只载 embedding 层)记 backlog。
4. **`enable_prompt_embeds` 的 runner 行为差异**:目标仓 v1 的 prompt_embeds 路径未经源仓在 NPU 上的等价联调,fill 分支的 dtype/shape 约定(hidden_size、bf16)需在 S3 冒烟时用小模型先验证。

---

## 9. 新标准修订(2026-09-07,优先级高于前文冲突处)

> 五条新约束,已据此重立骨架 `vllm/v1/lwd/`;与前文冲突处以本节为准。

### 9.1 单向通信:边 -> 云,无结果面无水位

- 删除:POST_OUT 结果通道(CloudResultPublisher/EdgeResultReceiver)、
  消费水位信号(EdgeEmbedConsumed/LwdConsumedSignal)、快路径 set_fast_handler。
- 边侧请求生命周期全本地驱动:全段 ack 即终结(状态机去掉 FINISHED)。
- **背压语义变更**(推翻 §2.4 的动态水位背压):credit 退化为静态上限
  (per-request 未 ack 段 < credit_per_request,全局 < max_outstanding_segments)。
  云侧内存只能静态封顶 + lwd_stats()/zombie 日志观测,无自动回压。

### 9.2 重计算与 abort 必须显式建模

- retain:被抢占请求已收段保留在 LwdCloudEmbedStore,重 prefill 从缓存重放
  (gather_range 从头再切片即命中),不重新接收、不发回边侧;
- abort:LwdAbortSignal(边->云)-> 段仓整请求丢弃 + 登记清理;
  在途 recv 的 tag 无人认领即作废。边侧 abort 等末次 ack 后清理。
- SchedulerOutput 仅保留 1 个 additive 字段 lwd_edge_retain(§9.4 台账)。

### 9.3 SO 载具取消:边云通信物自带 seqno

- EDGE_EMBED SchedulerOutput 运载整体取消:dispatcher 产出 LwdEmbedSegment
  (request_id/segment_idx/offset/num_tokens/seqno/token_ids)经本地环 MQ 直达
  边 worker,worker 按序 isend(tag=LWD_WIRE_TAG_BASE+seqno)。
- worker_base 守卫分支按消息类型(LwdEmbedSegment)路由,不再看 BatchType。
- **主仓契约面收敛:11 项 -> 8 项**(§7.1 台账修订):
  A4 EdgeEmbedChunk 取消;A6 BatchType.LWD_EDGE_EMBED 取消;
  A5 减为 lwd_edge_retain 1 字段;其余(A1/A2/A3/A7)不变,A3 更名 LwdEmbedPlan。

### 9.4 命名:chunk 词根统一改 segment/embed

- LwdChunkPlan -> LwdEmbedPlan;LwdChunk/EdgeEmbedChunk -> LwdEmbedSegment;
  LwdChunkNotify -> LwdSegmentNotify;LwdEdgeChunkAck -> LwdEdgeSegmentAck;
  LwdCloudChunkStore -> LwdCloudEmbedStore;lwd_cloud_chunks.py -> lwd_cloud_embeds.py;
  配置 chunk_align/max_outstanding_chunks -> segment_size/max_outstanding_segments。
- 词根判定:数据单元=segment,张量内容=embed(s),通信预告=notify。

### 9.5 云侧 fill 对齐上游实现

- 上游填充锚点:gpu_model_runner.py:1949-1982 的 req_prompt_embeds 循环,
  仅依赖 .shape[0] 与切片访问 -> LwdCloudRemoteEmbeds 按同款鸭子类型实现,
  切片触达即 ensure/recv;上游 fill 代码零改动,只换数据源挂载点
  (_lwd_cloud_admit_request 把视图挂在请求 prompt_embeds 位置)。

### 9.6 被推翻/修订的前文章节索引

| 前文 | 修订 |
| --- | --- |
| §2.4 背压对称化 | 队满返 False 保留;水位背压删除(9.1) |
| §2.6 try_dispatch_embed 拆分 | chunk -> segment 命名(9.4),语义不变 |
| §2.7 命名映射表 | chunk 系条目按 9.4 更新 |
| §7.1 A4/A5/A6 | A4/A6 取消,A5 减 1 字段(9.3) |
| §8.3 数据面 | POST_OUT/水位/快路径删除;tag 直传保留(9.1) |
| §8.4 A5=2 字段 | 减为 1 字段 lwd_edge_retain(9.2/9.3) |

### 9.7 准入门取消(2026-09-07 迭代修订)

- 删除责任链三类(LwdEdgeAdmissionGate / LwdEdgeCreditGate /
  LwdEdgeOutstandingGate)、dispatcher._outstanding_segments、
  装配侧 _lwd_edge_read_credit,及配置字段 credit_per_request /
  max_outstanding_segments。
- 派发节奏回归天然节拍:每步至多派发一个 segment + worker ack 推进终结;
  9.1 的"静态上限封顶"表述随之作废,云侧内存观测仅剩 lwd_stats()/zombie 日志。
- 2026-09-07 版架构/时序图仍含"两道准入门"节点,属历史快照;图经确认后另行刷新。

### 9.8 step_wrapper 收敛:唯一扩展点是 step 接口(2026-09-07 迭代修订)

- **动机**:源仓的全部改动本质上只替换了 step_fn;因此不必用"3 个空默认钩子"
  (§2.1/A7)织入差异,直接把 step 变成可委托接口即可。
- **core.py additive 契约变更**(A7 三钩子作废):
  1. `EngineCore.__init__` 增加 `self.step_wrapper: LwdStepCore | None = None`;
  2. `step_with_batch_queue`(core.py:484)顶部加守卫:
     `if self.step_wrapper is not None: return self.step_wrapper.step_with_batch_queue()`;
     `step_fn` 接线(core.py:218)不动。
- **内核结构变更**:
  - 新增 `lwd_step_core.py`:`LwdStepCore(ABC)`,唯一接口 `step_with_batch_queue`
    (与上游同签名同返回);
  - `LwdCloudEngine(EngineCore)` 子类体系删除 -> `lwd_cloud_core.py` 的
    `LwdCloudCore(LwdStepCore)`:不继承 EngineCore,真实 EngineCore 实例照常装配,
    scheduler/executor 经 `LwdCloudEnginePort` 触达(端口方法集由 5 个 step 公共 API
    改为 step 编排触达面,S2 定稿);
  - 边侧步进(lwd_edge_process_step 及 _step_* 三函数)收编为
    `lwd_edge_core.py` 的 `LwdEdgeCore.step_with_batch_queue`;
  - 云侧 add/abort 只经 PRE_OUT drain 到达,公共接口全部删除;
    busy loop/has_work 回归原生 EngineCoreProc 驱动。
- **模式变化**:模板方法(3 钩子)-> 策略/包装器(单一委托);内核继承点 2 -> 1
  (仅剩 AsyncScheduler)。
- 2026-09-07 版架构/时序图仍标注"3 钩子覆写/模板方法",属历史快照;图经确认后刷新。

### 9.9 segment 概念删除:分块复用原生 schedule()(2026-09-07 迭代修订)

- **动机**:上游 Scheduler.schedule() 原生即做 chunked prefill 处理
  (num_scheduled_tokens / max_num_batched_tokens),边云不自造分割。
- **删除**:LwdEmbedSegment / LwdSegmentNotify / LwdEdgeSegmentAck /
  LwdEdgeRequestState 状态机 / LwdConfig.segment_size / 派发器自建分块簿记;
  **A3 契约项 LwdEmbedPlan 作废**(主仓契约面再 -1:struct 与
  EngineCoreRequest.lwd_chunk_plan 字段均不落位)。
- **新数据流**:
  - 边侧 step = 原生 schedule() 出分块决策 -> 按调度范围发 LwdEmbedNotify
    (request_id/offset/num_tokens/seqno,offset 取 num_computed)-> 原生 SO
    原样过本地环 MQ -> worker 按角色守卫路由做嵌入 -> isend(tag);
  - 请求进度由原生调度器自持(update_from_output 语义),回执只驱动进度更新;
  - 云侧 embeds 仓按 (request, offset 范围) 落位,fill 懒接收不变。
- **seqno 对齐**:dispatcher 单调分配且 notify 先于数据;worker 按
  (request, 已发次数) 本地推演同一 seqno,云侧按 notify 登记。
- 词汇表更新:数据单元=调度范围(range),张量内容=embeds;segment 一词全仓移除。
- 2026-09-07 版架构/时序图为 segment 时代快照,经确认后刷新。
