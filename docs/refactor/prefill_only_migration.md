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

### 9.10 边侧调度器补位:LwdEdgeScheduler(2026-09-07 迭代修订)

- **问题**:9.9 的"边侧复用原生 AsyncScheduler"只覆盖分块这一半;原生调度器
  的两个前提在边侧不成立——prompt 算完会转 decode 调度、请求只能由模型输出
  终结(边侧无模型执行且只发不收)。
- **新增** `lwd_edge_scheduler.py`:`LwdEdgeScheduler(AsyncScheduler)`
  - `schedule()` = 纯 prefill 调度(分块复用原生),不进 decode;
  - `lwd_edge_update_progress(acks)` = 回执推进 num_computed,
    prompt 全部嵌入完成即本地终结(不依赖任何模型输出,与原生
    update_from_output 的唯一语义差)。
- **接线**:lwd_edge_try_assemble 注入 scheduler_cls=LwdEdgeScheduler,
  与云侧(scheduler_cls=相位调度器)同款方式。
- 继承例外恢复为 2 个(两个调度器文件);纯 prefill 原语与云侧
  _lwd_schedule_pure_prefill 同源,S2 实现时评估下沉公共基类。

### 9.11 空壳类清理:删执行器适配器,embed 降为模块函数(2026-09-07 迭代修订)

- **删除**:`lwd_edge_executor_adapter.py` 整文件、`LwdEdgeExecutorPort`、
  `LwdFuture`——§9.9 后该适配层已零转换(原生 SO 原样过 MQ),
  `LwdEnginePort.lwd_execute_model` 完全覆盖,独立端口成空壳。
- **类降函数**:`LwdEdgeEmbedHandler` 类删除,`lwd_edge_embed.py` 改为模块函数
  `lwd_edge_execute_embeds(model, scheduler_output)`(+两个私有子函数),
  worker_base 按角色守卫直接调用;torch.distributed 白名单文件不变。
- `LwdEnginePort` 增补 `lwd_drain_embed_acks()`(边侧回执取回)。
- 端口-适配器模式保留于:LwdEnginePort/LwdEnginePortAdapter 与调度器视图;
  前端 executor 侧不再有 Lwd 适配层。

### 9.12 范围裁剪:只保留控制面(2026-09-07 迭代修订)

- **删除数据面文件**:`lwd_edge_embed.py`(嵌入前向 + isend)与
  `lwd_cloud_embeds.py`(按需 recv/段仓/retain/消费释放/fill 视图)整体移出;
  数据面另行落位,经既有接缝对接:
  - 边侧:`LwdEdgeScheduler.lwd_edge_update_progress`(执行量来源);
  - 云侧:`LwdCloudCore._lwd_handle_embed_notify`(接收登记)与
    `_lwd_cloud_admit_request`(请求侧挂载点)。
- **删除 dispatcher**:`LwdEdgeDispatcher` 职责完全并入
  `LwdEdgeScheduler`(notify 发布 / abort / seqno 单调分配),
  publisher 经装配期 partial 注入调度器;边侧控制面出口唯一化。
- 连带清理:`LwdEdgeEmbedAck`/`LwdEnginePort.lwd_drain_embed_acks`、
  `LwdConfig.debug_wire`、`LwdLog.wire`、`LWD_WIRE_TAG_BASE`、
  云侧 core 的 retain/消费释放私有段(embeds 仓依赖);
  `tools/lwd_check_budget.py` 的 torch.distributed 文件数检查移除
  (本目录零数据面,该检查移交数据面落位侧)。
- 目录现状:16 个文件,全部为控制面(调度器×2 + 准入策略 + step 载体×2 +
  通道×2 + 消息 + 装配×2 + 支撑×4 + 台账)。
- v3 架构图含 worker/embedstore 节点,属数据面时代快照,经确认后刷新。

---

## 10. 范围修订:两仓控制面整体收编(2026-09-08,优先级高于前文冲突处)

> 用户裁定:lwd_control 目录(control_scheduler ×8 + control_communication ×2)
> 是**已定稿的完整框架**,不再新增/调整结构;本次迁移只改 vllm 仓,
> 之前 vllm 与 vllm-ascend 两仓改动中**凡是控制面与控制面通信相关的,
> 全部收编进本目录**。vllm-ascend 仓 diff = 0 硬门禁不变(§8)。

### 10.1 收编口径

- **进目录**:prefill_only 控制面逻辑与控制面通信(调度决策、准入、
  请求生命周期、ZMQ 控制通道、装配入口)——无论原先住在 vllm 还是
  vllm-ascend(含原 vllm 仓 core.py 的 step 改动,§9.8 已溶入 step_wrapper)。
- **不进目录、留在上游文件**:非控制面逻辑最小接线守卫(core.py 的
  step_wrapper 字段/守卫/装配点/shutdown 守卫、serve.py:173 的云入口守卫),
  全部 additive,非 PO 路径逐字不变。
- **不迁(另行落位/延后)**:
  - PD 分离专属文件(passive_scheduler / pd_separated_scheduler / passive_core /
    patch_pd_scheduler_shim / scheduler_conflicts / pd_separation_config 主体 /
    edge_cloud_comm 8 文件):属 PD 特性,被 Lwd 设计取代,框架无对应槽位;
    PO 控制面经 §9.1/§9.12 已零依赖,只吸收其端口布局约定(5558/5559 +
    dp_rank×2)与线程_owned_socket 线规。
  - 数据面(prefill_only_recv_manager / worker / model_runner hunk、
    vllm 仓 gpu_model_runner/worker_base 等):§9.12 延后,经既有接缝对接。
  - 运行时底座(parallel_state 角色 / executor MQ 拓扑 / config 字段):
    延后;控制面阶段的模式判定与配置解析自持于 lwd_control 内,不依赖上游字段。

### 10.2 源 → 槽位映射(收编总表)

| 源(vllm-ascend) | 目标槽位 | 语义转换 |
| --- | --- | --- |
| edge_cloud/prefill_only_channel.py(467 行) | control_communication/lwd_control_publisher + lwd_control_subscriber(方向原语,side-agnostic) | 只留 PRE_OUT 单向;POST_OUT/结果面/水位/fast-handler 删除(§9.1);publish 队满返 False(§2.4);线上消息定义收进通信层 |
| edge_cloud/edge_prefill_dispatcher.py(576 行) | control_scheduler/lwd_edge_scheduler | chunk 状态机/round-robin/两道背压门取消(§9.7/§9.9);分块复用原生 schedule();notify/abort/seqno 并入调度器(§9.12) |
| edge_cloud/prefill_only_engine_core.py(280 行) | control_scheduler/lwd_edge_core + lwd_edge_assemble | step 四拍改编排;6 个 monkey-patch → core.py in-tree 守卫 + 装配单点开关 |
| edge_cloud/active_engine_core.py(1074 行) | control_scheduler/lwd_cloud_core + lwd_cloud_assemble | ~200 行 step 拷贝经 §9.8 step_wrapper 消解;准入策略族后经 §10.10 收编进相位调度器(原 lwd_cloud_admission.py 删除);进程入口/调度器视图/唯一准入交互点收进装配文件 |
| edge_cloud/pure_phase_scheduler.py(166 行) | control_scheduler/lwd_cloud_phase_scheduler | 容器交换 + 空步翻转机制保真;prefill_first/decode_first + 工厂 |
| patch_engine_core.py PO hunk | lwd_edge_assemble(+core.py 守卫) | 挂钩逻辑内聚装配文件 |
| patch_serve_headless.py PO hunk | lwd_cloud_assemble(+serve.py 守卫) | 同上 |
| pd_separation_config.py 端口段 | control_communication 常量 + env 覆盖 | 替代整文件依赖 |

| 源(vllm 仓) | 处置 | 依据 |
| --- | --- | --- |
| v1/engine/core.py step 改动 | 溶入 lwd_step_core + LwdCloudCore;上游只留 4 处守卫 | §9.8 唯一扩展点 |
| v1/engine/__init__.py(ChunkPlan + 字段) | 不落位,EngineCoreRequest 原样使用 | §9.9 原生分块 |
| v1/core/sched/output.py(SO 扩展) | 不落位;retain 随数据面延后 | §9.3/§9.12 |
| utils/__init__.py 判定函数 | 收敛为 lwd_control 内唯一判定(装配函数自判) | W9/§2.3,主仓不 import 判定 |
| parallel_state / config / executor / worker_base / gpu_model_runner 等 hunk | 延后(底座/数据面) | §9.12 |

### 10.3 支撑内容就近落位(框架不增文件)

lwd_message/lwd_config/lwd_ports 的内容折入既有文件,S2 落位时定稿:
- 线上消息(LwdRangeNotify/LwdRequestNotify/LwdAbortNotify,定名见 §10.7)→ control_communication;
- LwdConfig(模式判定 + env/additional_config 解析,唯一入口)→ 装配文件,内核收 plain 值;
- LwdEnginePort 协议 → lwd_step_core(与 step 抽象同文件),适配器在两侧 assemble。

**传输层去边化(2026-09-08 追加,修订 §2.7 命名判定)**:通信类只认
方向原语——LwdControlCommunicator(基类)/ LwdControlPublisher
(OUTBOUND)/ LwdControlSubscriber(INBOUND),不认边/云;侧别与
bind/connect 降为装配期 wiring(lwd_edge_assemble:connect;
lwd_cloud_assemble:bind)。演进代价:加平面 = lwd 消息加结构体 +
两侧 assemble 各一行实例化,传输层零改动;原 lwd_edge_channel /
lwd_cloud_channel 按侧别命名的文件随之移除。

### 10.4 阶段计划(每阶段独立提交、可回滚)

| 阶段 | 内容 | 出口判据 |
| --- | --- | --- |
| S0 落位准备 | 修复悬空 import(`vllm.v1.lwd.*` → `vllm.v1.lwd_control.*`);支撑内容归属定稿;lwd_check_budget 路径对齐 | 骨架 import 图干净、预算脚本可运行 |
| S1 通信层 | 两个通道实现(有界队列 1000、线程持 socket、publish 返 False、drain 非阻塞、shutdown join 2s) | 通道单测(队满/排空/关停) |
| S2 边侧控制面 | lwd_edge_scheduler(纯 prefill + notify/abort/seqno + 本地终结)+ lwd_edge_core + lwd_edge_assemble | 边侧单测(notify 先行序、本地终结语义) |
| S3 云侧控制面 | 相位调度器 + 准入策略 + lwd_cloud_core(drain→准入→步进→记账)+ lwd_cloud_assemble | 云侧单测(空步翻转、三路径清理唯一实现、首预告门) |
| S4 接线 | core.py 4 处守卫 + serve.py 入口守卫(additive) | 非 PO 路径行为逐字不变;守卫单测 |
| S5 验收 | 预算检查(import 白名单/getattr/env)、≤50 行、vllm-ascend diff=0、映射表与例外表回写 | 验收清单全绿 |

### 10.5 行为不变量(控制面,迁移必保)

1. notify 先于数据(seqno 登记序 = 派发序);
2. 元数据不丢:订阅端阻塞等待,发布端队满返 False 不丢已发消息;
3. 首预告门:请求收到首个 notify 前不可调度(防 fill 空等超时杀请求);
4. `_lwd_release_request` 唯一实现,abort/finished/finish_reason 三路径共调;
5. 空步一次性翻转,不重复调 schedule()(不虚增 step 计数/KV 记账);
6. 非 PO 路径零行为变化(step_wrapper 恒 None,守卫短路)。

### 10.6 落地状态(2026-09-08 回写)

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| S0 落位准备 | ✅ | 悬空 import 全部修复(30 处包内 import 解析验证);支撑归属按 10.3 定稿;lwd_check_budget 可运行 |
| S1 通信层 | ✅ | 公共基类 LwdControlCommunicator(线程持 socket/幂等关停/join 2s)+ 线上消息与编解码折入其中;边 publish 队满返 False(默认 1000);云 drain 非阻塞,坏帧同时捕获 DecodeError/ValidationError(msgspec 平行异常类) |
| S2 边侧控制面 | ✅ | 纯 prefill = 原生 schedule() 全复用(边侧无输出 token,完结当步由 lwd_edge_update_progress 本地终结,decode 分支不可达);notify offset 取调度后回退量;未派发重试按 update_from_output 的拒绝回退同款语义回滚 num_computed;add 预告短退避重试超限告警放行(zombie 兜底) |
| S3 云侧控制面 | ✅(准入后经 §10.10 收编进调度器) | 相位调度 = 队列手术复用(decode-ready 暂存/等待队列整体暂存),空步检查先行、单次 super().schedule()(不变量 5);准入策略族 + 工厂;首预告门在 `_lwd_apply_scheduling_policy` 落地(不变量 3);release 三路径唯一实现(不变量 4);zombie 观测 |
| S4 接线 | ✅ | core.py 4 守卫 + serve.py 入口守卫,纯增量 22 行(0 删除);装配分流 lwd_try_assemble/lwd_shutdown/lwd_serve_guard 住 lwd_control/__init__;云装配与 __init__ 尾守卫幂等共存 |
| S5 验收 | ✅(静态) | ruff check/format 全绿;check_functions 全绿(≤50 行/4 层);lwd_check_budget 全绿;py_compile 全过;线上编解码冒烟通过(往返/tag 联合/默认值/坏帧);vllm-ascend diff = 0 |

10.3 折入的落位结果(框架不增文件,根目录仅 __init__ 台账):

- 线上消息 + 编解码 → `lwd_message.py`(原折入 communicator,2026-09-08
  拆出;TYPE_CHECKING re-export EngineCoreOutputs 例外随迁);
- LwdConfig + is_lwd_prefill_only(唯一实现)→ `lwd_edge_assemble.py`
  承载,云装配 import 复用;env 唯一入口随之落在该文件(预算脚本已对齐);
- LwdEnginePort/LwdCloudSchedulerView 协议 + LwdStepSettings(plain 值
  载体)+ LwdLog → `lwd_step_core.py`;边/云端口适配器分别落
  `lwd_edge_assemble.py` / `lwd_cloud_assemble.py`(云侧含 wrapper
  翻转防递归,单线程 step 循环下安全);
- 云侧 lwd_cloud_main(args, engine_core) 保留 serve 入口形态,与
  __init__ 尾守卫经幂等检查共存。

例外与 backlog(台账登记):

- `typing.Union` 而非 `X | Y` 定义线上联合(msgspec 解码器全版本路径,noqa UP007);
- 调度器两文件 import RequestStatus(AsyncScheduler 继承面既有传递依赖);
  §10.10 后 lwd_cloud_phase_scheduler 增 TYPE_CHECKING Request 注解
  (staging 池签名);§10.11 后 RequestStatus 升为运行时 import(abort
  终结状态值)+ TYPE_CHECKING LwdRequestNotify(门池注解,运行时零
  触达);
- 调度器装配为 __init__ 尾整实例替换(重建一次前缀缓存管理器);
  kv_connector 在位时降级原生(重建丢握手态);入口期 scheduler_cls
  注入可免重建 —— backlog,随上游接线/数据面落位处理;
- 本环境无 torch/msgspec 完整运行栈:仅静态套件 + AST import 图 +
  线上编解码冒烟;S1-S3 的通道/边侧/云侧运行时单测与双进程冒烟
  待可运行环境补(出口判据见 10.4);
- 数据面与运行时底座按 10.1 延后,接缝位置:边侧
  lwd_edge_update_progress(executed 来源)、云侧 _lwd_handle_range_notify
  与 _lwd_cloud_build_admit_request(prompt_embeds 挂载)。

### 10.7 线上消息命名定稿(2026-09-08)

- 三个 PRE_OUT 消息类统一 Notify 词根(§9.4 词根法"通信预告=notify"),
  预告对象一律用名词:**LwdRequestNotify**(请求预告,原 LwdAddRequest)/
  **LwdRangeNotify**(调度范围预告,原 LwdEmbedNotify,偏离 §9.9 明文名)/
  **LwdAbortNotify**(abort 预告,原 LwdAbortSignal)。
- 动机:LwdAddRequest 的 "Request" 头部易误读为请求对象本身;
  LwdEmbedNotify 的 "Embed" 是操作当名词,且与数据面张量内容词根冲突
  (§9.9 词汇法:数据单元=调度范围 range,张量内容=embeds)——Range
  即取自该词汇法;Message 后缀方案否决——与文件名/union 名三重冗余。
- 云侧 handler 同构:_lwd_handle_request_notify / _lwd_handle_range_notify /
  _lwd_handle_abort_notify。
- §2.7 映射表与 §9.9 的 LwdEmbedNotify 等为历史快照,以本节为准;
  LwdRequestNotify 的字段取舍(3 字段瘦身)为唯一无前文出处的实现决策,
  max_tokens=16 为占位,采样参数 additive 待数据面补齐。

### 10.8 云侧调度初版对齐(2026-09-08)

- 目录二次分组:control_scheduler 拆为 control_edge_scheduler /
  control_cloud_scheduler 两包(原 control_scheduler/ 仅留包说明);
  重组后 18 处包内 import 修复,预算脚本 vllm.v1.engine 例外键
  随 §10.7 拆分改指 lwd_message.py。
- **空步翻转恢复**(源 _force_other_phase 对齐,gap-1):相位调度器
  schedule() 选相后若出 0-token 空步且另一相有活,置一次性翻转标志,
  下一步换相;空步照返不重复调 schedule()(不变量 5)。
- **SeparatePhases 源语义恢复**(gap-2):准入条件由"decode 活跃即挡"
  改回源定义 —— 调度器完全排空(unfinished == 0)才整批放行;放行
  截断到 max_num_seqs(经 lwd_cloud_admission_policy 工厂注入,值取
  scheduler_config;<=0 不截断),溢出留待下一轮排空相。宽松版不保留。
- **batch_queue fail-fast**(gap-3 前置):lwd_cloud_try_assemble 在云角色
  装配前断言 engine_core.batch_queue 非空 —— 无 batch queue 时 step_fn
  绑定同步 step,step_wrapper 守卫(core.py:509)永不触发,drain 静默
  失效;部署错误在装配期显式崩溃(源装配断言同款)。
- 0-token 空批预完成 future(源增强 1)仍未落:原生 step(core.py:491)
  对空批照常派发 worker,fork 血统 runner 回 None 即 core.py:576
  "unexpected error";触发组合 immediate + KV 压力。处置选项(in-tree
  守卫 / 限制 immediate 用途 / 真机复现后定)待拍板,见 §10.5-6 之外
  新增挂账。
- **门序修正**(等价性审计发现):初版先策略截断再过首预告门,队首
  未预告请求占满截断名额即卡死准入;改为源同序 —— 先过滤 eligible
  (pending ∩ 已预告)再交策略,截断只数已预告。
- **has_work 接线洞(挂账,待拍板)**:原生驱动 has_work(core.py:1228)
  = engines_running ∨ scheduler.has_requests() ∨ batch_queue,不含 Lwd
  pending;首请求只经 PRE_OUT 到达时循环停在 input_queue 阻塞等,
  step_wrapper 永不执行 → 云端聋。源以其自有 busy loop(_has_work 含
  pending)规避;§9.8"busy loop/has_work 回归原生"在该到达模型下不
  成立。推荐修法:in-tree 守卫 #6 —— has_work 追加 step_wrapper 非空时
  的 lwd_has_work()(LwdStepCore 增接口,LwdCloudCore 实现为
  pending 非空 ∨ 订阅通道有积压),§10.1 守卫计数 5 → 6。
- **源步接口移植稿(裁剪定稿)**(2026-09-08):源 _native_step_bq_prefill_only
  → lwd_native_step_bq_prefill_only,**并入 lwd_cloud_core.py 文件尾**
  (框架不增文件;engine_core 显式入参免 MethodType)。
  步内保留:空批预完成 future(增强 1,§10.8 挂账项的本体落位)+
  [PO-RPC] 预检;增强 2(seqnos)/ 增强 3(retain)经确认不需要已剪除
  —— 两者均为数据面载体,落位时经 SchedulerOutput 扩展另行对接;
  POST_OUT/水位按 §9.1 裁剪。步外结论:源 _process_engine_step 相对
  原生(core.py:1267)仅余僵尸日志诊断,水位/POST_OUT/step_index 均可
  裁 —— **步外直接用原生**(当前接线即如此,Lwd 记账全在 step_wrapper
  内);lwd_cloud_process_engine_step 已删。scheduler 模块差异已盘点:
  源 4 文件 426 行(output.py PO 字段族 293/request.py np 缓存 37/
  scheduler.py 热路径 107/async_scheduler PD guard 21),控制面均不依赖;
  相位调度器依赖的 is_prefill_chunk 与 skipped_waiting 基线已有,目标仓
  齐备。函数超 50 行为移植暂态;预算例外 v1.outputs(lwd_cloud_core.py)。
- **LwdCloudCore 结构镜像源 ActiveEdgeCloudEngineCore**(2026-09-08 定稿):
  `step_with_batch_queue` = 入口单拍(drain → 准入 → `_process_engine_step`),
  `_process_engine_step` = 源步外接口替换稿(增强步体 + finished 清理;
  POST_OUT/水位 §9.1 裁剪,post_step/GIL 让出由外层原生步外承担不重复,
  返回元组而非源的 bool 以嵌套 step_fn 委托链),增强步体 =
  `lwd_native_step_bq_prefill_only`。绑定随之定稿:adapter 增
  `lwd_engine_core()` 访问器,步体不再回调原生步体 → wrapper 翻转/
  `lwd_bind_wrapper`/端口 `lwd_step_with_batch_queue` 整体移除(协议
  LwdEnginePort 同步修订);has_work 守卫 #6 挂账仍独立存在。
- **LwdCloudCore 全方法照搬**(2026-09-08 二次迭代,取代上一条 curated
  结构):源 ActiveEdgeCloudEngineCore 全部 14 方法
  (__init__/_try_fast_forward/_record_chunk_notify/_drain_control_plane/
  _handle_add_request/_handle_abort/_forward_chunk_hint/_forward_drop/
  _apply_scheduling_policy/_admit/_publish_consumed_watermarks/_has_work/
  _process_engine_step/run_busy_loop)照搬进 LwdCloudCore,方法名保持源
  名;另有守卫入口 step_with_batch_queue(drain→准入→步外)与 lwd_stats。
  12 条照搬差异清单住文件 docstring(消息名映射/chunk_idx→offset==0 门/
  step_fn 绑定不搬防双步进/步体直调/results 可空注入/水位传输点日志化/
  fast path 不接/_admit 委托闭包/post_step 与 sleep 不搬防双份/state 两
  字段版/run_busy_loop 不启用/超 50 行暂态)。旧 curated 方法
  (_lwd_*族/_notified/_embed_registry)删除;§10.5-4"三路径唯一实现"
  暂被源双处清理形态取代,由后续收敛。预算:getattr 容忍文件集增
  lwd_cloud_core.py(源防御式访问暂存,收敛时清零);v1.outputs 例外
  不变。has_work 守卫 #6 挂账不变。
- **相位调度器同步照搬**(2026-09-08):lwd_cloud_phase_scheduler.py 整
  文件替换为源 pure_phase_scheduler.py 照搬版(类名映射
  PurePhaseSchedulerBase/PrefillFirst*/DecodeFirst* → LwdCloudPhaseScheduler/
  LwdCloudPrefillFirstScheduler/LwdCloudDecodeFirstScheduler;工厂保留源名
  get_pure_phase_scheduler_cls,装配层同步改引)。curated 的
  is_prefill_chunk 工作纯手术版删除 —— 相位语义基线定为源的"按人口分伙"
  (prefill 步只看 WAITING,decode 步只看 RUNNING,已开动请求的 prefill
  尾巴在 decode 步续算);native_mix 语义归装配层裁决(不装 = 原生混合,
  当前装配恒装,配置传 native_mix 会告警回退 prefill_first)。
- **scheduler 构造期注入定稿**(2026-09-08):云侧相位调度器改由
  serve 守卫注入 —— `lwd_serve_guard` 在 vllm_config 建成后、引擎构造前
  写 `scheduler_config.scheduler_cls = <LwdCloudPhaseScheduler 全限定名>`
  (字符串形式,跨进程序列化安全;上游一等配置,EngineCore.__init__:139
  get_scheduler_cls 构造期解析)→ 引擎出生即相位调度器。整实例替换
  (`_lwd_cloud_install_scheduler`)删除:前缀缓存管理器重建与
  kv_connector 降级两条款随之消失;装配层少一次调度器重建。边角色不
  注入(LwdEdgeScheduler 需 publisher 构造注入,仍走 __init__ 尾
  instance swap)。scheduler_name 选类能力保留:守卫经源工厂
  get_pure_phase_scheduler_cls(config.scheduler_name) 取类对象注入
  (类按模块引用序列化),prefill_first/decode_first 照常可选。

### 10.9 通信层组合化 + 词根统一 notify(2026-09-08)

组合化重构(问题量化 P1-P5 与行为不变量见
docs/refactor/lwd_control_communication_composition.md):

- `LwdControlCommunicator` 由模板方法基类(线程持 socket/幂等关停)
  瘦身为纯收发句柄 send/recv/close/terminate(无线程,单线程亲和);
  publisher/subscriber 脱离继承,自有线程 + 成员组合。公开 API 不变,
  消费面零改动。§10.6 S1 行的"公共基类"描述由本节取代。
- **P5 关停缺陷修正(冒烟实测发现的存量缺陷)**:跨线程 close(0)
  不唤醒阻塞 recv(macOS/pyzmq 实测),旧"close 即线程唯一退出路径"
  从未生效——关停 join 必超时、泄漏阻塞线程、context 永不 term。
  修正:term 是跨线程打断阻塞收发的可靠手段(ETERM);subscriber
  close→term→join(2s),publisher 哨兵入队→join(2s)→卡死 term 兜底。

词根统一(§10.7 Notify 词根的收尾,"wire"与"message"退场):

- LwdWireMessage → **LwdNotify**(三 Notify 类型的 Union;"wire"本无
  对应物——线上表示只是 bytes,该类型是解码后的类型联合);
- lwd_encode_wire/lwd_decode_wire → lwd_encode_notify/lwd_decode_notify,
  _WIRE_DECODER → _NOTIFY_DECODER;
- lwd_message.py → **lwd_notify.py**(git mv,历史保留);§10.3 落位结果
  与 §10.8 预算例外键中的 lwd_message.py 改指 lwd_notify.py(脚本已同步);
- 冒烟(ipc 端到端):203 条三类型消息 FIFO 保序、drain 取空、双侧关停
  毫秒级且幂等、无线程泄漏;改名后 ruff/check_functions/lwd_check_budget
  复跑全绿。

### 10.10 云侧准入收编 scheduler_cls(2026-09-08)

> 依据:prefill_only_core_reuse_scheduler_only_plan §2.3/§4(core 全复用、
> 调度逻辑唯一归宿 = scheduler_cls);本次只落调度面,step/驱动/传输不动。

- **动机**:EngineCore 侧(源 ActiveEdgeCloudEngineCore → LwdCloudCore)
  持有准入决策(`_apply_scheduling_policy` + SeparatePhases/Immediate
  策略族),与"core 零自有调度逻辑"的收敛方向冲突;批次纪律是纯调度
  知识,归宿是 scheduler_cls。
- **收编形态**(`lwd_cloud_phase_scheduler.py`):
  - `LwdCloudPhaseScheduler` 基类增 staging 池 `_staged`(rid -> Request,
    FIFO;源 pending 池的调度器侧后半段,不计入 unfinished —— 否则释放
    条件永假);
  - `add_request` override:separate_phases(默认)进池;immediate 直通
    super(原生等价);
  - `_lwd_release_staged()` 释放闸挂 `schedule()` 顶部、先于选相:未满
    unfinished == 0 不放行;放行截断 max_num_seqs(<=0 不截断),溢出
    留待下一轮排空相;放行走原生 add_request 全路径(簿记/connector/
    统计事件不缺);语义逐条对齐源 SeparatePhasesPolicy(gap-2 口径);
  - `has_requests()` override 含暂存池(排空窗口到达时 schedule 可被
    驱动,防 staged-only 死等);`finish_requests()` override 对池内
    请求就地摘除(abort 到达暂存态的清理路径;未入原生簿记,无 KV/
    队列需释放),按返回契约上抛 (rid, client_index);
  - 两纪律折叠为类属性 `LWD_CLOUD_IMMEDIATE_ADMISSION`(部署决策 =
    类身份,经 scheduler_cls 注入,跨进程按模块引用序列化安全,不引
    partial/动态类);注册表折叠源 BATCH_POLICY_REGISTRY 与准入策略族
    两张表为 (phase, admission) 二维表,四个具体类
    (LwdCloud{PrefillFirst,DecodeFirst}{,Immediate}Scheduler),工厂
    `get_pure_phase_scheduler_cls(name, admission_name)` warn-and-fallback
    口径不变。
- **首预告门不进调度器**:chunk-0 就绪判定依赖 PRE_OUT notify(传输层
  知识),留在 LwdCloudCore 控制面;`_apply_scheduling_policy` 收缩为
  `_admit_pending`(门内即交调度器,纪律自持);调度器只见"已可跑的
  请求"。(**已被 §10.11 取代**:门状态与通知处理接口后经用户裁定收编
  进调度器。)
- **删除**:`lwd_cloud_admission.py` 整文件(策略族 + 工厂 + admission
  state);LwdCloudCore 构造的 admission_policy 注入线;装配层
  `lwd_cloud_admission_policy` 接线(max_num_seqs 由调度器自取
  scheduler_config,装配不再传值)。
- **接线**:serve 守卫 `lwd_serve_guard` 改传 (config.scheduler_name,
  config.admission_name) 二维选类;`edge_cloud_config.admission` 配置键
  与默认 separate_phases 语义不变。
- **顺手收敛**:lwd_cloud_core 照搬暂态的 ruff 违例(UP037 引号注解 ×5、
  F821 幽灵名 CloudResultPublisher → `object | None`,§9.1 裁发布器仅
  判空)与一处 format 差异;85 行步体函数维持 §10.8 豁免不动。
- **验证**:ruff check/format、lwd_check_budget、py_compile、
  check_functions(新改函数)全绿;stub 父类行为冒烟覆盖二维工厂解析/
  暂存/截断与不截断/排空窗口/未排空不放行/abort 摘除/immediate 直通/
  未知名回退;运行时单测随 §10.6 backlog 待可运行环境补。
- **未决承接**:源步体缺失/双步进暂态、has_work 守卫 #6 挂账、空批
  预检等 step/驱动面项不属本节,按 prefill_only_core_reuse 方案另批
  落地;数据面挂载点(`_lwd_cloud_build_admit_request`)不变。

### 10.11 控制面功能接口二次收编进调度器(2026-09-08,用户裁定)

> 用户裁定:lwd_cloud_core.py 中与 EngineCore 模块强耦合的逻辑留原文件,
> 功能类接口全部整合进 lwd_cloud_phase_scheduler.py。**偏离方案 §2.3
> "chunk-0 门控不进 scheduler、留桥线程 A"的备注,记录在案**;调度器
> 不触达传输对象与 EngineCore 的底线保持。

- **移入调度器**(`lwd_cloud_phase_scheduler.py`):
  - 首预告门状态:`_lwd_gate_pending`(rid -> LwdRequestNotify,未过门
    线上元数据)+ `_lwd_gate_ready`(已收首预告 rid);LwdRequestNotify
    仅 TYPE_CHECKING 注解,运行时不触达传输对象;
  - 三类通知处理接口:`lwd_cloud_on_request_notify`(门池暂收 + 乱序
    防御)/ `lwd_cloud_on_range_notify`(过门即建请求进暂存)/
    `lwd_cloud_on_abort_notify`(门池就地清理;暂存与已准入走
    finish_requests,RequestStatus 运行时 import 走调度器台账例外);
  - 请求工厂绑定:`lwd_cloud_bind_request_factory`(装配层唯一建请求点
    经此注入;工厂未绑时过门告警留门,不丢请求);
  - 消费水位推导 `lwd_cloud_publish_consumed_watermarks` 与统计
    `lwd_cloud_stats`(门池/门标记/暂存规模);
  - 准入时序由轮询改事件驱动:过门即进暂存池,源每拍轮询 +
    5s 节流诊断日志随之删除(PRE_OUT 单连接 FIFO,预告后到由乱序
    防御兜底)。
- **LwdCloudCore 收敛为 EngineCore 强耦合残部**:PRE_OUT 泵
  (`_drain_control_plane`:线上语义 offset==0 判首留泵侧,门信号转发
  调度器接口)、数据面接缝(`_record_chunk_notify` seqno 登记 + hint/
  drop 转发,§10.1 接缝名同步台账)、步体(`lwd_native_step_bq_prefill_
  only`)与驱动(`run_busy_loop`/`_has_work`/`_process_engine_step`,
  has_work 不再含门池 —— 过门即进暂存,has_requests 已覆盖);删除
  `_try_fast_forward`(死代码,方案 §4 删除项)/`_handle_add_request`/
  `_admit_pending`/`_admit`/`_publish_consumed_watermarks`/`lwd_stats`。
- **装配**:`_lwd_cloud_build_admit_request` → `_lwd_cloud_build_request`
  (只构建 Request,不再代调 add_request),经
  `lwd_cloud_bind_request_factory` 绑给调度器;LwdCloudCore 构造删
  admit_request 注入线;§10.1 数据面挂载点条目同步改名。
- **验证**:ruff check/format、lwd_check_budget、py_compile、
  check_functions 全绿;stub 冒烟扩展:门池暂收(未过门不进暂存)/
  过门经工厂进暂存/乱序防御(预告先行)/abort 门池态与暂存态分路径/
  工厂未绑防御/水位空簿记;运行时单测仍随 §10.6 backlog。
- **未决承接**:步体/驱动/has_work 守卫 #6/runner 空批契约修复仍属
  step 面,按 prefill_only_core_reuse 方案另批落地;文件删除(用户
  诉求终态)待步体面收敛后随 step_wrapper 机制一并评估。

### 10.12 云侧脱离 step_wrapper:空批垫片 + 桥线程,lwd_cloud_core 删除
(2026-09-08,用户裁定;承接 prefill_only_core_reuse 方案的云侧部分)

> 用户裁定:适配尽量收在调度器;经推演权衡,泵线程与垫片落装配层
> (调度器实例摸不到 executor/订阅通道,装配层是接线本职),调度器
> 只增注入接口与关停转发。core.py 与 vllm-ascend **零改动**。

- **执行流推演结论**(换调度器 + 原生循环的直接模拟):启动换装/调度
  步/输出簿记/空闲唤醒六环节天然通过;两个真断点 = ①云部署无前端,
  input_queue 无人喂(引擎聋)②fork runner 0-token 批回 None 撞
  core.py:576;两处小接线 = 提升/abort 的 marshal 与关停路由。
- **空批契约垫片**(`lwd_cloud_assemble.lwd_cloud_install_empty_batch_
  contract`):装配期包装 model_executor.execute_model,0-token 派发
  短路为预完成 EMPTY_MODEL_RUNNER_OUTPUT(上游契约值;原步体增强 1
  的下沉形态,附带省一次 worker 往返)。原 83 行步体拷贝
  (`lwd_native_step_bq_prefill_only`)与 [PO-RPC] 预检/相位日志随之
  删除(§2.2"日志/预检可选诊断可弃"落地);runner 侧修复落地后垫片
  可整体拆除。非 PO 不安装,PD/原生零影响。
- **PRE_OUT 桥线程**(`lwd_cloud_assemble.LwdCloudBridge`,daemon):
  循环 drain → 翻译三类消息;数据面接缝(seqno 登记/hint/drop)与
  首预告线上语义(offset==0 判首)留桥侧,门接口转发调度器。
  生命周期:try_assemble 启动;停桥句柄经 bind 注入调度器。
- **两线程分工(零锁纪律)**:桥线程独占门状态与数据面接缝;调度
  状态(_staged/waiting/running)只经 input_queue 的 ADD/ABORT 原生
  元组分发在循环线程变更。has_work 守卫 #6 挂账就此消解 —— 桥投
  input_queue 直接唤醒阻塞 get(WAKEUP 同款机制),不依赖轮询。
- **调度器新增**:`lwd_cloud_bind_bridge(admit_sink, abort_sink, stop)`
  三注入(缺省直进/自终结,兼容直连形态与单测);`shutdown()` 覆写
  转发停桥 —— 原生 shutdown 无条件调 scheduler.shutdown,core.py
  关停路径零改动;`lwd_cloud_control_plane_bound()` 为装配幂等判据。
  `lwd_cloud_on_abort_notify` 收缩为纯门清理(终结经 ABORT 分发)。
- **删除**:`lwd_cloud_core.py` 整文件(LwdCloudCore +
  lwd_native_step_bq_prefill_only + run_busy_loop/_has_work/
  _process_engine_step/lwd_stats);装配层云侧端口/视图适配器与
  lwd_cloud_main 入口形态(零引用,§10.12 记录在案);
  lwd_step_core 的 LwdCloudSchedulerView 协议与 LwdEnginePort 云方法
  (边侧只用 lwd_scheduler/lwd_execute_model)。
- **接线变化**:`lwd_cloud_try_assemble` 幂等判据改
  control_plane_bound(不再看 step_wrapper),角色判定先于调度器触达
  (原生调度器无 Lwd 接口);`lwd_shutdown` 收缩为仅边角色(云关停
  走调度器钩子);预算例外迁移:lwd_cloud_assemble 增 v1.outputs
  (垫片契约值)与 v1.engine(ADD/ABORT 请求元组类型),lwd_cloud_core
  条目随文件删除;getattr 容忍文件集删 lwd_cloud_core.py。
- **云侧文件终态**:lwd_cloud_phase_scheduler.py(纯调度 + 门 +
  注入接口)+ lwd_cloud_assemble.py(装配 + 垫片 + 桥)两个文件;
  check_functions 全目录首次全绿(超限步体随删除清零)。
- **验证**:ruff check/format、lwd_check_budget、py_compile、
  check_functions 全绿;stub 冒烟扩展桥注入用例(提升走 sink 不直改
  暂存/abort 走 sink/shutdown 转发停桥且幂等);推演核对的运行时项
  (input_queue 唤醒、finally 关停链、原生 ADD 校验轻量)以行号记录
  于推演记录,真机双进程冒烟仍随 §10.6 backlog。

### 10.13 外部 block_hash 接入适配:链随预告下发(2026-09-08)

> 依据:《外部block_hash接入修改清单》(v0.23.0_ori)+ 用户澄清:
> 哈希链的真实数据源是边侧 ZMQ 预告消息的成员变量,不存在 HTTP 外部
> 服务(方案文档的 HTTP 客户端变体不落位)。core.py 零改动。

- **云侧为什么需要**:云 prompt token 是占位零值,本地哈希算不出真实
  链 —— 链随边侧预告下发,云侧前缀缓存按真实内容命中(边云同源)。
- **线上消息**(`LwdRequestNotify` 增 `block_hashes: list[bytes] = []`,
  缺省空,msgspec 缺省字段旧格式解码兼容):边侧本地算好的 prompt 全量
  满块链,自位置 0 起,`len == num_prompt_tokens // hash_block_size`,
  字节语义与本地 sha256 算法一致(边侧本地 hasher 的产物,同一性由
  构造保证)。边侧出口 `lwd_edge_notify_request` 增参透传;调用方随
  边 add 路径落位时传入 Request.block_hashes(挂账,当前无调用方)。
- **云侧获取**(`_lwd_cloud_build_request` 工厂内,wire hasher 闭包):
  prompt 首建(链空 + 无输出)且边侧链长度与期望满块数一致 → 直接用
  wire 链;decode 续算回本地 hasher,外链末块即本地续算 parent
  (append-only 无缝衔接);边侧未提供/长度不符回退本地(占位链,命中
  无效但不崩,fail-open)。prefix caching 未启用(无 hasher)时请求
  不挂 hasher,整链机制不激活。原方案的 env/HTTP 客户端/长度重推
  (get_hash_fn_by_name 链路)不落位。
- **落位**:链的获取逻辑住云装配请求工厂(哈希链是请求构建期知识,
  非调度决策);`vllm/v1/external_block_hash.py` HTTP 变体已建即删
  (被 wire 方案取代,记录在案);预算无新增例外(kv_cache_utils
  例外已在)。
- **对比原方案**:取数时机从"输入线程首次建链"变为"边侧预告时已定、
  云构建即用"(桥线程零网络等待);接口 key 仍为 prompt token ids
  的哈希链;部署契约不变(§5:全量满块链/字节一致/hash_block_size
  对齐)。wire 体积:每请求 32B × 满块数(8k prompt/16 块粒度 ≈
  16KB),PRE_OUT 有界队列(1000)语义不变。
- **验证**:静态套件全绿;wire 冒烟(新字段往返/旧格式缺省兼容/其它
  消息类型不受影响);原方案 §6 清单的哈希逻辑分支(链长校验/续算
  本地/缺失回退)由工厂闭包逐条对齐;真机项 = 边云同源 prefix cache
  命中观察,随 §10.6 backlog。

### 10.14 云侧引擎子类化:lwd_cloud_engine 替代 lwd_cloud_assemble(2026-09-08,用户裁定)

**决策**:云侧从"构造期 scheduler_cls 注入 + 装配层接线"升级为"引擎子类"——
`LwdCloudEngineCore(EngineCoreProc)` 经 core.py `run_engine_core` 类选择点出生即
云形态,`lwd_cloud_assemble.py` 整文件删除。引擎子类收敛为五接口、
`__init__` 两行介入(用户裁定的两段式):`__init__` = super + 两调用
(空批垫片、ZMQ 装配),`_process_input_queue` 收发接入(三类 PRE_OUT
消息是协议:request 进门池 / range offset==0 开门 / abort 终结),
`_lwd_pump_pre_out` 收取分派,`_lwd_build_request` 请求构建。
动机:装配/引擎态归位、端口适配器消失(出口直接 `self.input_queue` 原生
marshal)、线程生命周期原生(subscriber 建于 `__init__`,停桥仍经原生
`scheduler.shutdown` 钩子转发)。数据面接缝(hint/drop 转发)未随迁 ——
worker `cloud_recv_hint_mq` 本就未接线,接缝随 §9.12 数据面落位时再接。

**上游接线变化**:core.py 新增第 5 处 additive 守卫 —— `run_engine_core`
类选择点(经本模块 `lwd_resolve_engine_cls` 分流,非 PO/边角色返回 None 用
原生 `EngineCoreProc`;MoE+DP 走 `DPEngineCoreProc` 分支不换类)。原 4 处
守卫保留(服务边侧)。注意:spawn 按模块引用 pickle 目标函数,父进程
monkeypatch 不可达子进程,类选择必须在子进程内做。

**通信层**:subscriber 回归无线程纯句柄 —— `recv_available(timeout)` 阻塞
至多有消息或超时返回整批;云侧收发移入 run_busy_loop 循环线程
(`_process_input_queue` 覆写:空闲阻塞等 PRE_OUT 替代原生
`input_queue.get()`,headless 云无客户端请求,前者才是真实工作源;
超时轮询保 signal 关停响应)。门/暂存/调度簿记随之全单线程化,调度器
缺省直进即正确 —— 桥线程、接收回调、sinks marshal、跨线程纪律约束
整体消失;步执行在途期间 PRE_OUT 缓冲于 zmq socket,下一轮循环消化
(与桥线程设计的实际处理时机一致)。客户端 UTILITY/EXECUTOR_FAILED
经 input_queue,空闲期最坏感知延迟 = 空闲轮询超时(0.1s)。

**随之完成的清理**:
- seqno 注册表旁路(`_po_chunk_seqnos`)删除 —— §10.12 tag 匹配裁决的
  彻底执行,其"未初始化即 AttributeError 致首预告门永不开"的缺陷随之
  消除;线上 `LwdRangeNotify.seqno` 与 hint 转发保留(§9.3/§9.12 数据面
  接缝,worker tag=seqno 匹配用)。
- 消费水位推导(`lwd_cloud_publish_consumed_watermarks`/`_lwd_consumed_sent`)
  删除 —— 全仓无调用方的死代码,§9.1"无结果面无水位"裁决的彻底执行。

**装配顺序不变量**:工厂/出口绑定先于订阅通道创建(subscriber 构造即
启动接收线程,顺序即序,消除"消息先到、工厂未绑"窗口 —— 原 assemble
"绑定后才 bridge.start()"的零窗口语义保真)。

**台账**:checker 文件例外表 `lwd_cloud_assemble` → `lwd_cloud_engine`
(新增 `engine.core` 例外:EngineCoreProc 继承);getattr 容忍文件同步换名;
继承例外 2→3。线程纪律不变:接收线程独占调度器门状态
(`lwd_cloud_on_*_notify`),调度状态变更一律经 input_queue marshal,
接收线程不触达 `_staged`/`requests`。

**修订(同日,用户裁定,以本段为准)**:
1. 空批契约垫片移除 —— 相位调度器刻意空步的崩溃防护挂 runner 侧
   契约修复(复现表现 = core.py:576 RuntimeError),垫片代码整体删除;
2. 首预告门/PRE_OUT 处理自调度器迁云引擎子类 —— §10.11 的桥线程
   前提(跨线程触达调度器需接口收编)消失,调度器
   `bind_*`/`on_*_notify`/`_lwd_promote`/门状态整体删除,
   只保留 §10.10 准入(暂存池/释放闸)与相位排批;
3. `_lwd_setup_zmq` 收敛为仅建订阅通道(+ 门状态);
4. 通信层 subscriber 回归无线程纯句柄(阻塞 `recv()` 一条解码一条);
5. **接收驱动点定稿(最终形态)**:core.py 零改动(`_process_input_queue`
   与 `process_input_sockets` 均保持原生)—— 引擎子类覆写原生
   socket IO 线程入口 `process_input_sockets`,super 引用父类原版
   照跑(前端消息零复制),本线程跑边侧 PRE_OUT 循环,过门请求转
   Request 投 input_queue 走原生 `(ADD, (request, 0))` /
   `(ABORT, [rid])` 分发(`_handle_client_request` 零改动)。空闲唤醒
   由原生机制自然解决(IO 线程 put ADD 唤醒主循环的
   `input_queue.get()`),无轮询、无空闲切换。门归该 IO 线程独占,
   调度器只经 input_queue 被主循环碰。
6. **准入收敛 immediate(用户裁定,最终形态)**:separate_phases
   准入族(暂存池/释放闸/`add_request`/`has_requests`/`finish_requests`
   覆写/`LWD_CLOUD_IMMEDIATE_ADMISSION` 开关/Immediate 对照子类)
   整体删除 —— 引擎过门即投 input_queue,原生 `add_request` 随到随
   调度;调度器只剩纯相位排批(prefill_first/decode_first),注册表
   二维收敛一维(相位名),`LwdConfig.admission_name` 配置项删除。
