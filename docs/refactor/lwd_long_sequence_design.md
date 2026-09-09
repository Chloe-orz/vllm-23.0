# Lwd 长序列方案设计:chunkNotify 强制调度

> 状态:设计定稿,待实施。
> 代码基线:`prefill_only_base_br/vllm`(含 lwd_control 边云骨架)。
> 关联文件:`vllm/v1/core/sched/scheduler.py`、`vllm/v1/core/sched/async_scheduler.py`、
> `vllm/v1/lwd_control/control_communication/lwd_notify.py`、
> `vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_engine.py`、
> `vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_phase_scheduler.py`、
> `vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_scheduler.py`。

---

## 1. 背景与目标

### 1.1 方案定位

长序列场景下,prompt 无法(或不值得)在单一引擎内一次性完成 prefill。本方案将
prefill 拆到边侧纯 prefill 引擎上分块流水执行,云侧作为 decode 引擎消费边侧
产出的 chunk。整条链路的调度节奏由**数据到达**驱动:边算一块、通知一块、云算一块。

现状(已落地):

- 边侧 `LwdEdgeScheduler`(纯 prefill):请求进入 waiting,原生 chunked prefill
  分块调度,每块经 `LwdRangeNotify(request_id, offset, num_tokens, seqno)` 预告云端,
  prompt 全部嵌入完成即本地终结(`lwd_edge_update_progress`)。
- 云侧 `LwdCloudEngineCore`:PRE_OUT 线程收三类 notify。
  - `LwdRequestNotify` → 门池 → 构建占位请求(prompt 为全零占位,哈希链用边侧
    预告链)→ `input_queue(ADD)` → 原生 `add_request` → **waiting**(下文简称
    **reqNotify 路径**,已通);
  - `LwdRangeNotify` → **当前是空消费点**(`lwd_cloud_engine.py` `_lwd_dispatch`
    的 RangeNotify 分支,注释"随数据面落位再接");
  - `LwdAbortNotify` → 原生 abort 路径。
- 云侧 `LwdCloudPhaseScheduler`:prefill_first / decode_first 纯相位批次,
  容器交换手法,**原生 schedule() 零改动**。

### 1.2 本设计要解决的问题

到达云侧的请求有两种形态:

| 形态 | 语义 | 现状 |
|---|---|---|
| **reqNotify** | 新请求,建请求进 waiting,走原生生命周期 | 已实现 |
| **chunkNotify** | 不进任何既有队列;指令语义:**强制调度 running 中某请求的指定 chunk** | 本设计 |

没有 chunkNotify 时,云侧占位请求的 prefill 推进与边侧数据产出**解耦**:云侧会
按自身预算一路把占位零值算完,与数据面落位节奏无关。chunkNotify 把调度面
gate 从"数据面职责"收编到调度器:云侧每一步只算**已通知到达**的范围,并且
对该范围给与**强制优先**。

### 1.3 设计目标

1. 云侧 prefill 推进水位 == 已收到的 chunk 水位(不跑在数据前面);
2. chunkNotify 指定的 chunk 在下一步即被调度,且优先于其它 running 工作;
3. 原生 `schedule()` 主体零改动(唯一例外见 §4.2,一处 `min`);
4. 强制语义复用原生机制:token 预算、抢占、回退、worker 同步全部继承,
   不新增平行调度路径。

### 1.4 非目标

- 不支持乱序 chunk 的乱序**执行**(见 §6-2:执行仍严格按序,乱序只影响暂存);
- 不改变边侧行为(边侧已上线的 notify/对账/终结逻辑一律不动);
- 不动数据面(张量传输与落位不在本设计范围)。

---

## 2. 设计依据:原生调度器如何调度一个 running 请求

云侧调度器实际类链:`LwdCloudPhaseScheduler(AsyncScheduler(Scheduler))`。
理解以下机制是本设计正确性的前提。

### 2.1 调度入口与两阶段

`EngineCore.step()` 每步调用 `scheduler.schedule()`(core.py:481),产出
`SchedulerOutput` → worker 前向 → `update_from_output` 回收。schedule() 内部:

- **阶段 1(running 优先)**:按列表序遍历 `self.running`,逐请求算
  `num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens`
  (scheduler.py:403-407),经 `long_prefill_token_threshold`、剩余
  `token_budget`、`max_model_len` 截断——chunked prefill 即由此产生;然后
  `allocate_slots` 分配增量 KV 块,失败则抢占 running 尾部(FCFS 下
  `running.pop()`,scheduler.py:499)。
- **阶段 2(waiting 补位)**:本步无抢占时,从 waiting/skipped_waiting 提升,
  受 `max_num_seqs`、token_budget、KV 显存约束;原状态 WAITING 进
  `scheduled_new_reqs`,PREEMPTED 进 `scheduled_resumed_reqs`。

### 2.2 关键状态机

- 调度是**承诺**:阶段 1 只记账,`_update_after_schedule` 在调度时**乐观**
  推进 `num_computed_tokens`(scheduler.py:1003)并置 `is_prefill_chunk`;
  步末 `update_from_output` 按实际执行量勾稽。
- AsyncScheduler 占位符机制:非 prefill-chunk 的请求在
  `_update_after_schedule` 中 `num_output_placeholders += 1`(预期采样帧,
  async_scheduler.py:33),采样落地时在 `_update_request_with_output` 回冲。
- prefill chunk 的模型输出为空 token ids,不追加 `_output_token_ids`,
  不触发 stop。

> 依据结论:**"调度一个 running 请求的一个 chunk"所需的全部状态变更都内联在
> 阶段 1 + `_update_after_schedule` + `update_from_output` 中,没有可独立调用
> 的"调度单个请求"函数。** 这决定了 §4 的实现形态。

---

## 3. 总体设计

### 3.1 核心决策:指令队列,不是调度单元

chunk 队列**只存指令,不存调度过程**。schedule() 开头 drain 队列,把指令翻译成
两个作用量,然后照常走原生阶段 1:

1. **队首重排**:目标请求移到 `self.running[0]`;
2. **范围钉扎**:给请求设 `cap_end`(绝对位置),阶段 1 的 `num_new_tokens`
   计算多 min 一项。

```
chunkNotify(主循环分发)──> _lwd_chunk_queue.append((req_id, offset, num_tokens, seqno))

schedule():
    drain 队列,逐条校验生效(见 §6):
      请求不存在 / 已算完该范围      → 丢弃(幂等)
      offset >  num_computed_tokens → 留队,下步重验(乱序等待)
      offset == num_computed_tokens → 生效:挪 running[0] + 设 cap_end
    相位:chunk 队列非空 → 强制 prefill 相位
    super().schedule()  ── 原生阶段 1 自然先调它、范围被 cap 钉住
```

**为什么重排即"强制"**:阶段 1 按列表序消费共享 token_budget,队首先挑预算;
抢占从尾部弹,队首最后被牺牲。两个语义一次拿全,原生逻辑零改动。

**为什么不给队列写独立调度分支**:阶段 1 的调度体(allocate_slots、预算扣减、
抢占回退、记账)全部内联且无单请求抽象,复制必然漂移;且两条路径操作同一份
`num_scheduled_tokens`/`token_budget`,正确性无法论证。

### 3.2 消息与指令定义

`LwdRangeNotify` 的消费分支即 chunkNotify 入口(消息字段已够用:
`request_id / offset / num_tokens / seqno`;offset/num_tokens 取自边侧原生
调度决策,天然按块对齐)。新增调度器侧结构:

```python
@dataclass
class _LwdChunkDirective:      # 队列条目
    request_id: str
    offset: int
    num_tokens: int
    seqno: int

# LwdCloudPhaseScheduler.__init__:
self._lwd_chunk_queue: deque[_LwdChunkDirective] = deque()
# 同请求去重:已生效/已完成的 (request_id, offset) 集合
self._lwd_chunk_seen: set[tuple[str, int]] = set()

# Request 附加字段(附加,不改动既有字段语义):
request.lwd_cap_end: int | None   # 本请求当前强制范围终点;None = 无 cap
```

### 3.3 cap 的作用点(对原生的唯一改动)

阶段 1 的 `num_new_tokens` 截断链(scheduler.py:410 附近)追加一项:

```python
if request.lwd_cap_end is not None:
    num_new_tokens = min(num_new_tokens, request.lwd_cap_end - request.num_computed_tokens)
```

cap **持久到满足为止**:chunk 大于剩余 budget 时,原生自动切块、下步续调,
cap_end 保持,直到 `num_computed_tokens >= cap_end` 时清除。不要做成一次性,
否则大 chunk 只会被调一半。

---

## 4. 数据结构影响清单

### 4.1 自动继承(设计正确的验证标准:下表所有项的赋值不出现在新代码里)

调度时:

| 结构 | 谁改 | 说明 |
|---|---|---|
| `request.num_computed_tokens` | `_update_after_schedule`(scheduler.py:1003) | 乐观推进,勾稽前是承诺值 |
| `request.is_prefill_chunk` | 同上(scheduler.py:1005) | cap 截短后自然为 chunk 中间态 |
| `request.num_output_placeholders` | async_scheduler.py:33(仅非 chunk) | 强制 chunk 补完 prompt 的那步自动转入采样帧,云侧本就要 decode,语义正确 |
| `request.spec_token_ids` / `next_decode_eligible_step` | async_scheduler.py:36/41 | 云侧 spec 关闭,无害 |
| `token_budget`、`num_scheduled_tokens`、`req_to_new_blocks`、`scheduled_running_reqs` | 阶段 1 内联 | 打进 SchedulerOutput |
| `kv_cache_manager`(块表 / BlockPool / 前缀缓存 touch) | `allocate_slots` | 只提供数量,分块复用全自动 |
| `self._inflight_prefills` | 阶段 2 加 / update_after_schedule 摘 | 云侧无 KV connector,无消费者 |
| `self.prev_step_scheduled_req_ids` | schedule 尾部 | 自动 |

执行后:

| 结构 | 谁改 | 说明 |
|---|---|---|
| `request._output_token_ids` / `_all_token_ids` | `_update_request_with_output` | prefill chunk 输出为空,不追加 |
| `request.num_output_placeholders` | async_scheduler.py:59 | 采样帧落地回冲 |
| `kv_cache_manager.cache_blocks(...)` | async_scheduler.py:64-66 | 按真实水位落前缀缓存;cap 来源是边侧原生决策(offset/num_tokens 块对齐),满块哈希正常进链,**无需处理,但禁止把 cap 改成非对齐切分** |
| `self.running` / `self.waiting` 的 stop 移除 | scheduler.py:1574-1579 | chunk 不触发 stop |

worker 侧:`scheduled_cached_reqs`(CachedRequestData)对 running 请求只发
增量,worker input_batch 自动跟上。worker 完全不感知强制语义。

### 4.2 新代码需要维护的全部状态(共三项)

1. **`lwd_cap_end` 生命周期**:drain 时设置;`num_computed_tokens >= cap_end`
   即清;请求被抢占(`num_computed` 归零)后阶段 2 会重新全量 prefill——
   阶段 2 不读 cap,语义自动正确,但 drain 时要清掉过期 cap,防残留截断。
2. **队列条目与请求生命周期同步**:finish / preempt / 请求查无即弃条目。
3. **勾稽回退**:本步未执行量回退 `num_computed_tokens`、置
  `is_prefill_chunk = True`,cap 不动,下步重试——**照抄边侧
  `_lwd_reconcile_progress` 语义**(lwd_edge_scheduler.py:178-191),不新发明。

---

## 5. 与云侧相位调度器的配合

- **相位联动**:decode_first 相位下 prefill 尾巴被容器交换藏起
  (lwd_cloud_phase_scheduler.py:103-122)。chunk 队列非空必须强制回 prefill
  相位:在 `_prefer_prefill()` 增加 `or bool(self._lwd_chunk_queue)`。
- **重排时机**:drain 与重排发生在 `schedule()` 覆写入口、容器交换**之前**;
  `_lwd_split_running` 保序,强制请求是 prefill tail(`num_computed <
  num_prompt`),必然进 prefill 相位可见集,队首位置随交换保留。
- **队首优先级每步重设**:容器交换会临时替换 `self.running`,强制优先
  不能设一次管永久;drain 本就每步执行,天然满足。

---

## 6. 边界语义规约(实现前定死)

| # | 场景 | 语义 |
|---|---|---|
| 1 | `offset < num_computed_tokens`(重复/迟到预告) | 幂等丢弃;按 `(request_id, offset)` 去重(与 lwd_notify.py RangeNotify 幂等约定同款) |
| 2 | `offset > num_computed_tokens`(乱序到达) | **留队等待,禁止丢弃**。原生 num_computed 严格顺序推进、KV 块按序连续分配,跳块不可执行;每步 drain 重验,水位追上即生效 |
| 3 | 同请求多条 chunk | 合并相邻区间,cap_end 取最大;seqno 定序 |
| 4 | 请求仍在 waiting(从未被调度) | no-op 丢弃:首次 prefill 本覆盖从头起的范围;chunk 对未调度请求无独立意义 |
| 5 | 请求已 decode-ready(prompt 算完) | 丢弃(stale 指令) |
| 6 | 请求被抢占 / abort / finish | drain 时查 `self.requests` 查无或非 running 即弃;强制请求在 running 队首,被抢占概率最低(抢占从尾弹) |

补充约定:

- **预算关系**:若 chunk 体积超过 `max_num_scheduled_tokens` 全局预算,"强制"
  只保证"本步开始算、连续优先地算完",不突破总预算(原生 assert
  `total <= max_num_scheduled_tokens` 不被打破,仍走正常扣减)。
- **失败重试**:`allocate_slots` 失败时原生 break,cap 与队列条目保留,下步
  自然重试;执行未落地(notify 队满等)走 §4.2-3 勾稽。
- **线程边界**:队列只被引擎主线程碰。入队走既有惯例:PRE_OUT 线程 →
  `input_queue` → 主循环分发时推进(与 ADD/ABORT 同款,零锁)。

---

## 7. 实施清单

| # | 改动 | 位置 | 规模 |
|---|---|---|---|
| 1 | RangeNotify 分支激活:`input_queue` 投递 + 主循环分发 | `lwd_cloud_engine._lwd_dispatch` 空分支 + core.py 分发一行 | ~30 行 |
| 2 | chunk 队列 + drain + 重排 + cap 设置 + 去重/乱序等待 | `LwdCloudPhaseScheduler`(新增方法 + schedule 入口) | ~60 行 |
| 3 | per-request cap 字段 + 阶段 1 一处 min | `Request` 附加字段、scheduler.py:410 附近 | ~10 行 |
| 4 | 相位联动 | `_prefer_prefill` | ~5 行 |
| 5 | 勾稽回退 | 云侧调度器(抄边侧模板) | ~30 行 |

合计 ~140 行;对原生的侵入仅 #3 的一处 `min`,其余全部在 lwd_control 层。

实施顺序建议:#3(cap)先行并配单元验证 → #2(drain/重排/去重)→ #1(通道)
→ #4/#5。每步均可独立回归(不接通道时队列恒空,行为退化为原生)。

---

## 8. 备选方案与否决理由

| 备选 | 说明 | 否决理由 |
|---|---|---|
| 队列走独立调度分支 | schedule() 内为 chunk 队列单写取出→分配→记账路径 | 阶段 1 调度体内联无单请求抽象,复刻必漂移;双路径共写 `num_scheduled_tokens`/`token_budget` 无法论证正确性 |
| 纯"可见集交换" | 容器交换令 `self.running = [强制请求]` 跑一步 | 强制 chunk 小于预算时其余 running 全停一步,吞吐浪费;且多请求强制时仍需 cap |
| 请求水位增长建模 | reqNotify 建短请求,chunkNotify 追加 `num_tokens`/`_all_token_ids` | 侵入 Request 内部(`_all_token_ids`、`num_prompt_tokens`、block 哈希链前缀重算),改动面大于一个附加 cap 字段;留作后续演进路径 |
| 乱序执行 chunk | 按 offset 任意序分块计算 | 与原生 num_computed 顺序推进 + KV 块连续分配假设根本冲突,等于重构块分配;若数据面无法保证 per-request 有序,应在数据面/边侧解决有序性,而非调度器 |

---

## 9. 风险与验收

风险点:

1. **乱序语义**(最高风险):数据面若不能保证 per-request chunk 有序,队列
   退化为等待缓冲,长序列尾延迟受影响;需与数据面确认有序性承诺。
2. **cap 残留**:抢占/abort 路径漏清 cap 会导致后续调度被错误截断;验收需
   覆盖"强制中被打断"用例。
3. **勾稽一致性**:强制 chunk 与 notify 队满重试叠加时,回退量必须与边侧
   对账语义一致(单一事实源:`_lwd_last_scheduled` 同款模式)。

验收标准(逐条可测):

- [ ] 云侧 prefill 推进水位恒 <= 已收 chunk 水位(无数据时云侧零 prefill 调度);
- [ ] chunkNotify 到达后,下一调度步该请求位于批首且 `num_scheduled_tokens`
      恰为 min(chunk 量, 剩余预算);
- [ ] 大 chunk 跨步续调,cap 持续生效至 `cap_end`,随后自动清除;
- [ ] 重复/乱序/waiting/decode-ready/abort 五类边界条目按 §6 处置,队列无泄漏;
- [ ] 不接通道(队列恒空)时,云侧调度行为与改造前逐位一致(退化安全);
- [ ] 新代码对 §4.1 所列既有结构零赋值(评审检查项)。
