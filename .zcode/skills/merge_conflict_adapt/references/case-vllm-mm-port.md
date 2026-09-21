# 真实案例:多模态 PR 跨基线移植(vllm + vllm-ascend 双仓库)

> 本文件是 merge_conflict_adapt skill 的参考案例,记录一次完整移植的全貌,供下次适配时对照模式。产出文档均在 `/Users/wangwei/prefill_only_qwen_mm_pr/` 下:`ascend_v01_port_design.md`(方案)、`port_pr_diff_review.md`(校验报告)、`pr22_pr52_code_walkthrough.md`(原 PR 逐块走读)。

## 背景

- 源:PR #22(vllm)/ #52(vllm-ascend),`feat(lwd): prefill_only 支持 Qwen 多模态(image)`,单提交,目标 prefill_only_v2。
- 新优先基线:`aolin123/{vllm,vllm-ascend}_v23.0_prefill_only` @ `prefill_only_v0.1_mtp_perf`(用户口述 `mtp_perf_test`,ls-remote 求证后纠正)。
- 拓扑:两条基线在 PR 提交父级附近分叉;PR 基线没动过受影响文件,目标基线在 ascend 侧重构了 UP 数据面。

## 分叉的实质(决定了哪些块要重设计)

目标基线把 UP 数据面从"一段式 world broadcast + host 组装缓冲"重构为:

- **两段式通道**:通道层只做端点 P2P(isend/irecv);TP 组内广播挪到云 runner 消费时刻、计算流上以 launch+wait 原子对完成。动因:torch_npu 跨流排序不可依赖,post 时刻广播会交付残缺。
- **设备侧注入**:收到的 embeds 直写 `inputs_embeds.gpu` 调度窗口;CPU 组装缓冲降级为旁路(仅供 MTP draft provider,stream 序 D2H)。
- **消费点收进 runner**:worker 层的 `take_lwd_up_embeds`(host 阻塞收割)消失,非端点 rank 的 future=None 语义只有 runner 能处理。

## 锚点验证结果(方案的地基)

| 锚点 | 验证方式 | 结论 |
|---|---|---|
| `_calc_mrope_positions` 数据源 | 读 vllm 目标基线 gpu_model_runner 实现 | 仍从 `req.mrope_positions` 表切片 + `delta` 现算 → mrope 注入方法(B16)**逐字零改动** |
| 边引擎三个调度器挂钩 | grep 引擎文件 | 全在,EMBED 批仍随完整 SO 下发(B7 依赖 `scheduled_new_reqs` 契约不破) |
| `channel.py` 两段式 | 与 v0.1 零 diff | aux 帧重设计只针对传输原语,纪律(seqno 双帧)保留 |
| `LwdEmbedBatch` 字段 | 读 output.py | 目标基线无 `token_offsets` → 直接用原 PR 的 `prompt_offsets`,vllm 侧零冲突 |

## 适配点分级落地

| 适配点 | 性质 | 结果(numstat 原→新) |
|---|---|---|
| types.py / future.py aux 字段 | 原样 | +22/−0 → +22/−0;+12/−3 → +12/−3 |
| mrope 注入方法 | 原样(锚点稳) | 逐字相同 |
| channel.py aux 双帧 | 重设计:broadcast→P2P 双 isend/irecv | +42/−8 → +46/−6 |
| cloud_worker 预挂 | 重设计:仅端点 recv + aux 预告;`take_lwd_up_embeds` 不复活 | +18/−9 → +7/−1 |
| cloud_runner 注入 | 重设计:aux 第二广播(独立原子对)+ record_stream + mrope 挂点 | +68/−5 → +83/−2 |
| edge_worker EMBED | 小改:三参穿 SO + mm helpers 原样 + 与 TTFT 探针嵌套共存 | +168/−4 → +173/−4 |

## 关键决策与理由(审核时的解释口径)

1. **aux 第二广播独立成对**:不与主广播共用等待句柄——遵守基线"launch+wait 同流原子对"纪律,防跨流乱序交付。
2. **`take_lwd_up_embeds` 删除**:新架构下非端点 future=None 会在第一行崩;TP 广播必须插在收与消费之间且在计算流上;host 阻塞 wait 会重新引入基线修掉的竞态。功能由 runner 内等价实现。
3. **fail-fast 保留基线版**:基线的行数对账(`n != scheduled` raise)比原 PR 更严,原 PR 版本是子集。
4. **`record_stream(aux_flat)` 对称补齐**:端点 recv buffer 由通道流分配器管理而 copy 在计算流,不登记会被复用覆写(跨 seqno 串数据)——基线主帧的同款坑。
5. **探针口径变化要写进报告**:移植后多模态批的 `forward` 计时包含视觉塔+merge——是口径扩展不是退化,但日志对比时会看到数字变大。

## 校验对比方法(可复用的命令模板)

- vllm 侧:patch 剔除 index 哈希 + hunk 行号后 diff 为零 → **内容 100% 一致**(仅 hunk 头平移)。
- ascend 侧:逐文件 numstat 对照 + 差异归因表(D1..D8 全部映射到方案编号,无未归因差异)。
- 提交:53bd485a7(vllm)/ 91048e7c6(ascend),均带溯源信息;分支 `prefill_only_v0.1_mtp_perf_mm` 推 qxxxw origin。

## 上机验证项(风险降序)

1. U2(最高):目标线是 MTP perf 线——draft 起草读 `req_prompt_embeds` + `should_sync_mrope_positions` 对 mm 请求的覆盖,开/关 MTP 各跑一轮。
2. U1/U5:aux 第二广播交付正确性、TP>1 非端点广播配对。
3. U3:长图跨 chunk + 引擎多批在飞的缓存摘除时序。
4. 回归底线:纯文本不进任何 `has_mrope`/`aux_*` 门控路径,应与移植前逐位一致。
