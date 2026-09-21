# Skill: merge_conflict_adapt

---
name: merge_conflict_adapt
description: 把一个 PR/分支的代码改动移植(port/cherry-pick)到另一条已分叉的基线分支上,并保证差异可审核。Use whenever 用户说"分支落后了重新适配/重做适配"、"以 XX 分支为基准重新改"、"把 PR 移植到 XX 分支"、"cherry-pick 冲突怎么解"、"换基线/换目标分支做适配"、"port to branch X"——即使没明说 port 这个词,只要意图是把已有改动在新基线上重现,就用本 skill。
---

# 把已有改动移植到分叉基线(跨分支适配)

适用场景:某个特性 PR 提在旧基线上(如 prefill_only_v2),而主线/优先分支已经走到另一条线(如 prefill_only_v0.1_mtp_perf),需要**重新适配而不是 merge**——通常为了 PR 审查面干净、或两条线的架构已分叉到 merge 会拖进大量无关变更。

配套仓库示例是 vllm / vllm-ascend 边云项目,但流程与仓库无关。完整真实案例见 `references/case-vllm-mm-port.md`(多模态 PR 跨基线移植,建议首次使用前通读)。

## 总原则(先读)

1. **锚点验证优先于动手**:移植前先找出改动依赖的接口/调用链/数据结构("锚点"),逐个在目标基线上验证是否仍成立。锚点稳 → 原样搬;锚点变 → 重设计。这一步能消灭八成的不确定性。
2. **冲突解决跟随目标基线的新范式**:把特性"编织"进基线的新结构,**绝不把旧基线的范式搬回来**。基线更强的校验/fail-fast 保留基线版,原 PR 的弱版本丢弃。
3. **特性门控零污染**:所有新增分支以特性标志(如 `has_mrope`/`aux_*`)为门,不带特性的请求不进任何新代码路径——这是"移植后老行为逐位不变"的结构保证,也是回归底线。
4. **协议字段 additive**:跨进程字段带缺省值,旧端新端可混跑。
5. **每步产出可审核的文档**:方案文档(动手前)→ 提交(带溯源)→ 校验对比报告(程序化)。差异必须全部可归因,不允许"顺手改动"。

## 工作流程

### Phase 0 澄清目标

- 确认三要素:源(PR 号/提交 sha/分支)、目标基线分支、目标远端。
- **别信口述的远端分支名**:`git ls-remote --heads origin | grep -i <关键词>` 求证。分支名差一个后缀很常见(真实案例:口述 `mtp_perf_test`,实际叫 `mtp_perf`)。
- 若源是 PR:网页 PR 页经常不渲染 diff,**本地算才准**:`git fetch` 后 `git diff origin/<target>...origin/<source-branch>`(三点号)。同时确认 PR 是否恰好等于单提交(`git log target..source`)。

### Phase 1 拓扑与分叉量化

```bash
git fetch origin <target> <source> ...        # 多远端时 git remote add + fetch
git merge-base <target> <pr-base>            # 找 fork 点
git merge-base --is-ancestor A B && echo YES # 判包含关系
```

- **单边分叉分析**(核心洞察:冲突只来自动过文件的那一侧):
  - `git diff <fork> <target> --stat -- <受影响文件>` — 目标基线走了多远
  - `git diff <fork> <pr-base> --stat -- <同文件>` — PR 基线动过没有
  - 常见结果:PR 基线没动、目标动了 → 冲突全部来自目标侧的新范式,解法是"特性织入基线"。
- 逐个检查受影响文件在目标基线上的**接口形状**(函数签名/返回元数/消费点),用 `git show <target>:<path> | grep -n <标记>` 直接看远端分支文件,不用切分支。

### Phase 2 原理理解 + 锚点验证

- 把源 PR 拆成编号代码块(B1..Bn),每块标注:所属流程、依赖的锚点、上下游接口。
- 每个锚点在目标基线上验证三件事:**还存在吗、签名变了吗、数据源/消费点还是同一张表吗**。锚点验证要看实现不只看声明(案例:`_calc_mrope_positions` 声明没变,真正要确认的是它仍从 `req.mrope_positions` 表取数——这让 B16 整块零改动)。
- 产出分级清单:原样搬 / 小改(改名/穿参)/ 重设计(换范式)。

### Phase 3 试 cherry-pick(沙箱,不脏工作区)

```bash
git worktree add -f /tmp/wt_x <target-base> --quiet
cd /tmp/wt_x && git cherry-pick <sha> --no-commit
git status --porcelain | grep -E "^UU"        # 真实冲突清单
git worktree remove --force /tmp/wt_x          # 用完即删
```

- 零冲突 ≠ 高枕无忧:**自动合并的文件也要过语义**(案例:types/future 干净合并,但消费它们的 worker 接口在基线上已整体消失,那个"干净"的方法体成了死代码)。
- 冲突数与分叉量对照:冲突文件应是 Phase 1 里目标侧动过的文件,出现意外文件要回去重查。

### Phase 4 适配方案文档(动手前写,给人审)

固定格式,每个适配点一节:

```
### A<n> <位置>
**背景差异**:基线在这条流程上变成了什么样(引用基线真实代码/注释)
**原 PR 修改**:(贴源提交代码)
**重新设计**:(贴适配后代码)
**为什么**:为什么这么改、为什么某些原 PR 代码不移植
```

外加:总览表(适配点 × 性质 × 原 PR 块号)、跨仓库一致性清单(双仓库联动时)、不确定项表(编号 + 风险级 + 验证方式)。方案里明确**哪些原 PR 接口不复活、被什么等价替代**——这是审核最关心的决策。

### Phase 5 正式落地

```bash
git checkout -b <target>-<feature> <target-base>   # 命名惯例:<基线>-<特性>
git cherry-pick <sha>                              # 解冲突用 Edit 工具逐块处理
```

- 解冲突纪律:每个冲突块先看两侧各自的意图,按 Phase 4 方案选边或嵌套;两侧改了同一行但意图正交时(案例:perf 探针 vs 特性分支)**嵌套共存而不是二选一**。
- 删除基线上已消失的接口时,留一行注释说明去向(功能等价物在哪)。
- 完工自检三件套:
  1. `grep -rn "^<<<<<<<\|^>>>>>>>" <目录>` 无残留;
  2. `python3 -m py_compile <所有触碰文件>`(或对应语言的语法检查);
  3. 关键逻辑点抽查:新分支里引用的变量都有定义、调用点同步改了参数(如穿参 scheduler_output 时两端一起改)。
- 提交信息带溯源:`Port to <base> from <sha> (PR #n)`,正文列适配点清单。

### Phase 6 校验对比报告(程序化,给人审)

- **patch 规范化对比**(判断"内容是否逐行一致"):
  ```bash
  git show <orig-sha> > /tmp/a.patch; git show <new-sha> > /tmp/b.patch
  # 剔除 index 哈希与 hunk 行号后 diff;为零则内容 100% 一致
  sed -n '/^diff --git/,$p' /tmp/a.patch | grep -v "^index " | \
    sed 's/@@ -[0-9]*,[0-9]* +[0-9]*,[0-9]* @@/@@H@@/' > /tmp/a2.patch
  # 同样处理 b.patch,diff /tmp/a2.patch /tmp/b2.patch
  ```
- 逐文件 numstat 对照表:原 PR `+x/−y` vs 移植 `+x'/−y'`。
- **差异归因表**:每处差异编号(D1..Dn)+ 归因(对应方案 A 编号)+ 风险级。审核者只看这张表就能逐条核对。
- 验证项清单:上机冒烟项,按风险排序,注明"哪些是移植新增的设备/运行时行为"。
- 推送(`git push origin <branch>`)+ PR 创建指引(源/目标/标题/描述附归因表)。没有远端 API 权限时,给出精确的源分支 → 目标仓库:分支,让用户网页操作。

## 常见坑

- 网页 PR 页显示"0 文件改动"多半是渲染问题,以本地三点 diff 为准。
- 本地工作分支可能比远端 PR 分支新(有未推送的本地提交)——对比一律用 `origin/<pr-branch>` 而不是本地 HEAD。
- 引用的行号要标注基准提交(`file:line @<sha>`),基线漂移后行号会误导。
- 多仓库联动(vllm + vllm-ascend 这类)两侧要成对验证:一侧新增的协议字段是另一侧的消费输入,写进跨仓库一致性清单。
