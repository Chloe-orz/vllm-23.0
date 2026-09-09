# Claude Code Session

> **Resume in CLI:** `claude --resume 8debdd6a-4d3a-425c-b5d9-debae51af7ab`

| Field | Value |
|---|---|
| **Project** | `c:\prefill-only-wangwei` |
| **Session ID** | `8debdd6a-4d3a-425c-b5d9-debae51af7ab` |
| **Working Dir** | `c:\prefill_only_wangwei` |
| **Started** | 2026/9/8 23:54:17 |
| **Last Updated** | 2026/9/9 09:15:28 |
| **Messages** | 110 |

---

## User <sup>2026/9/8 23:54:17</sup>

阅读一下C:\prefill_only_main\.claude-code-history，理解prefill_only和同步情况


---

## Assistant <sup>2026/9/8 23:54:20</sup>


---

## Assistant <sup>2026/9/8 23:54:21</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "ls -la \"C:/prefill_only_main/.claude-code-history\" 2>/dev/null || ls -la \"C:/prefill_only_main\" | head -50",
  "description": "Check .claude-code-history path"
}
```

</details>


---

## User <sup>2026/9/8 23:54:27</sup>

<details>
<summary>Tool Result</summary>

```
total 348
drwxr-xr-x 1 root 197121      0 Sep  8 23:53 .
drwxr-xr-x 1 root 197121      0 Sep  8 23:53 ..
-rw-r--r-- 1 root 197121 349927 Sep  8 23:53 2026-09-08_234727_ide_selectionThe-user-selected-the-lin_2731f840.md
```

</details>


---

## Assistant <sup>2026/9/8 23:54:30</sup>


---

## Assistant <sup>2026/9/8 23:54:31</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "wc -l \"C:/prefill_only_main/.claude-code-history/2026-09-08_234727_ide_selectionThe-user-selected-the-lin_2731f840.md\"",
  "description": "Count lines in history file"
}
```

</details>


---

## User <sup>2026/9/8 23:54:31</sup>

<details>
<summary>Tool Result</summary>

```
10535 C:/prefill_only_main/.claude-code-history/2026-09-08_234727_ide_selectionThe-user-selected-the-lin_2731f840.md
```

</details>


---

## Assistant <sup>2026/9/8 23:54:33</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "C:\\prefill_only_main\\.claude-code-history\\2026-09-08_234727_ide_selectionThe-user-selected-the-lin_2731f840.md",
  "limit": 800
}
```

</details>


---

## User <sup>2026/9/8 23:54:33</sup>

<details>
<summary>Tool Result</summary>

```
1	# Claude Code Session
2	
3	> **Resume in CLI:** `claude --resume 2731f840-8192-4676-a2b4-4fbc31a1a55c`
4	
5	| Field | Value |
6	|---|---|
7	| **Project** | `c:\prefill-only-main` |
8	| **Session ID** | `2731f840-8192-4676-a2b4-4fbc31a1a55c` |
9	| **Working Dir** | `c:\prefill_only_main` |
10	| **Started** | 2026/9/8 17:00:31 |
11	| **Last Updated** | 2026/9/8 23:47:27 |
12	| **Messages** | 567 |
13	
14	---
15	
16	## User <sup>2026/9/8 17:00:31</sup>
17	
18	<ide_selection>The user selected the lines 72 to 72 from c:\prefill_only_main\vllm_tywww\vllm\v1\lwd_control\control_scheduler\lwd_edge_scheduler.py:
19	def add_request
20	
21	This may or may not be related to the current task.</ide_selection>
22	
23	先阅读一下C:\prefill_only_main\1.md
24	
25	
26	---
27	
28	## Assistant <sup>2026/9/8 17:00:35</sup>
29	
30	<details>
31	<summary>Tool: <code>Read</code></summary>
32	
33	```json
34	{
35	  "file_path": "C:\\prefill_only_main\\1.md"
36	}
37	```
38	
39	</details>
40	
41	
42	---
43	
44	## User 
...(truncated)
```

</details>


---

## Assistant <sup>2026/9/8 23:54:36</sup>


---

## Assistant <sup>2026/9/8 23:54:37</sup>

历史记录长达 10,535 行。我将高效地提取关键内容（用户消息和助手文本结论），而不是逐个读取工具输出。


---

## Assistant <sup>2026/9/8 23:54:37</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "grep -n \"^## User \\|^## Assistant \" \"C:/prefill_only_main/.claude-code-history/2026-09-08_234727_ide_selectionThe-user-selected-the-lin_2731f840.md\" | head -100",
  "description": "List message boundaries in history"
}
```

</details>


---

## User <sup>2026/9/8 23:54:38</sup>

<details>
<summary>Tool Result</summary>

```
16:## User <sup>2026/9/8 17:00:31</sup>
28:## Assistant <sup>2026/9/8 17:00:35</sup>
44:## User <sup>2026/9/8 17:00:35</sup>
71:## Assistant <sup>2026/9/8 17:00:41</sup>
76:## Assistant <sup>2026/9/8 17:00:55</sup>
105:## User <sup>2026/9/8 17:06:02</sup>
114:## Assistant <sup>2026/9/8 17:06:18</sup>
119:## Assistant <sup>2026/9/8 17:06:18</sup>
136:## User <sup>2026/9/8 17:06:18</sup>
152:## Assistant <sup>2026/9/8 17:06:18</sup>
169:## User <sup>2026/9/8 17:06:19</sup>
183:## Assistant <sup>2026/9/8 17:06:24</sup>
199:## User <sup>2026/9/8 17:06:24</sup>
231:## Assistant <sup>2026/9/8 17:06:29</sup>
236:## Assistant <sup>2026/9/8 17:06:30</sup>
254:## User <sup>2026/9/8 17:06:30</sup>
275:## Assistant <sup>2026/9/8 17:06:30</sup>
294:## User <sup>2026/9/8 17:06:31</sup>
314:## Assistant <sup>2026/9/8 17:06:34</sup>
332:## User <sup>2026/9/8 17:06:34</sup>
373:## Assistant <sup>2026/9/8 17:06:42</sup>
378:## Assistant <sup>2026/9/8 17:06:42</sup>
394:## User <sup>2026/9/8 17:06:42</su
...(truncated)
```

</details>


---

## Assistant <sup>2026/9/8 23:54:42</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"C:/prefill_only_main/.claude-code-history\" && awk '/^## User /{u=1; buf=\"\"; next} /^## Assistant /{u=0; next} u && !/^<details>|^```json|^```$|^<\\/details>|^Tool Result/{buf=buf\"\\n\"$0} /^---/{if(u && buf!=\"\"){print \"===== USER =====\"buf; buf=\"\"}}' \"2026-09-08_234727_ide_selectionThe-user-selected-the-lin_2731f840.md\" | grep -v \"^\\s*$\" | head -300",
  "description": "Extract all user messages from history"
}
```

</details>


---

## User <sup>2026/9/8 23:55:33</sup>

<details>
<summary>Tool Result</summary>

```
===== USER =====
<ide_selection>The user selected the lines 72 to 72 from c:\prefill_only_main\vllm_tywww\vllm\v1\lwd_control\control_scheduler\lwd_edge_scheduler.py:
def add_request
This may or may not be related to the current task.</ide_selection>
先阅读一下C:\prefill_only_main\1.md
---
===== USER =====
<summary>Tool Result</summary>
1	梳理prefill_only边侧控制面（enginecore进程）功能点
2	
3	已工作 59 秒
4	prefill_only 边侧控制面功能点梳理（EngineCore 进程内）
5	边侧 EngineCore 进程在 prefill_only 模式下没有 compute scheduler，控制面职责由三件套承担：EdgePrefillDispatcher（调度/记账）+ PrefillOnlyControlPublisher / CloudOutputReceiver（两个 ZMQ 面）+ 装配胶水（prefill_only_engine_core.py）。装配入口是 setup_prefill_only_engine_core（由 patch_engine_core.py:148 的 init 分支单点调用，仅 is_edge_node=True 时生效）。
6	
7	1. 装配阶段（setup_prefill_only_engine_core）
8	功能点	说明
9	端口布局	PRE_OUT = pre_out_port + dp_rank*2（bind，云 connect）；POST_OUT = post_out_port + dp_rank*2（bind，云 connect）。双面只用 master_addr，无 TCPStore 发现
10	POST_OUT bind 覆盖	VLLM_ASCEND_PREFILL_ONLY_CLOUD_ADDR 可覆盖 wildcard 端点（联调选网卡
...(truncated)
```

</details>


---

## Assistant <sup>2026/9/8 23:55:38</sup>


---

## Assistant <sup>2026/9/8 23:55:38</sup>

awk 捕获到了工具结果。让我使用 Python 以结构化方式进行解析。


---

## Assistant <sup>2026/9/8 23:55:41</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"C:/prefill_only_main/.claude-code-history\" && python -c \"\nimport re\nwith open('2026-09-08_234727_ide_selectionThe-user-selected-the-lin_2731f840.md', encoding='utf-8') as f:\n    lines = f.readlines()\n\nmsgs = []\ncur_role, cur_time, buf = None, None, []\nfor ln in lines:\n    m = re.match(r'^## (User|Assistant) <sup>(.*?)</sup>', ln)\n    if m:\n        if cur_role:\n            msgs.append((cur_role, cur_time, ''.join(buf)))\n        cur_role, cur_time, buf = m.group(1), m.group(2), []\n    else:\n        buf.append(ln)\nif cur_role:\n    msgs.append((cur_role, cur_time, ''.join(buf)))\n\n# real user prompts: not tool results, not ide noise\nfor i, (role, t, body) in enumerate(msgs):\n    if role == 'User':\n        body_l = body.strip()\n        if body_l.startswith('<summary>Tool Result') or body_l.startswith('\\`\\`\\`'):\n            continue\n        if body_l == '---' or body_l == '':\n            continue\n        print(f'@@@ USER [{t}] @@@')\n        print(body_l[:1500])\n        print()\n\" 2>&1 | head -400",
  "description": "Parse real user prompts from history"
}
```

</details>


---

## User <sup>2026/9/8 23:55:44</sup>

<details>
<summary>Tool Result</summary>

```
Traceback (most recent call last):
  File "<string>", line 28, in <module>
    print(body_l[:1500])
    ~~~~~^^^^^^^^^^^^^^^
UnicodeEncodeError: 'gbk' codec can't encode character '\u2705' in position 133: illegal multibyte sequence
@@@ USER [2026/9/8 17:00:31] @@@
<ide_selection>The user selected the lines 72 to 72 from c:\prefill_only_main\vllm_tywww\vllm\v1\lwd_control\control_scheduler\lwd_edge_scheduler.py:
def add_request

This may or may not be related to the current task.</ide_selection>

���Ķ�һ��C:\prefill_only_main\1.md


---

@@@ USER [2026/9/8 17:00:35] @@@
<details>
<summary>Tool Result</summary>

```
1	����prefill_only�߲�����棨enginecore���̣����ܵ�
2	
3	�ѹ��� 59 ��
4	prefill_only �߲�����湦�ܵ�������EngineCore �����ڣ�
5	�߲� EngineCore ������ prefill_only ģʽ��û�� compute scheduler��������ְ���������׳е���EdgePrefillDispatcher������/���ˣ�+ PrefillOnlyControlPublisher / CloudOutputReceiver������ ZMQ �棩+ װ�佺ˮ��prefill_only_engine_core.py����װ������� setup_pr
...(truncated)
```

</details>


---

## Assistant <sup>2026/9/8 23:55:49</sup>


---

## Assistant <sup>2026/9/8 23:55:53</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"C:/prefill_only_main/.claude-code-history\" && python -X utf8 -c \"\nimport re, io, sys\nsys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')\nwith open('2026-09-08_234727_ide_selectionThe-user-selected-the-lin_2731f840.md', encoding='utf-8') as f:\n    lines = f.readlines()\n\nmsgs = []\ncur_role, cur_time, buf = None, None, []\nfor ln in lines:\n    m = re.match(r'^## (User|Assistant) <sup>(.*?)</sup>', ln)\n    if m:\n        if cur_role:\n            msgs.append((cur_role, cur_time, ''.join(buf)))\n        cur_role, cur_time, buf = m.group(1), m.group(2), []\n    else:\n        buf.append(ln)\nif cur_role:\n    msgs.append((cur_role, cur_time, ''.join(buf)))\n\nout = []\nfor i, (role, t, body) in enumerate(msgs):\n    b = body.strip()\n    if b.startswith('<details>'):\n        continue  # tool call/result\n    if b in ('', '---'):\n        continue\n    if role == 'User':\n        out.append(f'@@@ USER [{t}] @@@\\n{b[:2000]}')\n    else:\n        # assistant plain text\n        if len(b) > 30:\n            out.append(f'--- ASSISTANT [{t}] ---\\n{b[:2500]}')\nwith open('_extract.txt', 'w', encoding='utf-8') as f:\n    f.write('\\n\\n'.join(out))\nprint('wrote', len(out), 'entries')\n\"",
  "description": "Extract real user/assistant text to file"
}
```

</details>


---

## User <sup>2026/9/8 23:59:26</sup>

<details>
<summary>Tool Result</summary>

```
wrote 63 entries
Shell cwd was reset to c:\prefill_only_wangwei
```

</details>


---

## Assistant <sup>2026/9/8 23:59:28</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "C:\\prefill_only_main\\.claude-code-history\\_extract.txt"
}
```

</details>


---

## User <sup>2026/9/8 23:59:28</sup>

<details>
<summary>Tool Result</summary>

```
1	@@@ USER [2026/9/8 17:00:31] @@@
2	<ide_selection>The user selected the lines 72 to 72 from c:\prefill_only_main\vllm_tywww\vllm\v1\lwd_control\control_scheduler\lwd_edge_scheduler.py:
3	def add_request
4	
5	This may or may not be related to the current task.</ide_selection>
6	
7	先阅读一下C:\prefill_only_main\1.md
8	
9	
10	---
11	
12	--- ASSISTANT [2026/9/8 17:00:55] ---
13	已读完 [1.md](1.md)。这是一份移植工作的完整决策记录，脉络如下：
14	
15	## 背景与目标
16	- 将 prefill_only 边侧控制面（EngineCore 进程内的 EdgePrefillDispatcher + 两个 ZMQ 面 + 装配胶水）从源仓移植到 [vllm_tywww/vllm/v1/lwd_control](vllm_tywww/vllm/v1/lwd_control/)
17	- 关键变化：控制面与 Ascend 解耦、代码进 vllm 主仓、**继承原生类**（AsyncScheduler / EngineCore）而非 monkey-patch
18	- 新架构采用**单向控制面**（§9.1）：边→云只有 PRE_OUT（元数据/notify/abort），**云→边零应答**——与 C:\prefill_only\current 的双向版（含 POST_OUT 回程、consumed 水位、credit 门控）不同，这些都明确不移植
19	
20	## 已完成的工作
21	1. **骨架自洽化**：统一为 `lwd_control/{control_communication, control_scheduler}` 目录结构，补齐 5 个缺失 stub 文件（lwd_message / lwd_config / lwd_mode / lwd_ports / lwd_diagn
...(truncated)
```

</details>


---

## Assistant <sup>2026/9/8 23:59:34</sup>


---

## Assistant <sup>2026/9/8 23:59:35</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "rm \"C:/prefill_only_main/.claude-code-history/_extract.txt\"; cd /c/prefill_only_wangwei/vllm_tywww 2>/dev/null && git log --oneline -8 2>/dev/null; git status --short 2>/dev/null | head -20",
  "description": "Clean temp file and check wangwei repo state"
}
```

</details>


---

## User <sup>2026/9/9 00:07:13</sup>

<details>
<summary>Tool Result</summary>

```
cd295b1b8 feat(lwd): 实现边缘侧LWD预填充调度的请求管理流程
43cf06c18 fix(lwd edge scheduler): reset output placeholders when reverting request state
d936ccbb2 refactor(lwd_edge_scheduler): 调整步末对账逻辑，基于本步排程集而非执行量
cfa838760 feat(lwd): add external block hash support for prefix cache
46c39886e refactor: 完成云侧step_wrapper剥离与lwd_cloud_core删除
b39a98f1f refactor: 完成控制面通信层组合化重构与命名统一
785f829de refactor(lwd_control): 收编prefill-only控制面与通信到vllm/v1/lwd_control
1186069db refactor: 将Lwd控制面代码从根lwd目录迁移到control_scheduler子目录
```

</details>


---

## Assistant <sup>2026/9/9 00:07:24</sup>


---

## Assistant <sup>2026/9/9 00:07:25</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "git show --stat cd295b1b8 | head -15 && echo \"---STATUS---\" && git status --short | head",
  "description": "Inspect latest commit contents and working tree"
}
```

</details>


---

## User <sup>2026/9/9 00:07:26</sup>

<details>
<summary>Tool Result</summary>

```
commit cd295b1b88e1374d9e4633fcf85ea59b1c6b1307
Author: theroadtofree <hw.wangtongtong@huawei.com>
Date:   Tue Sep 8 23:48:22 2026 +0800

    feat(lwd): 实现边缘侧LWD预填充调度的请求管理流程
    
    新增LwdEdgeScheduler的请求校验、添加和中止接口，在EngineCore中接入LWD边缘调度路径，实现预填充模式下的控制面边界校验和请求转发逻辑，符合§9.12、§14.4等设计规范

 vllm/v1/engine/core.py                             | 19 +++++++
 .../control_edge_scheduler/lwd_edge_scheduler.py   | 58 ++++++++++++++++++++++
 2 files changed, 77 insertions(+)
---STATUS---
```

</details>


---

## Assistant <sup>2026/9/9 00:07:29</sup>


---

## Assistant <sup>2026/9/9 00:07:39</sup>

已读完整个历史（2026/9/8 17:00–23:47，567 条消息的会话记录），并核对了 wangwei 仓当前状态。整体脉络如下：

## 一、prefill_only 是什么

**边云分离架构**：边侧（小算力）只跑 embedding（prefill），embeds 经 HCCL 上云；云侧用 `enable-prompt-embeds` 混合模式跑真正的推理。控制面是两个 ZMQ 通道（源仓为 PRE_OUT 边→云 / POST_OUT 云→边，`(8B seq, pickle)` 帧）。

**移植主线**：从源仓 `C:\prefill_only\current`（vllm-ascend，monkey-patch 形态）→ `C:\prefill_only_main\vllm_tywww`（继承原生类 + 单向控制面重设计）→ **`C:\prefill_only_wangwei\vllm_tywww`**（当前活跃仓，独立目录 `control_edge_scheduler/`，另一套更完整的平行演进）。

新架构关键裁定：复用原生 `schedule()` 的分块（而非源仓自造 chunk plan）、publish 队满返 False 不丢消息（源仓是丢弃+自认 desync）、notify offset 需回退本步排程量（fork 在 schedule() 尾部乐观推进 `num_computed`）、`num_output_placeholders` 归零防幻影 decode token。

## 二、该会话完成的工作

1. **旁路分析**（未落码）：pickle RCE 风险→msgpack 方案；云侧确需全量 token ids（块哈希/前缀缓存）；len-only 传输记台账为开放项
2. **旧树（prefill_only_main）三步全实现**：step_with_batch_queue 空转主循环 + lwd_edge_notify → schedule() 纯 prefill 覆写 + update_progress（48+45 项 AST 隔离冒烟全过）
3. **同步到 wangwei 仓**——识别出 6 个缺口，**6 个全部完成**：

| # | 内容 | 状态 |
|---|---|---|
| 5 | update_progress 对账基准改 `_lwd_last_scheduled`（队满回退） | ✅ 已提交 `d936ccbb2` |
| 6 | 回退分支占位符归零 | ✅ 已提交 `43cf06c18` |
| 1 | core.py add_request 守卫（校验+LwdRequestNotify 出口） | ✅ 已提交 `cd295b1b8` |
| 2 | abort_requests 守卫（LwdAbortNotify 出口） | ✅ 同上 |
| 3 | step() 单批路径守卫 | ✅ 同上 |
| 4 | 五项入口校验 _lwd_validate_request | ✅ 同上 |

## 三、当前同步状态

wangwei 仓工作区**干净**，历史中所有改动均已提交（最后一笔 `cd295b1b8`，+77 行，正是会话末尾“未提交”的第 1-4 点）。两边边侧功能已对齐。

**下一步候选**（历史中提及、尚未开始）：
- 第四步：数据面——embeds isend + 回执对接 `lwd_edge_update_progress` 的 executed 来源（当前 executed 来源接缝按 §9.12 待重定义）
- 台账开放项：len-only 传输（前提：云侧前缀缓存策略确定 + wire-off 调试模式退役）、msgpack 替换 pickle、帧级压缩


---

## User <sup>2026/9/9 00:09:31</sup>

现在只搞边侧控制的移植，对比C:\prefill_only\current的边侧控制面实现（在asend仓），参考C:\prefill_only_main\1.md，是否还存在差距


---

## Assistant <sup>2026/9/9 00:09:36</sup>


---

## Assistant <sup>2026/9/9 00:09:36</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "C:\\prefill_only_main\\1.md"
}
```

</details>


---

## User <sup>2026/9/9 00:09:36</sup>

<details>
<summary>Tool Result</summary>

```
1	梳理prefill_only边侧控制面（enginecore进程）功能点
2	
3	已工作 59 秒
4	prefill_only 边侧控制面功能点梳理（EngineCore 进程内）
5	边侧 EngineCore 进程在 prefill_only 模式下没有 compute scheduler，控制面职责由三件套承担：EdgePrefillDispatcher（调度/记账）+ PrefillOnlyControlPublisher / CloudOutputReceiver（两个 ZMQ 面）+ 装配胶水（prefill_only_engine_core.py）。装配入口是 setup_prefill_only_engine_core（由 patch_engine_core.py:148 的 init 分支单点调用，仅 is_edge_node=True 时生效）。
6	
7	1. 装配阶段（setup_prefill_only_engine_core）
8	功能点	说明
9	端口布局	PRE_OUT = pre_out_port + dp_rank*2（bind，云 connect）；POST_OUT = post_out_port + dp_rank*2（bind，云 connect）。双面只用 master_addr，无 TCPStore 发现
10	POST_OUT bind 覆盖	VLLM_ASCEND_PREFILL_ONLY_CLOUD_ADDR 可覆盖 wildcard 端点（联调选网卡）
11	dispatcher 参数	chunk_size = max_num_batched_tokens（与云 1:1 对齐前提）、max_in_flight = max_num_seqs、credit_per_req（env VLLM_ASCEND_PREFILL_ONLY_CREDIT_PER_REQ，默认 3）
12	step 替换	实例属性 step_fn = _prefill_only_step，遮蔽 fork 的 _patched_step/_patched_step_with_batch_queue；_process_engine_step、shutdown 同样换成 prefill_only 专用版
13	另有 install_prefil
...(truncated)
```

</details>


---

## Assistant <sup>2026/9/9 00:09:36</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "ls -R /c/prefill_only_wangwei/vllm_tywww/vllm/v1/lwd_control | head -40",
  "description": "List wangwei lwd_control tree"
}
```

</details>


---

## User <sup>2026/9/9 00:09:37</sup>

<details>
<summary>Tool Result</summary>

```
/c/prefill_only_wangwei/vllm_tywww/vllm/v1/lwd_control:
__init__.py
control_cloud_scheduler
control_communication
control_edge_scheduler
control_scheduler

/c/prefill_only_wangwei/vllm_tywww/vllm/v1/lwd_control/control_cloud_scheduler:
lwd_cloud_assemble.py
lwd_cloud_phase_scheduler.py

/c/prefill_only_wangwei/vllm_tywww/vllm/v1/lwd_control/control_communication:
__init__.py
lwd_control_communicator.py
lwd_control_publisher.py
lwd_control_subscriber.py
lwd_notify.py

/c/prefill_only_wangwei/vllm_tywww/vllm/v1/lwd_control/control_edge_scheduler:
__pycache__
lwd_edge_assemble.py
lwd_edge_core.py
lwd_edge_scheduler.py
lwd_step_core.py

/c/prefill_only_wangwei/vllm_tywww/vllm/v1/lwd_control/control_edge_scheduler/__pycache__:
lwd_edge_scheduler.cpython-314.pyc

/c/prefill_only_wangwei/vllm_tywww/vllm/v1/lwd_control/control_scheduler:
__init__.py
```

</details>


---

## Assistant <sup>2026/9/9 00:09:44</sup>


---

## Assistant <sup>2026/9/9 00:09:44</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "C:\\prefill_only_wangwei\\vllm_tywww\\vllm\\v1\\lwd_control\\control_edge_scheduler\\lwd_edge_scheduler.py"
}
```

</details>


---

## User <sup>2026/9/9 00:09:44</sup>

<details>
<summary>Tool Result</summary>

```
1	"""边侧调度器:纯 prefill 调度 + 控制面发布(notify/abort/seqno,§9.10/§9.12)。
2	
3	原生 AsyncScheduler 的两个前提在边侧不成立:prompt 算完会转 decode、
4	请求只能由模型输出终结;边侧无本地解码,故需专用调度器。
5	
6	纯 prefill 的实现依据:schedule() 全量复用原生——边侧请求从不产生
7	输出 token(num_tokens_with_spec 恒等于 num_prompt_tokens),且 prompt
8	嵌入完成的当步即被 lwd_edge_update_progress 本地终结,原生 RUNNING 段
9	每步只会调度剩余 prefill,decode 分支不可达。
10	"""
11	
12	from __future__ import annotations
13	
14	import time
15	from typing import TYPE_CHECKING
16	
17	from vllm.logger import init_logger
18	from vllm.v1.core.sched.async_scheduler import AsyncScheduler
19	from vllm.v1.core.sched.output import SchedulerOutput
20	from vllm.v1.lwd_control.control_communication.lwd_notify import (
21	    LwdAbortNotify,
22	    LwdRangeNotify,
23	    LwdRequestNotify,
24	)
25	from vllm.v1.request import RequestStatus
26	
27	if TYPE_CHECKING:
28	    from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
29	        LwdControlPublisher,
30	    )
31	    from vllm.v1.request import Re
...(truncated)
```

</details>


---

## Assistant <sup>2026/9/9 00:09:44</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "C:\\prefill_only_wangwei\\vllm_tywww\\vllm\\v1\\lwd_control\\control_edge_scheduler\\lwd_edge_core.py"
}
```

</details>


---

## User <sup>2026/9/9 00:09:44</sup>

<details>
<summary>Tool Result</summary>

```
1	"""边侧执行类:唯一接口 step_with_batch_queue,由 core.py 守卫委托(§9.8/§9.12)。
2	
3	step 语义 = 原生步进的控制面替代:调度器出分块决策并发预告 ->
4	原生路径提交执行 -> 步末推进进度。数据面动作不在本层(§9.12)。
5	"""
6	
7	from __future__ import annotations
8	
9	from typing import TYPE_CHECKING
10	
11	from vllm.v1.lwd_control.control_edge_scheduler.lwd_step_core import (
12	    LwdLog,
13	    LwdStepCore,
14	    LwdStepSettings,
15	)
16	
17	if TYPE_CHECKING:
18	    from vllm.v1.lwd_control.control_communication.lwd_control_communicator import (
19	        EngineCoreOutputs,
20	    )
21	    from vllm.v1.lwd_control.control_edge_scheduler.lwd_step_core import LwdEnginePort
22	
23	
24	class LwdEdgeCore(LwdStepCore):
25	    """边侧 prefill_only core:只发不收(§9.1),输出恒为 (None, 是否有工作)。"""
26	
27	    def __init__(self, engine_port: LwdEnginePort, settings: LwdStepSettings) -> None:
28	        self._engine_port = engine_port
29	        self._log = LwdLog(settings.debug)
30	
31	    def step_with_batch_queue(
32	        self,
33	    ) -> tuple[dict[int, EngineCoreO
...(truncated)
```

</details>


---

## Assistant <sup>2026/9/9 00:09:58</sup>


---

## Assistant <sup>2026/9/9 00:09:58</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "C:\\prefill_only_wangwei\\vllm_tywww\\vllm\\v1\\lwd_control\\control_edge_scheduler\\lwd_edge_assemble.py"
}
```

</details>


---

## User <sup>2026/9/9 00:09:58</sup>

<details>
<summary>Tool Result</summary>

```
1	"""边侧 L3 装配:EngineCore.__init__ 守卫调用的单点开关(§8.2/§9.8)。
2	
3	单向语义:无云结果/水位 drain,步进逻辑全部收编进
4	LwdEdgeCore.step_with_batch_queue;本文件只负责装配与生命周期,
5	原 6 个 monkey-patch 由 core.py in-tree 守卫分支替代。
6	
7	本文件同时承载两侧共享的配置支撑(§10.3 折入):LwdConfig
8	(env/additional_config 唯一解析入口)与 is_lwd_prefill_only
9	(模式判定唯一实现);云侧装配经 import 复用,内核模块收 plain 值。
10	"""
11	
12	from __future__ import annotations
13	
14	import os
15	from dataclasses import dataclass
16	
17	from vllm.logger import init_logger
18	from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
19	from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
20	    LWD_PUBLISH_QUEUE_MAX,
21	    LwdControlPublisher,
22	)
23	from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_core import LwdEdgeCore
24	from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
25	    LwdEdgeScheduler,
26	)
27	from vllm.v1.lwd_control.control_edge_scheduler.lwd_step_core import LwdStepSettings
28	
29	logger = init_logger(_
...(truncated)
```

</details>


---

## Assistant <sup>2026/9/9 00:09:58</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "C:\\prefill_only_wangwei\\vllm_tywww\\vllm\\v1\\lwd_control\\control_edge_scheduler\\lwd_step_core.py"
}
```

</details>


---

## User <sup>2026/9/9 00:09:58</sup>

<details>
<summary>Tool Result</summary>

```
1	"""Lwd 执行类抽象与端口协议:边侧对 EngineCore 的唯一扩展点是 step 接口(§9.8)。
2	
3	core.py 的 step_with_batch_queue 顶部守卫:
4	    if self.step_wrapper is not None:
5	        return self.step_wrapper.step_with_batch_queue()
6	装配期(L3)把 prefill_only 专用 core 赋给 EngineCore.step_wrapper;
7	非 PO 路径 step_wrapper 恒为 None,上游行为不变。云侧 §10.12 起不走
8	step_wrapper(原生步体 + 桥线程),本抽象仅服务边侧。
9	
10	本文件同时承载内核共享的纯支撑(§10.3 折入):LwdEnginePort
11	端口协议、LwdStepSettings plain 值载体、LwdLog。
12	"""
13	
14	from __future__ import annotations
15	
16	from abc import ABC, abstractmethod
17	from dataclasses import dataclass
18	from typing import TYPE_CHECKING, Protocol
19	
20	from vllm.logger import init_logger
21	
22	if TYPE_CHECKING:
23	    from vllm.v1.lwd_control.control_communication.lwd_control_communicator import (
24	        EngineCoreOutputs,
25	    )
26	
27	logger = init_logger(__name__)
28	
29	
30	@dataclass(frozen=True)
31	class LwdStepSettings:
32	    """步进编排 plain 值;装配期由 LwdConfig 派生,内核不解析配置(§10.3)。"""
33	
34	    debug: bool = False
35	
...(truncated)
```

</details>


---

## Assistant <sup>2026/9/9 00:10:13</sup>


---

## Assistant <sup>2026/9/9 00:10:13</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "grep -n -B2 -A8 \"lwd\" vllm/v1/engine/core.py | head -120",
  "description": "Check core.py LWD guards in wangwei"
}
```

</details>


---

## User <sup>2026/9/9 00:10:14</sup>

<details>
<summary>Tool Result</summary>

```
grep: vllm/v1/engine/core.py: No such file or directory
```

</details>


---

## Assistant <sup>2026/9/9 00:10:17</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "grep -n \"lwd\" /c/prefill_only_wangwei/vllm_tywww/vllm/v1/engine/core.py | head -30",
  "description": "Find LWD guards in wangwei core.py"
}
```

</details>


---

## User <sup>2026/9/9 00:10:18</sup>

<details>
<summary>Tool Result</summary>

```
239:        from vllm.v1.lwd_control import lwd_try_assemble
241:        lwd_try_assemble(self)
361:            self.scheduler.lwd_edge_add_request(request)  # type: ignore[attr-defined]
402:            self.scheduler.lwd_edge_abort(request_ids)  # type: ignore[attr-defined]
644:            from vllm.v1.lwd_control import lwd_shutdown
646:            lwd_shutdown(self)
```

</details>


---

## Assistant <sup>2026/9/9 00:10:22</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "sed -n '230,250p;350,370p;395,415p;635,655p' /c/prefill_only_wangwei/vllm_tywww/vllm/v1/engine/core.py",
  "description": "Read LWD guard bodies in core.py"
}
```

</details>


---

## User <sup>2026/9/9 00:10:22</sup>

<details>
<summary>Tool Result</summary>

```
# Reduces pause times of oldest generation collections.
        freeze_gc_heap()
        # If enable, attach GC debugger after static variable freeze.
        maybe_attach_gc_debug_callback()
        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()

        # Lwd prefill-only assembly point: no-op unless the mode is enabled.
        from vllm.v1.lwd_control import lwd_try_assemble

        lwd_try_assemble(self)

    @instrument(span_name="Prepare model")
    def _initialize_kv_caches(self, vllm_config: VllmConfig) -> KVCacheConfig:
        start = time.time()

        # register all kvcache specs in enginecore process.
        register_all_kvcache_specs(vllm_config)

        # Get all kv cache needed by the model
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        # Lwd 
...(truncated)
```

</details>


---

## Assistant <sup>2026/9/9 00:10:26</sup>


---

## Assistant <sup>2026/9/9 00:10:26</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "grep -n \"step_wrapper\" /c/prefill_only_wangwei/vllm_tywww/vllm/v1/engine/core.py",
  "description": "Find step_wrapper guards"
}
```

</details>


---

## User <sup>2026/9/9 00:10:27</sup>

<details>
<summary>Tool Result</summary>

```
122:        self.step_wrapper = None
360:        if self.step_wrapper is not None:
401:        if self.step_wrapper is not None:
474:        if self.step_wrapper is not None:
475:            return self.step_wrapper.step_with_batch_queue()
528:        if self.step_wrapper is not None:
529:            return self.step_wrapper.step_with_batch_queue()
643:        if self.step_wrapper is not None:
```

</details>


---

## Assistant <sup>2026/9/9 00:10:32</sup>


---

## Assistant <sup>2026/9/9 00:10:32</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "grep -n \"def \\|class \\|PO-\\|VLLM_ASCEND_PREFILL_ONLY\" /c/prefill_only_current/vllm-ascend_qxxxw/vllm_ascend/edge_cloud/edge_prefill_dispatcher.py | head -60",
  "description": "List all functions/features in source dispatcher"
}
```

</details>


---

## User <sup>2026/9/9 00:10:33</sup>

<details>
<summary>Tool Result</summary>

```
grep: /c/prefill_only_current/vllm-ascend_qxxxw/vllm_ascend/edge_cloud/edge_prefill_dispatcher.py: No such file or directory
```

</details>


---

## Assistant <sup>2026/9/9 00:10:36</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "ls /c/prefill_only/ 2>/dev/null; ls /c/prefill_only/current/vllm-ascend_qxxxw/vllm_ascend/edge_cloud/ 2>/dev/null",
  "description": "Locate source repo dirs"
}
```

</details>


---

## User <sup>2026/9/9 00:10:37</sup>

<details>
<summary>Tool Result</summary>

```
Untitled-1.txt
current
origin
__init__.py
__pycache__
active_engine_core.py
edge_prefill_dispatcher.py
prefill_only_channel.py
prefill_only_engine_core.py
pure_phase_scheduler.py
```

</details>


---

## Assistant <sup>2026/9/9 00:10:40</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "grep -n \"    def \\|^def \\|^class \\|PO-\\|PO_\" /c/prefill_only/current/vllm-ascend_qxxxw/vllm_ascend/edge_cloud/edge_prefill_dispatcher.py",
  "description": "List source dispatcher functions and markers"
}
```

</details>


---

## User <sup>2026/9/9 00:10:41</sup>

<details>
<summary>Tool Result</summary>

```
50:class PrefillStatus(enum.Enum):
65:class EdgePrefillRequestState:
91:    def all_chunks_dispatched(self) -> bool:
95:class EdgeEmbedChunkNotify(msgspec.Struct, gc=False):
114:class EdgePrefillAbort(msgspec.Struct, gc=False):
121:class EdgeEmbedConsumed:
140:class EdgeEmbedChunkAck:
148:class EdgeEmbedBatchAck:
154:class EdgePrefillDispatcher:
178:    def __init__(
193:        # [PO-WIRE-OFF] kills notify/isend/irecv entirely (default ON,
234:    def has_requests(self) -> bool:
238:    def add_request(self, request: EngineCoreRequest) -> None:
251:            "[PO-TTFT][edge] event=meta_pub rid=%s ts=%.6f",
265:    def abort_requests(self, request_ids: list[str]) -> None:
297:    def on_request_outputs(self, outputs: EngineCoreOutputs) -> None:
308:        # [PO-TTFT] first token arrival (edge clock): first time this
319:                        "[PO-TTFT][edge] event=first_token rid=%s "
347:    def on_embed_consumed(self, request_id: str, upto: int) -> None:
362:    def try_dispatch
...(truncated)
```

</details>


---

## Assistant <sup>2026/9/9 00:11:05</sup>


---

## Assistant <sup>2026/9/9 00:11:06</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "C:\\prefill_only_wangwei\\vllm_tywww\\vllm\\v1\\lwd_control\\control_communication\\lwd_notify.py"
}
```

</details>


---

## User <sup>2026/9/9 00:11:06</sup>

<details>
<summary>Tool Result</summary>

```
1	"""Lwd 控制面通知(notify):仅依赖 msgspec,不携带 tensor(§2.7/§9,控制面专用,§9.12)。
2	
3	数据面不在本目录范围:notify 只承载调度决策预告,张量传输与落位
4	由数据面经既有接缝对接。上游输出类型经本模块 re-export,
5	其余内核文件不得直接 import vllm.v1.engine(白名单,§7.2)。
6	"""
7	
8	from __future__ import annotations
9	
10	from typing import TYPE_CHECKING, Union
11	
12	import msgspec
13	
14	if TYPE_CHECKING:
15	    # 上游输出类型 re-export:step 载体的返回注解共用,避免各文件直连上游
16	    from vllm.v1.engine import EngineCoreOutputs  # noqa: F401
17	
18	
19	class LwdRangeNotify(msgspec.Struct, gc=False, tag=True):
20	    """边->云调度范围预告(PRE_OUT);发布先于任何数据面动作。
21	
22	    offset/num_tokens 直接取自原生 SchedulerOutput 的调度决策
23	    (chunked prefill 的 num_computed/num_scheduled);云侧登记
24	    seqno->request 映射,数据面落位后由此对接接收。重复预告按
25	    (request_id, offset) 幂等登记(边侧队满重试天然产生重复)。
26	    """
27	
28	    request_id: str
29	    offset: int
30	    num_tokens: int
31	    seqno: int
32	
33	
34	class LwdRequestNotify(msgspec.Struct, gc=False, tag=True):
35	    """边->云请求预告(PRE_OUT);云侧据此建调度请求并驱动本地生命周期。
36	
37	    结果不回
...(truncated)
```

</details>


---

## Assistant <sup>2026/9/9 00:11:06</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "grep -n \"has_requests\\|has_unfinished\" /c/prefill_only_wangwei/vllm_tywww/vllm/v1/core/sched/scheduler.py | head; grep -n \"def has_requests\" /c/prefill_only_wangwei/vllm_tywww/vllm/v1/core/sched/async_scheduler.py",
  "description": "Check has_requests existence"
}
```

</details>


---

## User <sup>2026/9/9 00:11:07</sup>

<details>
<summary>Tool Result</summary>

```
(Bash completed with no output)
```

</details>


---

## Assistant <sup>2026/9/9 00:11:10</sup>


---

## Assistant <sup>2026/9/9 00:11:11</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "grep -rn \"def has_requests\\|def has_unfinished\" /c/prefill_only_wangwei/vllm_tywww/vllm/v1/core/sched/ | head; echo ---; grep -n \"has_requests\" /c/prefill_only_wangwei/vllm_tywww/vllm/v1/lwd_control/control_edge_scheduler/*.py",
  "description": "Verify has_requests method exists anywhere"
}
```

</details>


---

## User <sup>2026/9/9 00:11:12</sup>

<details>
<summary>Tool Result</summary>

```
/c/prefill_only_wangwei/vllm_tywww/vllm/v1/core/sched/interface.py:165:    def has_unfinished_requests(self) -> bool:
/c/prefill_only_wangwei/vllm_tywww/vllm/v1/core/sched/interface.py:185:    def has_requests(self) -> bool:
---
/c/prefill_only_wangwei/vllm_tywww/vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_core.py:36:        if self._engine_port.lwd_scheduler().has_requests():
```

</details>


---

## Assistant <sup>2026/9/9 00:11:17</sup>


---

## Assistant <sup>2026/9/9 00:11:18</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "grep -n \"def \\|sleep\\|_dropped\\|zombie\" /c/prefill_only/current/vllm-ascend_qxxxw/vllm_ascend/edge_cloud/prefill_only_engine_core.py | head -40; echo ====; grep -n \"def \\|queue.Full\\|_dropped\\|seq\" /c/prefill_only_wangwei/vllm_tywww/vllm/v1/lwd_control/control_communication/lwd_control_publisher.py | head -30",
  "description": "Compare source engine core and wangwei publisher"
}
```

</details>


---

## User <sup>2026/9/9 00:11:19</sup>

<details>
<summary>Tool Result</summary>

```
47:def setup_prefill_only_engine_core(engine_core: _core.EngineCore) -> None:
124:def _post_out_bind_endpoint(post_out_port: int) -> str:
145:def _prefill_only_step(
190:def _prefill_only_process_engine_step(self: _core.EngineCoreProc) -> bool:
200:        _time.sleep(0.001)
204:def _prefill_only_shutdown(self: _core.EngineCore) -> None:
219:def install_prefill_only_engine_core_patches() -> None:
236:    def _preprocess_add_request(self, request):
245:    def _add_request(self, request, request_wave=0):
253:    def _abort_requests(self, request_ids):
261:    def _has_work(self) -> bool:
====
37:    def __init__(
53:    def publish(self, msg: LwdNotify) -> bool:
57:        except queue.Full:
61:    def shutdown(self) -> None:
66:        with contextlib.suppress(queue.Full):
75:    def _send_thread(self) -> None:
```

</details>


---

## Assistant <sup>2026/9/9 00:11:24</sup>


---

## Assistant <sup>2026/9/9 00:11:24</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "sed -n '145,215p' /c/prefill_only/current/vllm-ascend_qxxxw/vllm_ascend/edge_cloud/prefill_only_engine_core.py",
  "description": "Read source step loop body"
}
```

</details>


---

## User <sup>2026/9/9 00:11:25</sup>

<details>
<summary>Tool Result</summary>

```
def _prefill_only_step(
    self: _core.EngineCore,
) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
    """Drain cloud results, poll embed acks, dispatch one embed chunk.

    Returns the per-client output dict (upstream ``step_fn`` contract)
    and whether this step did engine-side work.
    """
    dispatcher = self._prefill_dispatcher

    # 1) Cloud results: finish cleanup first (releases admission), then
    # regroup for the front-end output queue.
    outputs: dict[int, EngineCoreOutputs] | None = None
    drained = self._prefill_receiver.drain()
    for _client_idx, eco in drained:
        dispatcher.on_request_outputs(eco)
    if drained:
        outputs = {}
        for client_idx, eco in drained:
            existing = outputs.get(client_idx)
            if existing is None:
                outputs[client_idx] = eco
            else:
                existing.outputs.extend(eco.outputs)
                if eco.finished_requests:
                    if existing.finished
...(truncated)
```

</details>


---

## Assistant <sup>2026/9/9 00:11:57</sup>


---

## Assistant <sup>2026/9/9 00:12:15</sup>

对比完成。我把 1.md 的源仓功能清单逐项对照了 wangwei 当前 [control_edge_scheduler](vllm_tywww/vllm/v1/lwd_control/control_edge_scheduler/)、core.py 守卫（7 处齐全）和通信层，结论如下：

## 一、控制面核心功能：已对齐（6 个同步点 + 此前落位）

| 1.md 源仓功能点 | wangwei 现状 |
|---|---|
| 单点装配 + 模式判定收敛 | ✅ `lwd_try_assemble`（还多了 kv_connector 降级检查） |
| add_request 五项校验 | ✅ `_lwd_validate_request`，语义一致 |
| 元数据先行（§14.4 云视图领先） | ✅ `lwd_edge_notify_request` 先于本地入队 |
| notify 先于 execute + seqno 显式携带 | ✅ `lwd_edge_notify` + `_lwd_edge_next_seqno` |
| offset/seqno 配对契约 | ✅ offset 回退本步排程量（fork 语义修正） |
| abort 出口（含未知 rid 必达） | ✅ core.py:401 守卫 + `lwd_edge_abort` |
| shutdown 先关通道再 teardown | ✅ core.py:643 |
| publisher 后台线程 + 有界队列 + 队满 False | ✅ `LwdControlPublisher`（且 pickle→msgspec，消了 RCE 面） |
| 队满对账回退 + 占位符归零 | ✅ 第 5/6 点（`d936ccbb2`、`43cf06c18`） |

## 二、有意不移植（设计裁定，不算差距）

POST_OUT 回程及输出合并、consumed 水位 + credit 门控、全局 outstanding 门、`_waiting` 手搓准入（原生 waiting 替代）、EDGE_DONE 占坑（本地终结替代）、空转 sleep 1ms（has_work 原生阻塞）、轮转公平（原生调度替代）。

## 三、真实差距（按重要度）

**1. 数据面接缝是同步占位——控制面的“收 ack”半边没有真实来源**（最大差距）
[lwd_edge_assemble.py:136-138](vllm_tywww/vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_assemble.py#L136-L138) 的 `lwd_execute_model` 直接 `executor.execute_model(scheduler_output).result()`——跑的是**原生模型前向**，不是源仓的 EDGE_EMBED 嵌入批；`executed = 调度量` 是同步假象（[lwd_edge_core.py:56](vllm_tywww/vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_core.py#L56) 自己注明）。源仓 `poll_embed_acks` 的“逐 chunk 回执 → 进度/终结驱动”整条链没有对应物。这已在 §9.12 登记为“数据面落位时重定义接缝”，属于**已知开放项而非遗漏**，但要清楚：目前控制面是“骨架正确、数据面空转”。

**2. abort 的在途状态机没有等价物**
源仓三时机里“有在途 chunk”→ 标 `aborting`、**等末次 ack 才 finalize**（保证发送完整性）。wangwei 是立即 `finish_requests`（[core.py:401-407](vllm_tywww/vllm/v1/engine/core.py#L401-L407)）。同步执行占位下无害；数据面异步化后，“isend 不可撤销、请求不能提前释放”的延迟终结状态机需要回来——同样挂在 §9.12 接缝下，建议和第 1 点一起登记。

**3. [PO-WIRE-O] 调试逃生门缺失**
源仓 `VLLM_ASCEND_PREFILL_ONLY_DEBUG_NO_EMBED_WIRE` 可整体关掉 notify+isend 只发元数据（联调利器）。wangwei 的 `debug` 只是日志开关，无等价“只走控制面”模式。边云联调初期大概率会需要。

**4. 观测面降级**
源仓有 `[PO-TTFT]` 事件流（meta_pub / first_token / notify_pub / dispatched）+ outstanding 门日志。wangwei 只有 `LwdLog.phase`（debug 关时零输出），且配置里的 `zombie_log_interval_s` 传进了 settings 但边侧**没有任何消费点**。丢元数据/丢 abort 只有一次 warning，无计数器。移植文档说“要观测走 lwd_diagnostics”——但 lwd_diagnostics 本身还没落。

**5. 元数据只带 max_tokens，采样参数上不了云**
[LwdRequestNotify](vllm_tywww/vllm/v1/lwd_control/control_communication/lwd_notify.py#L34-L48) 只有 `num_prompt_tokens/max_tokens/block_hashes`，docstring 自认“采样参数等扩展由后续迭代增补”。源仓传全量 `EngineCoreRequest`（含 sampling_params），云能完整生成；新架构下云拿不到 temperature/top_p/stop——与“客户端从哪拿 token”同族的已登记开放项，但这条比回传更早爆炸：**云侧生成质量直接依赖它**。

**6. 小项**：① notify 持续队满时 busy loop 无 1ms yield（源仓有），自旋会与 ZMQ 发送线程抢 GIL；② core.py add_request 守卫旁路了 `request_wave` 参数（历史已记录，DP 场景需补）。

## 结论

**控制面的控制流（校验→元数据→调度→预告→对账→终结→abort→关停）已无功能缺口**；剩余差距集中在三类：①数据面接缝占位（§9.12 已登记的开放项，含 abort 在途状态机），②观测/调试面（PO-WIRE-O、TTFT、zombie 消费点），③元数据携带面（采样参数）。若严格按“只搞边侧控制面”的口径，建议优先补第 4 点的 `zombie_log_interval_s` 消费点和第 6-① 的 yield（都是控制面内的小改），其余登记台账即可。


---

## User <sup>2026/9/9 00:15:06</sup>

梳理一下当前边侧控制面流程，要包含下发本地worker和zmq发送给云


---

## Assistant <sup>2026/9/9 00:15:12</sup>


---

## Assistant <sup>2026/9/9 00:15:12</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "C:\\prefill_only_wangwei\\vllm_tywww\\vllm\\v1\\lwd_control\\control_communication\\lwd_control_publisher.py"
}
```

</details>


---

## User <sup>2026/9/9 00:15:12</sup>

<details>
<summary>Tool Result</summary>

```
1	"""传输层方向原语:OUTBOUND 发布端(side-agnostic)。
2	
3	边/云身份与 bind/connect 是装配期 wiring(control_scheduler 侧
4	assemble 文件决定);本类自有线程把有界队列的元数据编码发送,
5	对谁发布一无所知。
6	"""
7	
8	from __future__ import annotations
9	
10	import contextlib
11	import queue
12	import threading
13	
14	import zmq
15	
16	from vllm.v1.lwd_control.control_communication.lwd_control_communicator import (
17	    LwdControlCommunicator,
18	)
19	from vllm.v1.lwd_control.control_communication.lwd_notify import (
20	    LwdNotify,
21	    lwd_encode_notify,
22	)
23	
24	# 关停令:发布线程收到即退出;None 不会与通知混淆(它们是 msgspec 结构)
25	_LWD_PUBLISH_SHUTDOWN = None
26	
27	LWD_PUBLISH_QUEUE_MAX = 1000
28	
29	
30	class LwdControlPublisher:
31	    """控制面发布端;publish 队满返回 False,调用方视为未派发、下一步重试(§2.4 背压)。
32	
33	    PUSH 的阻塞语义天然形成对端背压:对端消费慢 -> send 阻塞 -> 内部
34	    有界队列涨满 -> publish 返回 False,本步不执行(§9.1 静态封顶)。
35	    """
36	
37	    def __init__(
38	        self,
39	        endpoint: str,
40	        *,
41	        bind: bool,
42	        queue_max: int = LWD_PUBLISH_QUEU
...(truncated)
```

</details>


---

## Assistant <sup>2026/9/9 00:15:12</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "C:\\prefill_only_wangwei\\vllm_tywww\\vllm\\v1\\lwd_control\\control_communication\\lwd_control_communicator.py"
}
```

</details>


---

## User <sup>2026/9/9 00:15:12</sup>

<details>
<summary>Tool Result</summary>

```
1	"""控制面通信收发器:ZMQ socket 句柄,只做收发,无线程(§9.1)。
2	
3	socket 单线程亲和:send/recv 只能由持有线程调用;close/term 是允许
4	跨线程的关停调用。term 是跨线程打断阻塞 recv/send 的可靠手段
5	(使其以 ETERM 返回);close(0) 不保证唤醒。关停编排由持有方负责。
6	本文件不知道任何通知类型;通知协议住 lwd_notify.py(§10.3)。
7	"""
8	
9	from __future__ import annotations
10	
11	import zmq
12	
13	
14	class LwdControlCommunicator:
15	    """PRE_OUT 平面收发器:endpoint socket 的收发句柄,无线程。"""
16	
17	    def __init__(self, endpoint: str, socket_type: int, *, bind: bool) -> None:
18	        self._context = zmq.Context()
19	        self._socket = self._context.socket(socket_type)
20	        # 退出时不因未发完的帧挂死进程;显式 close(0) 可覆盖
21	        self._socket.setsockopt(zmq.LINGER, 2000)
22	        if bind:
23	            self._socket.bind(endpoint)
24	        else:
25	            self._socket.connect(endpoint)
26	
27	    def send(self, data: bytes) -> None:
28	        """阻塞发送;无对端时阻塞,由调用方有界队列形成对端背压。"""
29	        self._socket.send(data)
30	
31	    def recv(self) -> bytes:
32	        """阻塞接收;已 close 的 socket 再 rec
...(truncated)
```

</details>


---

## Assistant <sup>2026/9/9 00:15:16</sup>


---

## Assistant <sup>2026/9/9 00:15:17</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "sed -n '460,545p' /c/prefill_only_wangwei/vllm_tywww/vllm/v1/engine/core.py",
  "description": "Read step/step_with_batch_queue guard context"
}
```

</details>


---

## User <sup>2026/9/9 00:15:17</sup>

<details>
<summary>Tool Result</summary>

```
" (dummy)" if is_dummy else "",
                ]
            )
        )
        self._iteration_index += 1

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """
        # Lwd prefill-only: same delegation as step_with_batch_queue — this
        # path is selected when batch_queue is None (step_fn wiring).
        if self.step_wrapper is not None:
            return self.step_wrapper.step_with_batch_queue()

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bit
...(truncated)
```

</details>


---

## Assistant <sup>2026/9/9 00:15:25</sup>


---

## Assistant <sup>2026/9/9 00:15:53</sup>

边侧控制面全景已梳理完（基于 wangwei 当前代码）。整个流程分五段：装配、请求入口、步进主循环、abort、关停。先给总图，再逐段展开。

## 总图

```
客户端请求 ──► EngineCore.add_request 守卫 (core.py:360)
                    │
                    ▼
        LwdEdgeScheduler.lwd_edge_add_request
        ├── ① _lwd_validate_request 五项校验
        ├── ② lwd_edge_notify_request ──► LwdRequestNotify ─┐
        └── ③ super().add_request 原生入 waiting             │
                                                              │ 有界队列(1000)
每步: EngineCore.step/step_with_batch_queue 守卫               ▼
  (core.py:474/528) ──► LwdEdgeCore.step_with_batch_queue   后台线程
                    │                                    msgspec.msgpack
                    ├─ schedule() 原生分块                  │ PUSH (connect)
                    │   记录 _lwd_last_scheduled            ▼
                    ├─ lwd_edge_notify ──► LwdRangeNotify ──► ZMQ tcp://host:5558 ──► 云
                    │   (offset = num_computed - 本步量,     (PRE_OUT, 单向)
                    │    seqno 单调)
                    ├─ lwd_execute_model ──► 本地 worker 前向(同步 .result())
                    └─ lwd_edge_update_progress 步末对账
                        ├── 回退未派发量 + 占位符归零
                        └── 全嵌入 → finish_requests(FINISHED_STOPPED)

abort: EngineCore.abort_requests 守卫 (core.py:401)
  ├── lwd_edge_abort ──► LwdAbortNotify ──► ZMQ ──► 云
  └── 原生 finish_requests(FINISHED_ABORTED) 本地清理
```

## 一、装配期（进程启动，一次）

1. `EngineCore.__init__` 尾部（[core.py:239-241](vllm_tywww/vllm/v1/engine/core.py#L239-L241)）调 `lwd_try_assemble(self)`——非 prefill_only 模式直接返回 False，零副作用
2. [lwd_edge_try_assemble](vllm_tywww/vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_assemble.py#L141-L165)：
   - `LwdConfig.from_env_and_config` 解析 `edge_cloud_config` 段 + `VLLM_ASCEND_LWD_*` env 覆盖（角色/端点/队列上限）
   - **建 ZMQ 通道**：`LwdControlPublisher(tcp://host:5558, bind=False)`——边侧 connect，云侧 bind 同一端点，自启动后台发送线程
   - **整实例替换调度器**：原生 scheduler → `LwdEdgeScheduler(publisher=...)`（publisher 经构造注入）
   - **建 step 委托对象**：`engine_core.step_wrapper = LwdEdgeCore(port, settings)`——此后 core.py 所有运行时守卫只看 `step_wrapper is not None`
3. kv_connector 在场时告警并降级原生（握手态会丢）

## 二、请求入口（每请求一次，控制面出口 ①）

`add_request` 守卫旁路原生路径，单点交给 [lwd_edge_add_request](vllm_tywww/vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_scheduler.py#L60-L81)：

1. **五项校验**（`_lwd_validate_request`）：客户端 embeds / 空 prompt / pooling / mm / structured → ValueError 拒于入队前
2. **元数据先行上云**：`lwd_edge_notify_request` 构造 `LwdRequestNotify(request_id, num_prompt_tokens, max_tokens, block_hashes)`——block_hashes 是边侧算好的 prompt 全量满块哈希链，让云侧占位零值 token 也能按真实内容命中前缀缓存（§10.13）。**publish 队满时短退避重试 3 次（0.1s/0.2s/0.3s），超限丢弃 + warning**（§8.3-2 裁定无自动回压，云侧 zombie 检测兜底）
3. **本地登记**：`super().add_request` 走原生——Request 入 waiting 队列、streaming/重复 id 处理全复用

## 三、步进主循环（每步，控制面出口 ②+本地 worker 下发）

EngineCore 主循环经 [step / step_with_batch_queue 守卫](vllm_tywww/vllm/v1/engine/core.py#L474-L475)（两条路径都委托，单批配置不漏）进入 [LwdEdgeCore.step_with_batch_queue](vllm_tywww/vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_core.py#L31-L41) 四段编排：

**1. schedule**（`_lwd_edge_step_schedule`）：调 `LwdEdgeScheduler.schedule()`——全量复用原生 chunked prefill 决策（token 预算 + KV 容量感知 + 原生 waiting 准入），仅多做一件事：把本步 `num_scheduled_tokens` 存进 `_lwd_last_scheduled` 作为步末对账基准。

**2. notify 上云**（[lwd_edge_notify](vllm_tywww/vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_scheduler.py#L83-L107)）：对每个新调度范围发 `LwdRangeNotify(request_id, offset, num_tokens, seqno)`：
- `offset = num_computed_tokens - 本步排程量`——fork 的 `_update_after_schedule` 在 schedule() 尾部已乐观推进 num_computed，必须回退
- `seqno` 由 `_lwd_edge_next_seqno` 全局单调分配，显式携带（云侧 seqno→request 登记、tag-less HCCL 配对契约的事实源）
- **publish 队满返回 False → 本步不派发**（原子性：数据绝不先于预告上云），下一步原生复现同一范围重试

**3. 下发本地 worker**（[lwd_edge_assemble.py:136-138](vllm_tywww/vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_assemble.py#L136-L138)）：`lwd_execute_model` 把 SchedulerOutput 经 `model_executor.execute_model(scheduler_output).result()` 提交本地 worker 跑前向并**同步等待**。当前这是占位接缝：跑的是原生模型前向，不是源仓的 EDGE_EMBED 嵌入批；`executed = 调度量` 是同步假象（§9.12 已登记：数据面落位后改报实际执行量、改走嵌入路径）。

**4. 步末对账**（[lwd_edge_update_progress](vllm_tywww/vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_scheduler.py#L152-L176)）：
- 遍历基准是 `_lwd_last_scheduled`（不是 executed）：`num_executed = executed.get(rid, 0)`，未派发即**全量回退** num_computed（消掉乐观推进的虚记）
- 回退分支同时 `is_prefill_chunk = True` 复原 + `num_output_placeholders = 0`（防幻影 decode token）
- `num_computed >= num_prompt_tokens` → `finish_requests(FINISHED_STOPPED)` **本地终结，不发任何信号**（单向模型：本地终结对云不可见，§9.10）

## 四、abort 路径（控制面出口 ③）

[abort_requests 守卫](vllm_tywww/vllm/v1/engine/core.py#L401-L407)：
1. 先 `lwd_edge_abort`——对**每个** rid（含未知 rid，“元数据已先行，abort 必达”）发 `LwdAbortNotify` 上云；队满尽力而为（warning，云侧自身 finish/zombie 兜底）
2. 再原生 `finish_requests(FINISHED_ABORTED)` 本地清理。在途数据不撤销（isend 不可取消，云侧收到 abort 后自行丢弃多余，§14.7）

## 五、ZMQ 发送通道内部（ publish 之后的路）

[lwd_control_publisher.py](vllm_tywww/vllm/v1/lwd_control/control_communication/lwd_control_publisher.py)：
- 主循环只 `put_nowait` 进有界队列（默认 1000）——**ZMQ 编码/发送成本完全不落主循环**
- 后台线程 `lwd-publisher`：`queue.get()` → `msgspec.msgpack.encode` → `LwdControlCommunicator.send`（PUSH socket，阻塞 send）
- **背压传导链**：云消费慢 → PUSH send 阻塞 → 队列涨满 → `publish` 返回 False → 本步不派发（§9.1 静态封顶）
- 关停：关停令入队 → join(2s) → 残留线程以 `context.term()` 收尾（跨线程打断阻塞 send 的可靠手段）

## 六、关停

[shutdown 守卫](vllm_tywww/vllm/v1/engine/core.py#L643-L646) → `lwd_edge_shutdown` → publisher.shutdown()（先关通道再走原生 teardown），幂等。

---

**一句话**：控制面在每条路径上只有三个出口——请求预告（入口）、范围预告（每步）、abort（断连），全部经同一个有界队列 + 后台线程 PUSH 上云；本地 worker 下发复用原生 `execute_model` 同步占位；请求生命周期（进度/终结/对账回退）完全由本地簿记驱动，不依赖云的任何回话。当前唯一的“假”环节是第三段的同步执行接缝，等数据面（EDGE_EMBED + 异步回执）落位时替换。


---

## User <sup>2026/9/9 09:04:48</sup>

lwd_edge_notify和lwd_edge_notify_request 有什么区别


---

## Assistant <sup>2026/9/9 09:05:00</sup>


---

## Assistant <sup>2026/9/9 09:05:10</sup>

两者都是边→云的预告（都走同一个 PRE_OUT publisher），但回答的问题、时机、粒度完全不同：

## 逐项对照

| 维度 | `lwd_edge_notify_request` | `lwd_edge_notify` |
|---|---|---|
| **回答云的问题** | “**有一个请求来了**”——谁、多长 prompt、最多生成多少 token | “**本步要发哪段嵌入**”——哪个请求、从哪个 offset 起、多少 token |
| **消息体** | `LwdRequestNotify(request_id, num_prompt_tokens, max_tokens, block_hashes)` | `LwdRangeNotify(request_id, offset, num_tokens, seqno)` |
| **触发时机** | 请求入口：`add_request` 守卫内，**每请求恰好一次** | 步进循环：每步对**新调度的每个范围**发，**一个请求通常发多次**（prompt 被切成 N 段就发 N 次） |
| **携带的关键载荷** | block_hashes 全量满块哈希链（云侧前缀缓存按真实内容命中，§10.13） | seqno 单调序号（云侧 seqno→request 登记 + tag-less HCCL 配对事实源） |
| **队满语义** | 短退避重试 3 次后**丢弃 + warning**（§8.3-2 无自动回压；丢了云永远不知道这个请求） | 返回 **False → 本步不派发执行**，下一步重试（原子性：数据不得先于预告上云） |
| **失败后果** | 云侧失步，靠 zombie 检测兜底 | 无损失——原生调度下一步自然复现同一范围，重复预告云侧按 (request_id, offset) 幂等登记 |

## 在流程里的位置

```
add_request 守卫
  └─ lwd_edge_notify_request   ← 每请求 1 次："登记这个请求"
       │
       ▼  (请求进 waiting，等调度)
每步 step_with_batch_queue
  ├─ schedule() 原生分块
  ├─ lwd_edge_notify            ← 每段 1 次："这段嵌入要来了"
  ├─ lwd_execute_model          ← 数据随后跟上
  └─ ...
```

## 为什么必须是两条消息而不是一条

源仓把两者合在 `EngineCoreRequest`（元数据本身携带 chunk plan）里一次发；新架构拆开，是因为**分块决策不再是入口时刻已知的**——复用原生 `schedule()` 后，一个 prompt 切几段、每段多大，取决于调度时的 token 预算和 KV 容量（可变切分），入口时刻算不出来。所以入口只发“请求级静态事实”（长度、哈希链、max_tokens），每步再补“范围级动态决策”（offset/num_tokens/seqno）。

一句话：**notify_request 是“登记”，notify 是“排期”**——前者告诉云这个请求存在且该怎么准备（前缀缓存、终结预算），后者告诉云即将到达的每一段嵌入属于谁、落在哪、按什么顺序配对。


---

## User <sup>2026/9/9 09:14:27</sup>

源仓EngineCoreRequest携带了prompt_tokenids，确认一下； 新架构是否不需要了


---

## Assistant <sup>2026/9/9 09:14:41</sup>


---

## Assistant <sup>2026/9/9 09:14:41</sup>

先确认源仓事实，再看新架构云侧怎么消化。


---

## Assistant <sup>2026/9/9 09:14:43</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "grep -n \"prompt_token_ids\" /c/prefill_only/current/vllm-ascend_qxxxw/vllm_ascend/edge_cloud/edge_prefill_dispatcher.py | head; echo ====; grep -n \"class PrefillOnlyChunkPlan\" -A 12 /c/prefill_only/current/vllm_qxxxw/vllm/v1/engine/__init__.py | head -20; echo ====; grep -n \"prompt_token_ids\\|prefill_only_chunk_plan\" /c/prefill_only/current/vllm_qxxxw/vllm/v1/engine/__init__.py | head",
  "description": "Confirm source wire struct carries token ids"
}
```

</details>


---

## User <sup>2026/9/9 09:14:44</sup>

<details>
<summary>Tool Result</summary>

```
69:    prompt_token_ids: list[int]
241:        n_tokens = len(request.prompt_token_ids)  # type: ignore[arg-type]
555:            token_ids=state.prompt_token_ids[offset:offset + num_tokens],
571:            prompt_token_ids=list(request.prompt_token_ids),  # type: ignore[arg-type]
584:            self._register(request, len(request.prompt_token_ids))  # type: ignore[arg-type]
607:        if not request.prompt_token_ids:
====
83:class PrefillOnlyChunkPlan(
84-    msgspec.Struct,
85-    array_like=True,  # type: ignore[call-arg]
86-    omit_defaults=True,  # type: ignore[call-arg]
87-    gc=False,
88-):  # type: ignore[call-arg]
89-    """Edge-cloud prefill-only chunk plan, carried by EngineCoreRequest.
90-
91-    Tells the cloud how the edge will stream this request's embedding
92-    chunks (chunk boundaries at multiples of `chunk_size`, in chunk_idx
93-    order).  The embeds themselves travel on the PREFILL_UP HCCL channel;
94-    this message never carries tensors.
95-    """
====

...(truncated)
```

</details>


---

## Assistant <sup>2026/9/9 09:14:48</sup>


---

## Assistant <sup>2026/9/9 09:14:48</sup>

源仓事实确认。再看新架构云侧如何从 `LwdRequestNotify` 构造请求：


---

## Assistant <sup>2026/9/9 09:14:49</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "grep -n \"num_prompt_tokens\\|prompt_token_ids\\|placeholder\\|占位\\|Request(\" /c/prefill_only_wangwei/vllm_tywww/vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_phase_scheduler.py | head -30",
  "description": "Check cloud-side request construction"
}
```

</details>


---

## User <sup>2026/9/9 09:14:49</sup>

<details>
<summary>Tool Result</summary>

```
79:        # 请求工厂(装配层绑定):线上元数据 -> Request(L3 唯一建请求点)
264:            n_prompt = request.num_prompt_tokens
301:            # Every running request is gated (async placeholders etc.) and
```

</details>


---

## Assistant <sup>2026/9/9 09:14:53</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "grep -n \"factory\\|_request_factory\\|def.*request\\|Request(\" /c/prefill_only_wangwei/vllm_tywww/vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_assemble.py | head -20",
  "description": "Find cloud request factory"
}
```

</details>


---

## User <sup>2026/9/9 09:14:54</sup>

<details>
<summary>Tool Result</summary>

```
109:    def _on_abort_notify(self, request_id: str) -> None:
132:    def _forward_drop(self, request_id: str) -> None:
186:    scheduler.lwd_cloud_bind_request_factory(_lwd_cloud_build_request(engine_core))
206:def _lwd_cloud_build_request(engine_core) -> Callable[[LwdRequestNotify], Request]:
226:        return Request(
248:    def hasher(request: Request) -> list[bytes]:
262:    def _admit(request: Request) -> None:
272:    def _abort(request_id: str) -> None:
```

</details>


---

## Assistant <sup>2026/9/9 09:14:57</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "sed -n '206,260p' /c/prefill_only_wangwei/vllm_tywww/vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_assemble.py",
  "description": "Read cloud request factory body"
}
```

</details>


---

## User <sup>2026/9/9 09:14:57</sup>

<details>
<summary>Tool Result</summary>

```
def _lwd_cloud_build_request(engine_core) -> Callable[[LwdRequestNotify], Request]:
    """唯一请求构建点:Request 构建/block_hasher 全收于此(§7.3-C1)。

    哈希链获取(§10.13):云 prompt token 是占位零值,本地算不出真实
    链 —— prompt 首建直接用边侧预告里的 block_hashes,decode 续算回
    本地 hasher;边侧未提供或长度不符回退本地(占位链,命中无效但不
    崩)。数据面挂载(prompt_embeds 视图)由数据面落位侧在此对接
    (§9.12);产物经提升出口进 input_queue,循环线程原生 ADD 分发到
    调度器暂存(§10.10/§10.12)。
    """
    local_hasher = engine_core.request_block_hasher
    if local_hasher is not None:
        # 与 core.py 构造期同函数同输入,结果确定一致
        _, hash_block_size = resolve_kv_cache_block_sizes(
            engine_core.scheduler.kv_cache_config, engine_core.vllm_config
        )
    else:
        hash_block_size = 0

    def _build(wire: LwdRequestNotify) -> Request:
        return Request(
            request_id=wire.request_id,
            # 占位 token:云侧调度只看长度,真值由边侧 embeds 经数据面提供(§9.5)
            prompt_token_ids=[0] * wire.num_prompt_tokens,
            sampling_params=SamplingParams(max_tokens=wire.ma
...(truncated)
```

</details>


---

## Assistant <sup>2026/9/9 09:15:11</sup>


---

## Assistant <sup>2026/9/9 09:15:28</sup>

## 源仓确认：携带，且是全量

- [vllm_qxxxw/vllm/v1/engine/__init__.py:111](C:\prefill_only\current\vllm_qxxxw\vllm\v1\engine\__init__.py#L111)：`EngineCoreRequest.prompt_token_ids: list[int] | None`——主载荷，边 tokenize 结果**全长随元数据一次发出**（末尾还追加了 `prefill_only_chunk_plan` 字段，:159）
- [edge_prefill_dispatcher.py:571](C:\prefill_only\current\vllm-ascend_qxxxw\vllm_ascend\edge_cloud\edge_prefill_dispatcher.py#L571)：publish 时 `prompt_token_ids=list(request.prompt_token_ids)` 原样带走——整条 prompt 的**内容**都在控制面线上（~3 B/token，64K prompt ≈ 200 KB）

## 新架构：确实不需要了，云侧用占位零值

线上元数据只有 `LwdRequestNotify(request_id, num_prompt_tokens, max_tokens, block_hashes)`——没有 token ids。云侧建请求时（[lwd_cloud_assemble.py:226-227](vllm_tywww/vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_assemble.py#L226-L227)）：

```python
# 占位 token:云侧调度只看长度,真值由边侧 embeds 经数据面提供(§9.5)
prompt_token_ids=[0] * wire.num_prompt_tokens,
```

**为什么成立**——此前分析出的“云侧读 token 内容的四个消费点”，新架构逐个有了替代：

| 源仓消费点 | 新架构替代 |
|---|---|
| ① 块哈希/前缀缓存（内容敏感，len-only 最危险——等长请求哈希碰撞会静默复用错 KV） | **block_hashes 全量满块链随预告下发**：prompt 首建直接用边侧链（[lwd_wire_hasher](vllm_tywww/vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_assemble.py#L249-L258)），decode 续算才回本地 hasher。链断了回退本地占位链（fail-open：命中无效但不崩） |
| ② 长度推导（num_prompt_tokens / mask / block table） | `num_prompt_tokens` 显式携带 |
| ③ worker batch 重建 / PP 恢复（要 `_all_token_ids` 链一致） | 占位零值本身就是 `_all_token_ids` 的来源，链自洽；decode 追加的是云**自己采样**的真值 token，无需边提供 |
| ④ wire-off 调试模式（云用 ids 跑原生 prefill） | **没有替代，通道已废**（新架构连 PO-WIRE-O 等价物都没有，上一轮已列为差距） |

深层原因：remote-embeds 模式下 prompt 的计算语义全部来自数据面的 embeds（prefill hidden = HCCL 收到的张量，`prompt_is_token_ids` 全 False 掩码），**token 内容在云的计算图里本来就不参与**——源仓传 ids 是为了①和④，不是为了计算。

**收益**：3 B/token 全部消失——64K prompt 的元数据从 ~200 KB 降到几十字节 + 哈希链（每满块一个 hash，64K/16 = 4096 × ~32B ≈ 128 KB，仍比 ids 小且换来前缀缓存能力）。

## 两个连带提醒

1. **decode 之后 hash 链的“续算”**：prompt 段用边侧链没问题，但请求进入 decode 后新增块的哈希由云本地 hasher 续算——此时输入是占位零值 + 真实采样 token 混合，**decode 段的缓存哈希对前缀复用无意义（本机不可能有别人复用）**，正确性无损，只是该段缓存价值为零——这在设计上可接受。
2. **这是“内容不需要”，不是“什么都不缺”**：`max_tokens` 之外的采样参数（temperature/top_p/stop…）仍在元数据之外（上轮差距第 5 点），且①成立的前提是边侧 `block_hashes` 算得准——`hash_block_size` 两边必须一致（云侧 resolve 用同函数同输入，[lwd_cloud_assemble.py:216-218](vllm_tywww/vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_assemble.py#L216-L218) 注释确认了这点）。

结论：**确认源仓携带全量 ids；新架构设计上不再需要**——长度走 `num_prompt_tokens`、前缀缓存走哈希链、计算走 embeds，四个内容消费点三真一废。


---
