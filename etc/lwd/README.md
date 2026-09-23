# LWD 单实例配置与 NPU 验证

控制面已按设计切换为边连云 ROUTER/DEALER：云侧 ROUTER bind
`ctrl_port`（唯一 bind 方），边侧 DEALER connect + 稳定 identity
（`edge-{id}-{dp}`），register/ack 取代旧 HELLO 首拍发现；消息携带
`edge_id/dp_idx`，云侧按 `"{edge}#{dp}#{rid}"` 前缀隔离请求（fan-in
拆发已就绪，多边场景的选路/策略未做）。数据面不变（no-PP、P2P +
云内 TP broadcast）；多 DP 调度或共享卡执行仍未实现。
本机已通过 ROUTER/DEALER 闭环自测（register/ack、定向回包、背压、
关停），以下是待上板验证的步骤，不是验证通过记录。

## 文件与支持范围

- `lwd_config.yaml`：当前上板入口，单实例、单 DP，边 TP=1、云 TP=8，world=9。
- `lwd_config_2dp.yaml`：配置预留示例，保留两个 DP；边 rank 0 共享，云每 DP 4 卡。
  解析器支持该结构，当前运行入口会明确报多 DP 暂不支持，不会只取 DP0 继续跑。
- 上板配置固定为 vllm 仓的 `etc/lwd/lwd_config.yaml`，不按模型重命名，
  vllm-ascend 不维护第二份。多 DP 文件仅为预留示例，不是当前启动入口。
  若仓库在 `/vllm-workspace/vllm`，实际路径就是
  `/vllm-workspace/vllm/etc/lwd/lwd_config.yaml`。
  不再使用系统目录 `/etc/lwd/` 作为本次部署位置。

两端都需要本次 `prefill_only_newsetting` 的 vllm 代码，以及包含 no-PP 的
vllm-ascend（本次基线 `11b6a1d54`）。不能只更新一个仓或只更新一侧。
如果代码尚未提交，`git pull` 不会取得工作区里的修改，需要先提交/同步代码。

## 配置替换

1. `lwd_config.yaml` 已填入当前 Qwen3.8 环境：边 `76.76.26.18`、
   云 `76.76.26.234`、云 `ctrl_port: 6453`。两侧使用相同内容，
   文件名及仓内位置固定；以后更换环境只修改文件内容。
2. 保留正常模型、网卡、可见 NPU、图模式等设置。边侧选 1 张卡，云侧选 8 张卡。
   `ranks` 是全局逻辑 rank，不是 `ASCEND_RT_VISIBLE_DEVICES` 的物理卡号。
3. `additional-config.lwd_config` 只填 `path/role/instance_id`。
   其他插件字段（CPU binding、NZ 等）保持不变。
4. 移除旧 `enabled/mode/edge_head_tail_layers/pre_out_host/post_out_host` 等 JSON 字段，
   移除 `--edge-npu-count`、`--cloud-npu-count`。卡数改由 YAML ranks 推导。
5. TP 必须显式一致：边 `--tensor-parallel-size 1`，云 `--tensor-parallel-size 8`。
   不再用云 TP=1 再自动覆盖成 8 的旧写法。PP=1、DP=1。
6. 不必重复传 `--nnodes/--node-rank/--master-addr`；当前适配层由 YAML/role 设置
   为两侧节点、边 0 云 1、master=edge.addr。若保留 `--master-addr`，必须与边地址一致。
   `--master-port` 不是本次 no-PP TCPStore 的端口。

缺少 lwd_config 表示非 LWD；提供 lwd_config 但路径缺失、文件错误、未知字段、
实例不存在或 TP 矛盾，均报错退出，不回退旧配置。
正确字段名为 `scene`、`enable_early_recv`。
两个 feature 开关默认 false，仅解析下发；设置 true 会提示尚未接入执行功能。

## 端口：云侧单端口双向（边侧零端口）

| 用途 | 来源 | 示例 |
| --- | --- | --- |
| 控制面 ROUTER（云 bind、边 DEALER connect；register/ack、Request/Range/Abort、C2e 全走这一个端口，双向） | 云 `dp.addr:ctrl_port` | 76.76.26.234:6453 |
| no-PP 共享世界 TCPStore | 边 `dp.addr` + 默认端口 29600 | 76.76.26.18:29600 |

边侧不再 bind 任何控制面端口。`VLLM_ASCEND_LWD_POST_OUT_PORT` 已退役：
**设置它会在启动时报错**（防旧脚本带毒），确认两侧脚本里已删除该 export。
旧传输配置的 PRE_OUT_HOST/PRE_OUT_PORT/POST_OUT_HOST/POST_OUT_BIND/
WIRE_STORE_PORT/HELLO_TIMEOUT_S/DEBUG 环境覆盖链均已移除。
控制端口只改 YAML 的 `ctrl_port`；TCPStore 固定 29600；register 等待
预算沿用 600 秒。确认云 6453 和边 29600 可达且未被占用。

## 边 1 卡 / 云 8 卡启动示例

先沿用板上已经可工作的 CANN/NPU、网卡和 PYTHONPATH 环境。下面模型路径、
模型名和可见卡号按实际修改；不要直接套用旧 GLM 1+16 脚本的卡数。
本例使用报错日志里的 Qwen3.8-27B；若用设计文档的 Qwen3.6，则两侧一起更换模型。

边侧（启动后可能等待云侧，两边需在各自终端启动，不要等边侧完全 ready 才启动云）：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
vllm serve /weight/Qwen3.8-27B \
    --host 0.0.0.0 --port 8060 \
    --served-model-name qwen3.8 \
    --trust-remote-code \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 1 \
    --additional-config '{"enable_cpu_binding":true,"enable_weight_nz_layout":true,"lwd_config":{"path":"/vllm-workspace/vllm/etc/lwd/lwd_config.yaml","role":"edge","instance_id":0}}' \
    --compilation-config '{"cudagraph_mode":"NONE"}' \
    --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 8192 \
    --no-enable-prefix-caching --async-scheduling --gpu-memory-utilization 0.9
```

云侧（不加 headless；保留云 EngineCore 和调度器）：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
vllm serve /weight/Qwen3.8-27B \
    --served-model-name qwen3.8 \
    --trust-remote-code \
    --tensor-parallel-size 8 --pipeline-parallel-size 1 --data-parallel-size 1 \
    --additional-config '{"enable_cpu_binding":true,"enable_weight_nz_layout":true,"lwd_config":{"path":"/vllm-workspace/vllm/etc/lwd/lwd_config.yaml","role":"cloud","instance_id":0}}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16]}' \
    --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 8192 \
    --no-enable-prefix-caching --async-scheduling --gpu-memory-utilization 0.9 \
    --speculative-config '{"num_speculative_tokens":3,"method":"mtp"}'
```

## 上板验收

启动时保持 INFO 级别（不要设置 `VLLM_LOGGING_LEVEL=ERROR`），搜索
`[LWD][config]`。新增日志按以下阶段区分，打印的是解析后保存/实际返回的对象，
不是重新打开 YAML，也不打印其他插件的 additional_config 内容：

| 日志标记 | 能确认什么 | 参数保存位置 |
| --- | --- | --- |
| `[LWD][config][parsed]` | 文件读取、schema 校验和实例查找已成功；完整保留五块拓扑、所有 DP、缺省 feature 值及拓扑指纹 digest | `vllm_config.lwd_config`，完整 YAML 在其 `.topology` |
| `[LWD][config][selected]` | 当前 role/instance_id 选中的实例及其全部 DP | `vllm_config.lwd_config.instance` |
| `[LWD][config] path=... TP=...` | 已通过当前单 DP 运行限制及 TP 一致性检查，并已映射并行参数 | `vllm_config.parallel_config` 及其 `.lwd_config` |
| `[LWD][config][transport]` | 实际适配出的 ROUTER bind 端点、per-link DEALER 端点、identity、wire_store、超时等所有传输值 | `LwdConfig.from_vllm_config()` 返回的控制面配置对象 |

`parsed`/`selected` 在运行限制检查之前打印，所以多 DP 文件也能看到完整解析结果，
随后才报运行时暂不支持；不能把这两条日志当成启动成功。
`transport` 在每个进程里对相同配置只打印一次，不在每个请求里重复打印。
原始入口仍在 `vllm_config.additional_config["lwd_config"]`，仅包含 path/role/instance_id。
feature 为 true 仍只是保留参数，不代表对应功能已经启用。

1. 配置日志 `[LWD][config]`：边 role=edge、ranks=(0,)、TP=1、PP=1、world=9；
   云 role=cloud、ranks=(1,...,8)、TP=8、PP=1、world=9；两侧 digest 一致。
2. 两侧 `[LWD] distributed init method from lwd_config` 都是边 IP:29600。
   并行组日志应显示云内 8 卡 TP、PP 单 rank 组；这只能说明初始化，不代表请求已成功。
3. 云侧日志出现 `edge registered`（报 identity 与 edge/dp），边侧出现
   `edge engine assembled`（报 DEALER 条数与 identity）；register 互校
   （版本/卡数/digest）不符会拒绝注册并使边侧 fail-fast，两侧拓扑文件
   不一致在这一步拦截。
4. 先完成一次请求的 prefill/decode 和正常返回，再运行已有 bench。
   使用与原可运行基线相同的模型、输入和 bench 设置进行对照。
   无 AttributeError/KeyError、无通信挂起、bench 正常结束才算全流程通过。

可先发一个短请求：

```bash
curl http://127.0.0.1:8060/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8","messages":[{"role":"user","content":"你好"}],"max_tokens":64,"temperature":0}'
```

若失败，保留两侧首次异常前后的完整日志、两仓 commit、YAML 内容及启动命令；
不要仅用后续 EngineDeadError 判断根因。
