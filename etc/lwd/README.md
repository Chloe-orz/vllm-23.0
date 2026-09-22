# LWD 单实例配置与 NPU 验证

本次只迁移配置，保留现有 no-PP（PP=1）、P2P + 云内 TP broadcast、
PUSH/PULL 通信。没有实现 ROUTER/DEALER、多 DP 调度或共享卡执行。
本轮按用户要求不运行测试，以下是待上板验证的步骤，不是验证通过记录。

## 文件与支持范围

- `lwd_config.yaml`：当前上板入口，单实例、单 DP，边 TP=1、云 TP=8，world=9。
- `lwd_config_2dp.yaml`：配置预留示例，保留两个 DP；边 rank 0 共享，云每 DP 4 卡。
  解析器支持该结构，当前运行入口会明确报多 DP 暂不支持，不会只取 DP0 继续跑。
- YAML 放在 vllm 仓的 `etc/lwd/`，vllm-ascend 不维护第二份。
  若仓库在 `/vllm-workspace/vllm`，实际路径就是
  `/vllm-workspace/vllm/etc/lwd/lwd_config.yaml`。
  `/etc/lwd/lwd_config.yaml` 仅是可选安装路径，并不要求复制过去。

两端都需要本次 `prefill_only_newsetting` 的 vllm 代码，以及包含 no-PP 的
vllm-ascend（本次基线 `11b6a1d54`）。不能只更新一个仓或只更新一侧。
如果代码尚未提交，`git pull` 不会取得工作区里的修改，需要先提交/同步代码。

## 配置替换

1. 修改 `lwd_config.yaml` 的两个示例 IP 为实际边/云控制面 IPv4 地址。
   两侧文件内容必须一致，文件的本地绝对路径可以不同。
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

## 过渡端口：不是单端口双向协议

| 用途 | 当前来源 | 示例 |
| --- | --- | --- |
| PRE_OUT（边 → 云控制） | 云 `dp.addr:ctrl_port` | 云 IP:5550 |
| POST_OUT（云 → 边控制/HELLO） | 边 `dp.addr` + 暂留的 POST_OUT export | 边 IP:6454 |
| no-PP 共享世界 TCPStore | 边 `dp.addr` + 默认端口 29600 | 边 IP:29600 |

双方暂时都保留这一行（不设置时默认 5559）：

```bash
export VLLM_ASCEND_LWD_POST_OUT_PORT=6454
```

除此之外，旧传输配置的 PRE_OUT_HOST/PRE_OUT_PORT/POST_OUT_HOST/POST_OUT_BIND/
WIRE_STORE_PORT/HELLO_TIMEOUT_S/DEBUG 环境覆盖链已移除，不再决定传输参数。
不要靠旧 export 改地址或端口；PRE_OUT 改 YAML，TCPStore 本版固定 29600，
HELLO 超时保持 600 秒。其他模块原有的性能、日志和 NPU 环境变量不在本次清理范围。
确认云 5550、边 6454 和边 29600 可达且未被占用。
如果要沿用旧 PRE_OUT=6453，把 YAML 的 ctrl_port 改成 6453，双方使用同一文件。
POST_OUT 不得与边 TCPStore 的 29600 冲突。

## 边 1 卡 / 云 8 卡启动示例

先沿用板上已经可工作的 CANN/NPU、网卡和 PYTHONPATH 环境。下面模型路径、
模型名和可见卡号按实际修改；不要直接套用旧 GLM 1+16 脚本的卡数。
本例使用报错日志里的 Qwen3.8-27B；若用设计文档的 Qwen3.6，则两侧一起更换模型。

边侧（启动后可能等待云侧，两边需在各自终端启动，不要等边侧完全 ready 才启动云）：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_ASCEND_LWD_POST_OUT_PORT=6454
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
export VLLM_ASCEND_LWD_POST_OUT_PORT=6454
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
| `[LWD][config][parsed]` | 文件读取、schema 校验和实例查找已成功；完整保留五块拓扑、所有 DP、缺省 feature 值及过渡 POST_OUT 端口 | `vllm_config.lwd_config`，完整 YAML 在其 `.topology` |
| `[LWD][config][selected]` | 当前 role/instance_id 选中的实例及其全部 DP | `vllm_config.lwd_config.instance` |
| `[LWD][config] path=... TP=...` | 已通过当前单 DP 运行限制及 TP 一致性检查，并已映射并行参数 | `vllm_config.parallel_config` 及其 `.lwd_config` |
| `[LWD][config][transport]` | 实际适配出的 pre_out_host/port、post_out_host/port、wire_store_port、超时等所有传输值 | `LwdConfig.from_vllm_config()` 返回的控制面配置对象 |

`parsed`/`selected` 在运行限制检查之前打印，所以多 DP 文件也能看到完整解析结果，
随后才报运行时暂不支持；不能把这两条日志当成启动成功。
`transport` 在每个进程里对相同配置只打印一次，不在每个请求里重复打印。
原始入口仍在 `vllm_config.additional_config["lwd_config"]`，仅包含 path/role/instance_id。
feature 为 true 仍只是保留参数，不代表对应功能已经启用。

1. 配置日志 `[LWD][config]`：边 role=edge、ranks=(0,)、TP=1、PP=1、world=9；
   云 role=cloud、ranks=(1,...,8)、TP=8、PP=1、world=9；POST_OUT_PORT 均为 6454。
2. 两侧 `[LWD] distributed init method from lwd_config` 都是边 IP:29600。
   并行组日志应显示云内 8 卡 TP、PP 单 rank 组；这只能说明初始化，不代表请求已成功。
3. 边侧 `cloud discovered via HELLO` 的 PRE_OUT 与 YAML 云 IP:ctrl_port 一致；
   `edge engine assembled` 后向边 API 发送请求。若两侧端点配置不同会明确报错。
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
