#!/usr/bin/env python3
"""lwd_sim_control_plane: 控制面端到端模拟(无 NPU/torch 环境)。

stub 掉重依赖(Request/SamplingParams/tensor/原生 schedule)后,把仓库
真实代码——云引擎消息处理(_lwd_on_edge_msg/_lwd_handle_*/build_request/
handle_model_output/C2e 拆发)、云调度器(分队列/轮转/点名)、通信层
(ROUTER/DEALER/loop)——用真实 ZMQ TCP 跑起来,覆盖:
  S1 单请求全链(register→Request→gate→registry→调度→C2e 还原)
  S2 多请求并发   S3 重复 RangeNotify 幂等   S4 abort 双时点
  S5 控制层来源隔离 + 拒绝混边 DOWN carrier（不验证多实例运行）
  S6 未注册来源拒收   S7 digest 互校拒绝
  S8 真调度点名行(_lwd_schedule_for_visible_reqs 的 requests[rid_key])

注意:必须独立进程运行(预注入 fake 模块,不可 import 进其他测试)。
用法: python3 tools/lwd_sim_control_plane.py   (退出码 0/1)
上板暴露的 KeyError/publish 改名一类问题,此模拟均可拦截。"""
import sys, types, logging, queue, time, enum
from types import SimpleNamespace
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
PORT = 15700

# ---------------- stub 重依赖(全部与被测逻辑无关) ----------------
def mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m

fake_vllm = mod("vllm"); fake_vllm.__path__ = [f"{REPO}/vllm"]
_lg = logging.getLogger("sim"); _lg.setLevel(logging.WARNING)
class _L:
    def __getattr__(self, n): return getattr(_lg, n)
    def info_once(self, m, *a): pass
mod("vllm.logger", init_logger=lambda n: _L())
t = mod("torch"); t.zeros = lambda *a, **k: None
class _FinishReason(enum.IntEnum):
    STOP=0; LENGTH=1; ABORT=2; ERROR=3; REPETITION=4
class _ReqType(enum.Enum):
    ADD=0; ABORT=1; WAKEUP=2; EXECUTOR_FAILED=3
mod("vllm.sampling_params", SamplingParams=lambda **kw: SimpleNamespace(**kw, update_from_generation_config=lambda *a: None))
v1 = mod("vllm.v1"); v1.__path__ = [f"{REPO}/vllm/v1"]
mod("vllm.v1.engine", EngineCoreRequestType=_ReqType, FinishReason=_FinishReason, EngineCoreOutputs=object)
mod("vllm.v1.engine.core", EngineCoreProc=type("EngineCoreProc", (), {}))
class _Request:
    def __init__(self, **kw): self.__dict__.update(kw)
mod("vllm.v1.request", Request=_Request)
mod("vllm.v1.core", __path__=[f"{REPO}/vllm/v1/core"])
mod("vllm.v1.core.kv_cache_utils", resolve_kv_cache_block_sizes=lambda *a: (None, 16))
mod("vllm.v1.outputs", LwdC2eMeta=SimpleNamespace)
# sched 链:fake 到可承载轮转测试的程度
SO = type("SchedulerOutput", (), {})
mod("vllm.v1.core.sched")
mod("vllm.v1.core.sched.output", SchedulerOutput=SO, LwdBatch=SimpleNamespace,
    LwdBatchType=SimpleNamespace(LWD_EMBED=0), LwdEmbedBatch=SimpleNamespace)
mod("vllm.v1.core.sched.async_scheduler", AsyncScheduler=type("AsyncScheduler", (), {}))
class FakeQueue(list):
    def remove_requests(self, reqs):
        for r in reqs: self.remove(r)
    def extend(self, reqs): list.extend(self, reqs)
    def prepend_requests(self, reqs): self[0:0] = reqs
    def add_request(self, r): self.append(r)
mod("vllm.v1.core.sched.request_queue", RequestQueue=object, create_request_queue=lambda p: FakeQueue())
class _Sentinel(Exception): pass
class _AsyncSched:
    def schedule(self): raise _Sentinel()
mod("vllm.v1.core.sched.async_scheduler", AsyncScheduler=_AsyncSched)
lc = mod("vllm.v1.lwd_control", __path__=[f"{REPO}/vllm/v1/lwd_control"])
mod("vllm.v1.lwd_control.control_communication", __path__=[f"{REPO}/vllm/v1/lwd_control/control_communication"])
mod("vllm.v1.lwd_control.control_cloud_scheduler", __path__=[f"{REPO}/vllm/v1/lwd_control/control_cloud_scheduler"])
mod("vllm.v1.lwd_debug", LwdDebug=SimpleNamespace(
    cloud_request_admitted=lambda *a: None, cloud_step=lambda *a: None,
    edge_tokens_delivered=lambda *a: None), LwdLogBase=SimpleNamespace(set_debug=lambda a: None))
# assemble 依赖 config.lwd(3.10 语法),fake 之(投影已单测过,模拟不经它)
mod("vllm.config.lwd", LWD_WIRE_STORE_PORT_DEFAULT=29600)
mod("vllm.v1.lwd_control.control_edge_scheduler",
    __path__=[f"{REPO}/vllm/v1/lwd_control/control_edge_scheduler"])
mod("vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble", LwdConfig=SimpleNamespace)
mod("vllm.v1.lwd_control.control_scheduler",
    __path__=[f"{REPO}/vllm/v1/lwd_control/control_scheduler"])

sys.path.insert(0, REPO)
import zmq
from vllm.v1.lwd_control.control_communication.lwd_control_communicator import LwdControlCommunicator
from vllm.v1.lwd_control.control_communication.lwd_control_loop import LwdControlLoop
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LwdRegisterNotify, LwdRegisterAckNotify, LwdRequestNotify, LwdRangeNotify,
    LwdAbortNotify, LwdC2eNotify, LWD_WIRE_VERSION, lwd_decode_notify,
    lwd_decode_cloud_notify, lwd_encode_notify, lwd_encode_cloud_notify)
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_phase_scheduler import (
    LwdCloudPhaseScheduler)
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
    LwdCloudEngineCore, _lwd_rid_key, _lwd_rid_split)

# ---------------- 装配:真实类 + 裸实例 ----------------
def make_cloud_engine():
    eng = object.__new__(LwdCloudEngineCore)
    eng._lwd_config = SimpleNamespace(instance_id=0, edge_npu_count=1,
                                      cloud_npu_count=8, topology_digest="dig0001",
                                      my_links=((0, 0, 0), (1, 0, 0), (9, 0, 0)))
    # Synthetic control-only fixture, not a runnable multi-instance topology.
    eng._lwd_peers = {}; eng._lwd_peer_ids = {}
    eng._lwd_gate_pending = {}; eng._lwd_seqno_registry = {}
    eng.input_queue = queue.Queue(); eng.aborts_queue = queue.Queue()
    eng.request_block_hasher = None
    eng.vllm_config = SimpleNamespace(model_config=SimpleNamespace(
        get_hidden_size=lambda: 4096, dtype="bfloat16"))
    eng.scheduler = object.__new__(LwdCloudPhaseScheduler)
    eng.scheduler.prefill_notify_queue = {}
    eng.scheduler._lwd_rr_cursor = 0
    eng.scheduler.lwd_seqno_registry = eng._lwd_seqno_registry
    return eng

def boot_cloud(eng):
    router = LwdControlCommunicator(f"tcp://127.0.0.1:{PORT}", zmq.ROUTER, bind=True)
    eng._lwd_io = LwdControlLoop({None: router}, routing=True,
                                 decoder=lwd_decode_notify, encoder=lwd_encode_cloud_notify,
                                 on_msg=eng._lwd_on_edge_msg)
    eng._lwd_io.start()

def boot_edge(identity, links):
    dealers = {lk: LwdControlCommunicator(f"tcp://127.0.0.1:{PORT}", zmq.DEALER,
                                          bind=False, identity=identity)
               for lk in links}
    box = {"c2e": [], "ack": [], "nack": {}}
    def on_msg(link, _id, msg):
        if isinstance(msg, LwdRegisterAckNotify): box["ack"].append((link, msg))
        elif isinstance(msg, LwdC2eNotify): box["c2e"].append((link, msg))
        else: box["nack"][type(msg).__name__] = box["nack"].get(type(msg).__name__, 0) + 1
    loop = LwdControlLoop(dealers, decoder=lwd_decode_cloud_notify,
                          encoder=lwd_encode_notify, on_msg=on_msg)
    loop.start()
    return loop, box

def register(loop, box, links, edge_id, dp=0, **over):
    for lk in links:
        loop.send(lk, LwdRegisterNotify(edge_id=edge_id, cloud_id=lk[1], dp_idx=lk[2],
                                        wire_version=LWD_WIRE_VERSION, edge_npu_count=1,
                                        cloud_npu_count=8, topology_digest="dig0001", **over))
    deadline = time.time() + 5
    got = {lk for lk, _ in box["ack"]}
    while time.time() < deadline and not got >= set(links):
        time.sleep(0.02); got = {lk for lk, _ in box["ack"]}
    return got >= set(links)

def drain(eng, seconds=0.6):
    """等云侧 IO 线程消化完消息(input_queue/aborts/loop 队列全静)。"""
    time.sleep(seconds)

def take_adds(eng):
    out = []
    while not eng.input_queue.empty():
        kind, payload = eng.input_queue.get_nowait()
        out.append((kind, payload))
    return out

class _Pinned(list):
    def tolist(self): return list(self)

def make_carrier(req_ids, seqno, numel=64, connection_key=(0, 0, 0)):
    n = len(req_ids)
    pinned = _Pinned(list(range(n)) + [1] * n + [1] * n)  # ranks/counts/seg_lens
    return (pinned, list(req_ids), numel, seqno, connection_key)

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

# ================= S1 单请求全链 =================
eng = make_cloud_engine(); boot_cloud(eng)
eloop, ebox = boot_edge(b"edge-0-0", [(0, 0, 0)])
check("S1 register/ack", register(eloop, ebox, [(0, 0, 0)], 0))
rid = "req-single-1"
eloop.send((0,0,0), LwdRequestNotify(request_id=rid, num_prompt_tokens=11,
                                     max_tokens=50, edge_id=0, dp_idx=0))
for i in range(3):
    eloop.send((0,0,0), LwdRangeNotify(request_id=rid, offset=i*4, num_tokens=4,
                                       seqno=i, edge_id=0, dp_idx=0))
drain(eng)
adds = take_adds(eng)
check("S1 gate ADD 一条且键带前缀", len(adds) == 1 and adds[0][1][0].request_id == f"0#0#{rid}",
      str([a[1][0].request_id for a in adds if a[0].name == "ADD"]))
check("S1 seqno registry 0,1,2", eng._lwd_seqno_registry[f"0#0#{rid}"] == [0, 1, 2])
sched = eng.scheduler
order = [sched._lwd_next_range().seqno for _ in range(3)]
check("S1 调度取队 seqno 顺序", order == [0, 1, 2], str(order))
# C2e 回程(真实 handle_model_output 路径)
eng.lwd_handle_model_output(
    SimpleNamespace(lwd_down_carrier=make_carrier([f"0#0#{rid}"], 7)),
    {0: SimpleNamespace(outputs=[], finished_requests=None)})
deadline = time.time() + 5
while time.time() < deadline and not ebox["c2e"]: time.sleep(0.02)
check("S1 C2e 到边", len(ebox["c2e"]) == 1)
link, c2e = ebox["c2e"][0]
check("S1 C2e rid 还原 + edge/dp/seqno", c2e.req_ids == [rid] and c2e.edge_id == 0
      and c2e.dp_idx == 0 and c2e.down_seqno == 7 and link == (0, 0, 0),
      f"{c2e.req_ids} seq={c2e.down_seqno}")

# ================= S2 多请求并发 =================
rids = ["req-m-1", "req-m-2", "req-m-3"]
for r in rids:
    eloop.send((0,0,0), LwdRequestNotify(request_id=r, num_prompt_tokens=8, edge_id=0, dp_idx=0))
    eloop.send((0,0,0), LwdRangeNotify(request_id=r, offset=0, num_tokens=8, seqno=0, edge_id=0, dp_idx=0))
drain(eng)
adds = take_adds(eng)
check("S2 三条 ADD 各带前缀", sorted(a[1][0].request_id for a in adds) ==
      sorted(f"0#0#{r}" for r in rids))
# 三请求同批 decode:C2e 拆发后单边一条,rid 全还原
eng.lwd_handle_model_output(
    SimpleNamespace(lwd_down_carrier=make_carrier([f"0#0#{r}" for r in rids], 8)),
    {0: SimpleNamespace(outputs=[], finished_requests=None)})
deadline = time.time() + 5
while time.time() < deadline and len(ebox["c2e"]) < 2: time.sleep(0.02)
last = ebox["c2e"][-1][1]
check("S2 C2e 合组回发 rid 全还原", sorted(last.req_ids) == sorted(rids), str(last.req_ids))

# ================= S3 重复 RangeNotify(队满重发) =================
ebox["c2e"].clear()
before = eng._lwd_seqno_registry[f"0#0#req-m-1"]
eloop.send((0,0,0), LwdRangeNotify(request_id="req-m-1", offset=0, num_tokens=8,
                                   seqno=0, edge_id=0, dp_idx=0))  # 重复同号
drain(eng)
check("S3 registry 幂等(重复同号不登记)",
      eng._lwd_seqno_registry[f"0#0#req-m-1"] == before == [0])

# ================= S4 abort 双时点 =================
eloop.send((0,0,0), LwdAbortNotify(request_id="req-m-2", edge_id=0, dp_idx=0))
drain(eng)
aborts = []
while not eng.aborts_queue.empty(): aborts.append(eng.aborts_queue.get_nowait())
check("S4 abort 走前缀键入双队列", aborts == [["0#0#req-m-2"]], str(aborts))
pending = list(eng._lwd_gate_pending)
eloop.send((0,0,0), LwdRequestNotify(request_id="late-abort", num_prompt_tokens=4,
                                     edge_id=0, dp_idx=0))
eloop.send((0,0,0), LwdAbortNotify(request_id="late-abort", edge_id=0, dp_idx=0))
drain(eng)
check("S4 abort 清 gate(请求未促发即终止)", "0#0#late-abort" not in eng._lwd_gate_pending)
take_adds(eng)  # 清场

# ================= S5 双边 fan-in(s3 场景) =================
eloop2, ebox2 = boot_edge(b"edge-1-0", [(1, 0, 0)])
check("S5 边1 register", register(eloop2, ebox2, [(1, 0, 0)], 1))
# 两边各自积压:边0 两连号,边1 两连号,交错投入
eng.scheduler.prefill_notify_queue.clear(); eng.scheduler._lwd_rr_cursor = 0
for r, sq in (("e0-a", 0), ("e0-b", 1)):
    eloop.send((0,0,0), LwdRangeNotify(request_id=r, offset=0, num_tokens=4, seqno=sq, edge_id=0, dp_idx=0))
for r, sq in (("e1-a", 0), ("e1-b", 1)):
    eloop2.send((1,0,0), LwdRangeNotify(request_id=r, offset=0, num_tokens=4, seqno=sq, edge_id=1, dp_idx=0))
drain(eng)
seq = [sched._lwd_next_range() for _ in range(4)]
owners = [(n.request_id.split("#")[0], n.seqno) for n in seq]
check("S5 边间不插队(排空一边再换)", owners == [("0",0),("0",1),("1",0),("1",1)]
      or owners == [("1",0),("1",1),("0",0),("0",1)], str(owners))
# 同号 seqno 两边不撞(registry 隔离)
check("S5 两边同号 seqno 不撞",
      eng._lwd_seqno_registry["0#0#e0-a"] == [0] and eng._lwd_seqno_registry["1#0#e1-a"] == [0])
# 混边 hidden 未拆包时禁止仅拆控制通知。多实例执行仍不在本轮范围。
ebox["c2e"].clear(); ebox2["c2e"].clear()
try:
    eng.lwd_handle_model_output(
        SimpleNamespace(lwd_down_carrier=make_carrier(
            ["0#0#e0-a", "1#0#e1-a", "0#0#e0-b", "1#0#e1-b"], 9)),
        {0: SimpleNamespace(outputs=[], finished_requests=None)})
except ValueError as exc:
    check("S5 拒绝未拆张量的混边通知", "matching DOWN packet" in str(exc))
else:
    check("S5 拒绝未拆张量的混边通知", False)
check("S5 未发送混边 C2e", not ebox["c2e"] and not ebox2["c2e"])

# ================= S6 未注册来源拒收 =================
ghost_loop, gbox = boot_edge(b"ghost-7-7", [(7, 0, 0)])
ghost_loop.send((7,0,0), LwdRequestNotify(request_id="evil", num_prompt_tokens=4,
                                          edge_id=7, dp_idx=0))
drain(eng)
check("S6 未注册来源不促发 ADD", not any(
    getattr(p, "request_id", "") == "7#0#evil" for _, p in take_adds(eng)))
ghost_loop.stop()

# ================= S7 互校拒绝(digest 不符) =================
bad_loop, bbox = boot_edge(b"edge-9-0", [(9, 0, 0)])
for lk in [(9, 0, 0)]:
    bad_loop.send(lk, LwdRegisterNotify(edge_id=9, cloud_id=0, dp_idx=0,
                                        wire_version=LWD_WIRE_VERSION, edge_npu_count=1,
                                        cloud_npu_count=8, topology_digest="WRONG"))
time.sleep(0.8)
check("S7 digest 不符不回 ack", len(bbox["ack"]) == 0 and b"edge-9-0" not in eng._lwd_peers)
bad_loop.stop()

# ================= S8 真调度点名路径(上板 KeyError 的原位置) =================
class _Sentinel(Exception): pass
class _Req:
    def __init__(self): self.request_id = "0#0#name-only"
    def __hash__(self): return hash(self.request_id)
    def __eq__(self, o): return isinstance(o, _Req)
req_obj = _Req()
eng.scheduler.requests = {"0#0#name-only": req_obj}
eng.scheduler.running = []; eng.scheduler.waiting = FakeQueue([req_obj])
eng.scheduler.skipped_waiting = FakeQueue()
eng.scheduler.max_num_scheduled_tokens = 8192
eng.scheduler.policy = None
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import LwdBaseScheduler
try:
    eng.scheduler._lwd_schedule_for_visible_reqs(["0#0#name-only"], token_budget_cap=4)
    hit = "NO-SENTINEL"   # 原生 schedule 未短路,不应发生
except _Sentinel:
    hit = "OK"            # 点名行通过,进入原生调度即哨兵
except KeyError as exc:
    hit = f"KEYERROR:{exc}"  # 修复前在这里炸
check("S8 调度点名 requests[rid_key] 命中", hit == "OK", hit)

eloop.stop(); eloop2.stop(); eng._lwd_io.stop(); time.sleep(0.3)

fails = [n for n, ok, _ in RESULTS if not ok]
print(f"\n==== 模拟结果: {len(RESULTS) - len(fails)}/{len(RESULTS)} PASS ====")
if fails:
    print("FAIL 项:", fails); sys.exit(1)
print("全部场景通过")
