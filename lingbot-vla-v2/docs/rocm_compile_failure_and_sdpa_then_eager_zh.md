# ROCm 训练无法启动的完整排查记录：编译崩溃 → SDPA 方案卡死 → 关闭编译解决

> **文档定位**：独立、自洽的**全过程说明**。读者不需要看别的文档就能明白"昨天为什么失败、
> 中途试了什么、最后怎么解决的"。
> **时间**：2026-10-09 夜 ~ 2026-10-10 夜（真机连续排查）
> **结论先行**：`configs/rocm/robotwin_official_paths_rocm.yaml` 里 **`train.use_compile: false`**
> ⇒ 训练首次跑通（连续 93+ 步、稳态 10.3 s/it、loss 下降、nmse 降 38–44%、零 OOM）。
>
> 前置相关文档（更偏"崩溃取证"）：`docs/rocm_triton_compile_crash_zh.md`。
> 本文是它的**续集与终章**：那篇停在"根因未 100% 钉死、下一步归属定位"，本文把后续全部走完。

---

## 0. 一页速览

| 阶段 | 做法 | 结果 |
|---|---|---|
| ① 原始配置 | `use_compile: true` + `attention_implementation: flex_cached` | 🔴 **`PassManager::run failed`**，7/7 rank 全崩，训练无法开始 |
| ② 尝试 A：换 SDPA（把 VLM 主干注意力从 flex 换掉） | 新增 `sdpa_attention_forward` + 配置切到 `sdpa` | 🟡 编译崩溃**消失**（进步），但**卡在首步永不推进**（新问题） |
| ③ 尝试 B：定位"卡住" | 逐层取证：显存？锁竞争？图断裂？ | 🟢 找到三个独立原因（见 §4） |
| ④ 尝试 C：per-rank 编译缓存隔离 | 每个 rank 独立 `TORCHINDUCTOR_CACHE_DIR` | 🟡 冻结点从 5846 kernel 推到 **15186**（有效但不够） |
| ⑤ **最终方案：关闭编译** | `train.use_compile: false`（与官方 recipe 一致） | ✅ **训练跑通**：Step 93+、10.3 s/it、零 OOM |

**一句话经验**：在这台 ROCm 机器上，`torch.compile` 有**三重**独立的坑；而**官方 recipe 本来就不开编译**，
所以"关编译"不是妥协，而是**回到官方口径**。

---

## 1. 环境与配置

| 项 | 值 |
|---|---|
| 机型 | AutoDL 8× Radeon PRO W7900D **48 GiB**（GPU[4] **已损坏**，实际用 7 张：`0,1,2,3,5,6,7`） |
| 架构 / 驱动栈 | gfx1100 / ROCm 7.2.1 |
| torch / triton | `torch 2.9.1+rocm7.2.1` / `triton 3.5.1+rocm` |
| Python | `/opt/robotwin-env/bin/python`（3.12） |
| 仓库 | `/workspace/lingbot_vla_2/lingbot-vla-v2` |
| 训练入口 | `experiment/robotwin/al_launch.py`（自动学习启动器）→ `al_50task_bf16.sh` → `tasks/vla/train_lingbotvla.py` |
| 训练配置 | `configs/rocm/robotwin_official_paths_rocm.yaml`（`data_parallel_mode: fsdp2`，`enable_full_shard: false`） |
| AL 配置 | `configs/auto_learning/al_eval2.yaml`（`eval_interval_steps: 50`） |
| 模型 | 6B VLA（`robbyant_lingbot-vla-v2-6b-bf16`），BF16 |

---

## 2. 问题一：编译崩溃（`PassManager::run failed`）

### 2.1 原始报错（7/7 rank 全部失败）

```
RuntimeError: PassManager::run failed
  File "…/triton/backends/amd/compiler.py", line 262, in make_ttgir
```
Triton 自诊断信息指出失败阶段：
```
Pipeline failed while executing [TritonAMDGPUOptimizeDotOperands]
```

### 2.2 定位：崩的是 **flex attention 的反向 kernel**

* 生成的 Inductor 源码里含：
  `def triton_tem_fused_slice_backward_transpose_view_zeros_2(arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, …)`、
  `def bwd_dq_inner`、`def bwd_dq_block_mn`
* 运行日志里有 **7 × `Using Flex Cached (prebuilt BlockMask) Attn`**（只出现在
  `modeling_lingbot_vla_v2.py:434`（`get_attention_interface()` 的 `flex_cached` 分支），即 **VLM 主干**）
  与 21 × `Using Eager Attn`（动作专家 pi0）：
  ⇒ **调用方是 VLM 主干**，不是动作专家、不是视频塔、不是 deepstack。
* 排除法（每条都实测过）：
  * 12 组 `num_warps × num_stages` 配置**全部失败** ⇒ 不是某个编译参数；
  * 把视频塔切到 `sdpa_block_causal` 无效 ⇒ 视频塔不是调用方；
  * 给 `_deepstack_process` 加 `@torch.compiler.disable` 无效 ⇒ deepstack 不是成因；
  * 缓存里"缺 ttgir"的 kernel 恒为 **1 个**（即这个 flex 反向 kernel）。
* **报错与"开编译"强绑定**：官方 recipe 是 `use_compile: false`，官方 `train_full_sft.sh`
  从不传 `--train.use_compile` ⇒ **官方根本不会触发这个 kernel**。

### 2.3 尝试 A：把 VLM 主干的注意力换成 SDPA

思路：flex 反向 kernel 是崩溃源头 ⇒ 换一条**数值等价但不走 flex** 的实现。

改动：
1. `lingbotvla/models/vla/lingbot_vla/flex_attention.py` 新增 `sdpa_attention_forward(...)`
   （`@torch.compiler.disable`；bool mask `True=keep` 直接传给 `[B,1,Q,KV]`；GQA 用
   `repeat_interleave`；Q/K/V 升 fp32；输出 `[B,L,H*D]`）；
2. `modeling_lingbot_vla_v2.py` 的 `get_attention_interface()` 增加 `"sdpa"` 分支；
3. `configs/rocm/robotwin_official_paths_rocm.yaml`：`attention_implementation: sdpa`。

**数值对拍（真实形状 d=256 / GQA 8:1 / seq 320）**：前向 ≤ **2.33e-06**、反向梯度 ≤ **3.46e-06**
⇒ 在数值上与 flex 路径等价。真机单测 **16 passed**。

### 2.4 结果：编译崩溃**确实消失了**

```
PassManager::run failed = 0        ← 多轮训练日志里全部为 0
[kernel 缓存] 缺 ttgir 的 kernel 数 = 1（与基线相同，未新增）
运行日志: UserWarning: Using AOTriton backend for Efficient Attention forward
          out = F.scaled_dot_product_attention(…)
```
⇒ **SDPA 生效**，且 VLM 与视频塔都在真实执行 SDPA。**但训练仍然起不来**（见问题二）。

---

## 3. 问题二：训练"卡在首步"（新问题）

### 3.1 现象（多轮复现）

```
Step: 0/5000        （永不前进；无 loss 行）
GPU 使用率: 0%（7 张可用卡全闲）
rank 进程 CPU: 各 36–50%（在烧 CPU 但不产出）
日志: 静默 6–7 分钟，最后一行是 Dynamo 的 lru_cache / graph break 警告
崩溃计数: PassManager = 0、OutOfMemoryError = 0     ← 不是崩溃，是卡死
```

### 3.2 关键判据（怎么区分"慢"与"卡死"）

用 `rocm-smi` + `/proc` + 编译缓存目录做**时间序列采样**，而不是"等一会儿看看"：

| 采样对象 | 正常推进 | 卡死 |
|---|---|---|
| triton 缓存 `*.source` 文件数 | 持续增长（实测 ~19 个/s） | **停滞**（45 秒新增 0） |
| torchinductor 文件数 | 增长 | 冻结 |
| 编译 worker 的 CPU 总量 | 与产物同步增长 | **烧 ~7.8 个核却零产物**（决定性判据） |
| GPU 使用率 | 有占用 | 0% |

**实测冻结点**：`al_v28` 停在 **4620** kernel、`al_v29` 停在 **5846** kernel，之后永久不动。

---

## 4. 问题二的三层根因（按发现顺序，每层都有实测证据）

### 4.1 第一层：7 个 rank **共享**同一个编译缓存目录 ⇒ 锁竞争死锁

* 症状：231 个 Inductor 编译 worker 存在，合计 **35,105 ticks / 45 秒 ≈ 7.8 个核满负荷**，
  但 triton 缓存 **45 秒内零新增文件**、torchinductor 文件数冻结、GPU 0%。
* 修法（**已实现并保留**）：`tasks/vla/train_lingbotvla.py` 新增
  `_al_per_rank_compile_cache()`，在 import 之后把 `LOCAL_RANK` 拼到
  `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` 末尾（`…/rank<N>`）。
  * 为什么放在训练脚本而不是启动器：启动器用 `shlex.quote` 渲染 `env KEY=VALUE`，
    **shell 变量在那层不会展开**（实测 dry-run 渲染出的是基础路径）。
  * 逃生开关：`AL_SHARED_COMPILE_CACHE=1` 恢复共享（仅排障）。
* 效果：冻结点 **5846 → 15186** kernel（编译确实走得更远），**但仍会卡** ⇒ 不是根因。

### 4.2 第二层：显存不足（被排除，但要知道边界）

* `micro 12 / gas 1`：**OOM** —— 报错发生在 FSDP 反向的 `reduce_scatter` 缓冲分配：
  ```
  torch.OutOfMemoryError: HIP out of memory. Tried to allocate 2.02 GiB.
  GPU 6 has a total capacity of 47.98 GiB of which 424.00 MiB is free.
  栈: fsdp/_fully_shard/_fsdp_param_group.py::post_backward → foreach_reduce → reduce_scatter_comm.allocate
  ```
* `micro 10`：同样 OOM；`micro 8`：峰值 **49,092 / 49,121 MiB（99.9%）**，Step 1 后 OOM。
* ⇒ **降 micro 几乎省不下显存**（实测 8→6→5 峰值 49,092→47,938→48,956 MiB）：
  显存大头是**与 batch 无关的固定占用**（参数 + 优化器状态 + 深度/未来视频对齐模型 + FSDP 缓冲）。
  降 micro 省下的是**时间**（14.1 → 10.3 s/it）。

### 4.3 第三层（真因）：**图断裂 ⇒ 7 rank 集合通信错序 ⇒ RCCL 死锁**

* 证据：训练日志里出现
  ```
  File "…/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 882,
    in torch_dynamo_resume_in__deepst…
  [compile] dynamo cache_size_limit 8 → 64
  ```
  即 **Dynamo 确实在编译**，且**断点打在 `_deepstack_process`**（为了绕开别的坑，我们给它和
  SDPA 都加了 `@torch.compiler.disable`）。
* 机制：FSDP 下 7 个 rank 的图**必须一致**；一旦某 rank 与其他 rank 的断点/图不同，
  集合通信（all-gather / reduce-scatter）**顺序错位** ⇒ 互相等待 ⇒ **死锁**（表现为 GPU 0%、
  CPU 空转、日志静默 —— 与 §3.1 完全吻合）。
* 决定性事实：**本配置 `use_compile: true`，而官方 recipe 是 `false`**：
  ```
  configs/rocm/robotwin_official_paths_rocm.yaml:95   use_compile: true    ← 我们
  /RoboTwin/experiments/…/lingbotvla_cli.yaml          use_compile: false   ← 官方
  experiment/robotwin/start_robotwin_*.sh              use_compile=False    ← 本仓推理脚本默认
  ```

---

## 5. 最终方案：关闭编译（并保留已修好的两处）

```yaml
# configs/rocm/robotwin_official_paths_rocm.yaml
train:
  use_compile: false      # ← 由 true 改回 false
```

**为什么这是正解而不是妥协**

| 理由 | 说明 |
|---|---|
| 与官方 recipe 同口径 | 官方从不开编译；本仓推理脚本默认也是 `False` |
| 编译在本环境有**三重**前科 | ① flex 反向 kernel 崩溃；② 跨 rank 共享缓存死锁；③ 图断裂死锁 |
| `@torch.compiler.disable` 补丁自动退化为 no-op | 没有编译图就没有断裂，**无需**移除补丁 |
| SDPA 那条路**依然有效** | 它修掉的是"flex 反向 kernel 崩溃"，在 eager 下同样生效，且数值对拍通过 |

**保留的改动（将来若实验性开编译仍然需要）**

1. `AL_VLM_ATTENTION=<flex|flex_cached|sdpa|eager>` 逃生舱（`modeling_lingbot_vla_v2.py`）
   —— 单变量 A/B 与崩溃归因；
2. `_al_per_rank_compile_cache()` —— per-rank 编译缓存隔离；
3. `--compute-fingerprint`（默认关）、显式缓存文件 + 模型名 —— 缓存不再因改代码失效。

---

## 6. 解决后的实测结果

### 6.1 训练（`al_v34`，2026-10-10 夜；`micro 5 / gas 1` / GBS 35 / 7 卡 / 关编译）

```
Step: 93/5000 [20:44<14:01:36, 10.29s/it]
OutOfMemoryError = 0     SIGABRT = 0     PassManager = 0
各卡已用(MiB): 48100 48109 48107 48105 [33] 48111 48099 47810   ← 稳定不涨
```

| 指标 | 实测 |
|---|---|
| 首步（模型加载 + 首次前反向 + FSDP 初始化） | **~150 s**（正常，不是卡死） |
| 稳态 | **10.3–11.9 s/it** ⇒ ≈327 步/小时 |
| 5000 步 | 纯训练 ≈15.3 h；含每 50 步评测 ≈21.5 h |
| 峰值显存 | **48,956 / 49,121 MiB（99.7%）** |
| GPU 利用率 | 95–100% |
| 评测开销 | 每次**阻塞约 60 s**（Step 49→50 的 s/it 由 10.67 跳到 27.20） |

### 6.2 效果（同一任务 `place_dual_shoes`，训练前 vs 50 步后）

| split | 训练前 nmse | 50 步后 nmse | 变化 |
|---|---|---|---|
| train | 1.4227 | **0.7994** | **−43.8%** |
| val | 1.4235 | **0.8872** | **−37.7%** |

⇒ nmse 已 **< 1.0**（优于 baseline），train/val 同步下降 ⇒ 暂无过拟合。
⚠️ baseline 是按旧配置算的（日志警告 config 指纹不一致）⇒ **趋势可信，绝对 PASS 门槛暂不可信**。

### 6.3 缓存收益（同期改造，与编译问题无关但同批交付）

| 项 | 实测 |
|---|---|
| scout 缓存（显式文件 + 模型名） | 50/50 命中，预检秒过（启动 ~1.5 min） |
| hardness 缓存 | **208.5 s → 4.2 s（约 44×）**，日志 `本次新增 0` 证明未重算 |
| 指纹计算默认跳过 | 不再哈希 11.9 GiB 权重分片（每轮省 1–3 min I/O） |

---

## 7. 复现与验证步骤（照做即可）

```bash
# 0) 确认无残留训练进程（否则启动器以退出码 2 自锁拒绝 —— 这是保护，不是 bug）
ps -eo args | grep "[t]rain_lingbotvla.py /workspace"      # 必须为空
rocm-smi --showmeminfo vram | grep "Used Memory"           # 各卡应 ~25 MiB 基线

# 1) 确认编译已关（必须为 false）
grep -n "use_compile" configs/rocm/robotwin_official_paths_rocm.yaml

# 2) 起训练（7 卡，排除坏卡 GPU[4]）
/opt/robotwin-env/bin/python -u experiment/robotwin/al_launch.py \
  --run-name al_vNN --steps 5000 --micro 5 --gas 1 \
  --hardness-cache-file /workspace/al/hardness_cache/hardness.json \
  --scout-cache-file    /workspace/al/scout_cache/scout.json \
  --model-name          robbyant_lingbot-vla-v2-6b-bf16

# 3) 判定"成功"的日志标志
grep -a "Using SDPA Attn"                 …/logs/train_al_vNN.log   # VLM 走 SDPA（7 处）
grep -a "Step: 1/5000"                    …/logs/train_al_vNN.log   # 首步成功（此前卡点）
grep -ac "PassManager::run failed"        …/logs/train_al_vNN.log   # 必须为 0
```

**如果再遇到"卡在 Step 0"，按这个顺序查（每步都有实测判据）**：

1. `grep -c "PassManager::run failed"` ⇒ 非 0 说明又开编译了（检查 `use_compile`）；
2. 采样编译缓存文件数 60 秒 ⇒ **不增长** + worker 烧 CPU ⇒ 编译阶段死锁（检查 per-rank 隔离是否生效）；
3. `grep -a "torch_dynamo_resume_in"` ⇒ 有 ⇒ 编译开着且存在图断裂；
4. 若以上都正常却仍卡 ⇒ 查显存峰值与 FSDP `reduce_scatter` 是否 OOM。

---

## 8. 附：本次排查踩过的操作坑（避免重犯）

| # | 坑 | 正确做法 |
|---|---|---|
| 1 | 用 `pgrep -f "train_lingbotvla.py /workspace"` 扫 `/proc`，**匹配到了执行它的那条 shell**（命令行文本含该串）⇒ 量到的是 `bash` 在 `do_wait`、CPU=0，得出"进程没在算"的**错误结论** | 按 `argv[2] == 'tasks/vla/train_lingbotvla.py'` 精确判定 |
| 2 | 为"抓栈"发 `kill -ABRT`，**误杀了刚启动的训练**（rank5 RCCL watchdog SIGABRT，`what(): HIP error: unknown error`） | 清理/杀进程**一律按显式 PID 列表**；**新 run 启动期间绝不清理** |
| 3 | 直觉认为"降 batch 能省显存"，连试 micro 12/10/8 三轮 | **先量再改**：实测降 micro 几乎不省显存（固定占用主导），省的是时间 |
| 4 | 本地 `--dry-run` 能过，真机却 `NameError`/`AttributeError` | 本地 dry-run 在"缺 `/workspace` 数据"处**提前返回**，走不到那些分支 ⇒ 用 **pyflakes 静态审计**（`tests/test_no_undefined_names.py`）补上 |

---

## 9. 结论

1. **"编译报错"的根因**：flex attention 的反向 Triton kernel 在 gfx1100 的 AMD pass 上崩溃
   （`PassManager::run failed`），**只在开编译时触发**；
2. **"换 SDPA"的收益与代价**：崩溃消失了（数值对拍 1e-06 量级），但**没有解决训练启动**，
   因为更底层的"卡死"来自**开编译本身**（跨 rank 缓存锁竞争 + 图断裂导致集合通信错序）；
3. **最终解法**：`train.use_compile: false` —— 与官方 recipe 一致；
4. **实测结果**：训练连续跑到 Step 93+、稳态 10.3 s/it、零 OOM，评测 nmse 相对基线下降 38–44%。
