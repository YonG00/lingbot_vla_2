# 说明：ROCm 上 `torch.compile` 的 Triton AMDU 后端崩溃 —— 现象、根因、解决思路

> 日期：2026-10-10　机器：Radeon Cloud `cpu1`，7×Radeon PRO W7900D 48G（gfx1100），ROCm 7.2.1
> 状态：**根因已定位**（Triton 自身 pass bug）；**处置方案待定**（见 §6）
> 相关记忆：`knowledge/perf.md`（编译成本）、`knowledge/training.md`（显存基准）

---

## 1. 现象

在 7 卡 FSDP2 上跑 AutoLearning 正式训练（`micro=12 / gas=1 / GBS=84`），**第一个 learning unit 的难度扫描全部跑完、`TrainRequest` 已发布之后**，进入第一步真训练：

```
loss.backward()
→ Inductor 为 backward 生成 kernel
→ RuntimeError: PassManager::run failed        （7/7 rank，确定性）
```

关键特征（三条都实测）：

| 特征 | 数值 | 含义 |
|---|---|---|
| 失败 rank 数 | **7/7** | 确定性，不是偶发 |
| 崩溃发生的编译路径 | **主编译 + 异步预热（`warm_cache_only=True`）两条都崩** | 与"预热可选"无关 |
| 全缓存 kernel 数 | **2652 个 `.ttir`，2651 个 `.ttgir`** | **只有 1 个 kernel 编不过**，其余 2651 个正常 ⟹ 不是环境坏了 |

失败的那个 kernel（由"唯一缺 ttgir 的目录"反查出）：

```
name  : triton_tem_fused_slice_backward_transpose_view_zeros_2
型号  : @triton_heuristics.template(num_stages=2, num_warps=4)
常量  : BLOCK_M=32 BLOCK_N=64 / BLOCK_M1=N1=16 / BLOCK_M2=N2=16
        BLOCKS_ARE_CONTIGUOUS=False / HAS_FULL_BLOCKS=True / SPARSE_Q_BLOCK_SIZE=128
指针  : 全部 *fp32；参数含 Q/K/V/LSE/DELTA/DO/KV_NUM_BLKS/KV_IDX/Q_NUM_BLKS
归属  : flex-attention 反向（slice/transpose/view/zeros 融合而来）
```

---

## 2. 证据链（可复核）

1. **失败点定位**（日志）：`triton/backends/amd/compiler.py:262 make_ttgir → pm.run(mod)`。
2. **缓存取证**：`find <triton_cache> -name '*.ttir'` 与 `-name '*.ttgir'` 计数差 1，逐目录对比找出唯一缺失目录。
3. **最小复现**（本文件 §7 的 demo 代码）：把该 `.ttir` 直接喂给 `triton.compile`，得到 Triton 自己的诊断：

```
error: Failures have been detected while processing an MLIR pass pipeline
note: Pipeline failed while executing [`TritonAMDGPUOptimizeDotOperands` on 'builtin.module' operation]:
      reproducer generated at `std::errs, please share the reproducer above with Triton project.`
```

4. **参数扫描**：`num_warps ∈ {1,2,4,8} × num_stages ∈ {1,2,3}` **12 组全部失败** ⟹ 调参绕不过。
5. **pipeline 位置**（复现器里的 pass 序列，崩溃点加粗）：

```
tritongpu-coalesce
→ tritongpu-remove-layout-conversions
→ tritongpu-optimize-thread-locality
→ tritonamdgpu-accelerate-matmul{arch-generation-name=gfx1100 kPack=1 matrix-instruction-size=0}
→ tritongpu-remove-layout-conversions
→ tritonamdgpu-optimize-epilogue
→ **tritonamdgpu-optimize-dot-operands{arch-generation-name=gfx1100}**   ← 崩在这里
→ …
```

---

## 3. 根因

**Triton 的 AMDGPU 后端 pass `OptimizeDotOperands` 在处理这个 kernel 的 TTGIR 时自身出错**，
导致 `pm.run()` 抛 `PassManager::run failed`。

- 与我们模型的 Python 代码无关（"Using Flex/Eager Attn" 之类的打印只是模块内另一处同名方法的输出，会误导）；
- 与显存无关（同一环境下 2651 个 kernel 编译正常）；
- 与 `num_warps/num_stages/autotune` 无关（12 组全灭）；
- Triton 自己建议"把 reproducer 交给 Triton 项目"，即**上游 bug**。

---

## 4. 已排除的三条错路（省得后来者重试）

| 尝试 | 结果 | 原因 |
|---|---|---|
| 清空 Triton 缓存重编 | **无效** | 失败是确定性的，不是缓存损坏 |
| `TORCH_COMPILE_DISABLE=1` 整个关掉编译 | **绕过了编译，但换来 OOM** | eager 少了算子融合，峰值从 36.9 GiB 涨到 **46.7 GiB** ⇒ 48G 卡爆（`HIP out of memory`）。**编译不只是速度，也是显存** |
| 把视频塔 `attention_mode: flex_block_causal → sdpa_block_causal` | **无效** | 配置确实生效（dump 的 cli yaml 可查），但失败 kernel **不是视频塔**生成的 |

---

## 5. 归属仍未 100% 钉死（下一步第一件事）

已知候选调用点：

| 候选 | 位置 | 现状 |
|---|---|---|
| 动作专家的 flex 注意力 | `lingbotvla/models/vla/pi0/modeling_pi0.py:1767 get_attention_interface()`，读 `attention_implementation`；配置值 `flex_cached`（`configs/rocm/robotwin_official_paths_rocm.yaml:35`） | **最可能** |
| 视频塔 DINO teacher 的 flex 注意力 | `lingbotvla/models/vla/vision_models/dino_video/lumos_dinov3/layers/block.py:247`；模型自带 `dino_video/config.yaml:52` 也写着 `flex_block_causal` | 已改训练侧配置但**未确认是否作用到它** |

⇒ 需一次只读核对（看该 kernel 到底由哪条路径生成），再决定走 §6 的哪条方案。

---

## 6. 解决思路（三选一，按代价排序）

### 方案 A：让这条注意力不走 flex（推荐先试）
- **做法**：`attention_implementation: flex_cached → eager`（`configs/rocm/robotwin_official_paths_rocm.yaml:35`）；pi0 已有 `our_eager_attention_forward` 实现。
- **优点**：不碰环境、不碰 Triton，改动一行。
- **代价**：注意力变慢、显存更敏感 ⇒ 可能要同时 `--micro 6 --gas 2`（GBS 仍 84）。
- **风险**：数值与 flex 路径不逐位相同（需在记忆/日志里写明口径变更）。

### 方案 B：改对"真正的调用方"
- 若 §5 核对发现该 kernel 来自**视频塔**，则只需把**它自己那份** `dino_video/config.yaml` 的 `attention_mode` 一起改成 `sdpa_block_causal`（训练侧那份已改）。
- **优点**：代价最小，且保住了动作专家的 flex（性能最好）。
- **前提**：必须先做 §5 的归属核对。

### 方案 C：动 Triton 环境（最重，但能保住编译）
- 关掉该 pass（`tritonamdgpu-optimize-dot-operands`）或换/升 Triton 版本（上游已收到同类 reproducer 的话可能已修）。
- **优点**：编译收益全保住。
- **代价**：改环境 = 影响所有训练；容器重建后需重做（`/opt/aiter` 那类教训已有先例）。
- **风险**：关 pass 可能改变数值或触发其它 pass 失败。

---

## 7. Demo 代码

### 7.1 最小复现（把失败 kernel 编一遍，打印 Triton 原话）

工具：`tools/rocm/repro_triton_ttgir_failure.py`（本次新增；纯标准库 + triton，已在真机验证）

```bash
CACHE=/models/robotwin-persistent/al_cache/triton

# 自动找出「缺 ttgir」的失败 kernel 并复现（默认扫 num_warps{1,2,4,8} × num_stages{1,2,3}）
/opt/robotwin-env/bin/python tools/rocm/repro_triton_ttgir_failure.py --scan-cache $CACHE

# 只测一组配置（更快）
/opt/robotwin-env/bin/python tools/rocm/repro_triton_ttgir_failure.py --scan-cache $CACHE --warps 4 --stages 2
```

**真机实测输出（2026-10-10，已验证）**：

```
发现 1 个编译失败的 kernel：
  - /models/robotwin-persistent/al_cache/triton/2VSC5TDJXFCD5VH7CUZZZ6Y5XUWW4ZUY3URMJTGMICYI5XCVWETQ/triton_tem_fused_slice_backward_transpose_view_zeros_2.ttir
target      : GPUTarget(backend='hip', arch='gfx1100', warp_size=32)
  num_warps=4 num_stages=2 ⇒ 失败
      RuntimeError: PassManager::run failed
error: Failures have been detected while processing an MLIR pass pipeline
note: Pipeline failed while executing [`TritonAMDGPUOptimizeDotOperands` on 'builtin.module' operation]:
      reproducer generated at `std::errs, please share the reproducer above with Triton project.`
=== 汇总结论 ===
  能编过的配置: 无
  编不过的配置: ['nw=4,ns=2']
  ⇒ 全部失败：该 kernel 在当前 Triton 上无法 lowering ⇒ 需换注意力实现或改 Triton（§6-A/C）
```

退出码：`0` = 有可用配置；`1` = 全部失败；`2` = 用法/环境错误（便于脚本判定）。

> ⚠️ 踩坑记录：Triton 缓存布局是 `<cache>/<POS_HASH>/<name>.ttir`（**一层**）。
> 该工具最初写成 `glob("*/*/*.ttir")`（两层）⇒ 什么都找不到，误报"没有失败项"；
> 正确的是 `glob("*/*.ttir")`。另：`triton.compile()` 的第一个参数是**文件路径**，不是源码字符串。

### 7.2 判定「这台机器能不能用 flex 反向」的冒烟（不跑训练，几分钟）

工具：`tools/rocm/flex_attention_backward_smoke.py`（本次新增）

```bash
/opt/robotwin-env/bin/python tools/rocm/flex_attention_backward_smoke.py
# 默认与线上失败 kernel 同形：fp32 / seq 512 / 4 heads / head_dim 64 / 因果 block mask
```

判据与退出码：

| 退出码 | 含义 | 行动 |
|---|---|---|
| `0` | 前向 + 反向都通过 | 本机 flex 反向可用，可开 `torch.compile` |
| `3` | **反向触发 Triton 崩溃**（`PassManager::run failed` / `OptimizeDotOperands`） | 走 §6-A（换注意力实现）或 §6-C（改 Triton） |
| `4` | 无 GPU / 导入失败等运行时问题 | 检查环境 |

**真机实测输出（2026-10-10，7×W7900D 单卡验证，已复现）**：

```
device=AMD Radeon PRO W7900D dtype=fp32 seq=512 heads=4 head_dim=64
用 torch.compile(flex_attention, fullgraph=False) ⇒ 走融合 kernel（同线上）
[ok] 前向通过
[FAIL] 反向失败（Triton AMDGPU pass 崩溃）
   triton/backends/amd/compiler.py:262 make_ttgir
   RuntimeError: PassManager::run failed
```

🔴 **同一个脚本加 `--eager` 会"通过"** —— 因为 eager 下 `flex_attention` 退回"展开实现"，
而线上是 `torch.compile(model)` ⇒ 走融合 kernel。**判据必须用编译路径**，否则得到假绿。

> 为什么值得单独做：本问题**只在反向出现**（前向、评测、难度扫描全部正常），
> 所以"模型能加载、能评测"完全不能说明"能训练"。这个冒烟把结论提前到起训练之前。

### 7.3 判定失败 kernel 归属（只读，最省事）

```bash
LOG=/models/robotwin-persistent/al_runs/logs/train_al_v22.log
# 统计走了哪条注意力实现（注意：pi0 模块的同名打印会误导，需按模块路径区分）
grep -a "sdpa_attention_packed\|flex_attention_packed\|Using Flex Attn\|Using Eager Attn" $LOG | sort | uniq -c
# 看 torch flex 的调用栈线索
grep -a -B 3 "flex_attention.py" $LOG | head -20
```

---

## 8. 验收清单（改完必须逐项过）

1. **编译不再崩**：训练日志里 `PassManager::run failed` 计数 = 0；
2. **越过 Step 1**：出现 `Step: 1/5000` 且打印 loss；
3. **显存留余量**：`rocm-smi --showmeminfo vram` 峰值 < 44 GiB（48G 卡）；
4. **数值口径留痕**：若走了非 flex 路径，在启动日志/记忆里写明"硬度噪声与注意力的口径变更"；
5. **CPU 契约测试**：改配置解析/注入点这类改动，必须补一条 CPU 用例（本轮已有 `tests/test_al_launch_env_passthrough.py` 等先例）；
6. **产物写 overlay**：输出/缓存/TMPDIR 全部指向 `/models/robotwin-persistent/...`（`/workspace` 只剩 ~20G）。

---

## 9. 附：本轮相关实测数字（供排期参考）

| 项 | 数值 |
|---|---|
| 单步真训练（开编译，历史基准） | **9.6 s/步** |
| 单步峰值显存（开编译，micro=12） | **36.9 GiB** |
| 单步峰值显存（关编译/eager） | **46.7 GiB ⇒ OOM** |
| 难度扫描（7 卡分片、probe 0.05、259 样本/任务） | 约 **5.3 分钟**（其中 data 约 3 分钟、score 约 2 分钟，含编译抢 CPU） |
| 难度扫描（未分片、probe 0.10、511 样本） | 约 **8.3 分钟** |
| 每帧读取成本 | **约 5 s/帧**（pyav；占扫描 92%） |
| 首步编译耗时 | 约 5–10 分钟（缓存全新建时） |
