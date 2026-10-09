# 训练/评测的模型加载加速（并行分片预读 + BF16 副本）

> 起因：2026-10-09 50-task GMean200 正式跑，`Prepare model → Start training` 共 **129 s**，
> 其中 `Loading checkpoint shards 0/6→6/6` 占 **84 s**（16 s/片 ≈ 290 MB/s）。
>
> **已实测（2026-10-09 晚，96G 机器）**：
> * 读盘：**冷读 84.2 s（0.28 GiB/s）** / **热读 4.2 s** / 并行热读 0.7 s；
> * BF16 副本（24 → 12 GB）生成成功，逐张量全量校验通过；
> * **真实启动：`Prepare model → Start training` 129 s → 20 s**（`[prewarm] 6 片 / 11.9 GiB / 0.6 s`；
>   `Loading checkpoint shards` 84 s → **≈3.4 s**）。⚠️ 该 20 s 是**页缓存已热**下的数字（副本刚写完并校验过），
>   **冷启动待重测**（推算 ≈46 s）。

## 1. 现状拆解

| 阶段 | 耗时 | 性质 |
|---|---|---|
| 读 6 个分片 | **84 s** | **读盘 24 GB**（该 ckpt 是 **F32**：1708 张量全 F32） |
| 分片读完 → 开训 | **~45 s** | 模型构建、Depth 模型、AdaNorm 零初始化、`.cuda()`、optimizer 初始化 |

两个关键事实：
1. 训练器**已经是流式直写**（`model.to_empty(device=…)` + 逐张量 `copy_`）⇒ 改"流式"没有收益；
2. 同一批文件**页缓存热**时，同类读取只要 **~2 s** ⇒ 84 s 主要是**冷读盘**（且 transformers 的
   `StateDictIterator` 是**顺序逐分片**读的，没有并发）。

## 2. 手段一：并行分片预读（已实现，默认开）

`lingbotvla/models/module_utils.py`：
* 新增 `_resolve_weight_files()`（把"解析权重文件列表"从 `_load_state_dict` 抽出，
  **查找顺序与原实现逐条一致**）；
* 新增 `_parallel_prewarm_shards()`：在 `_load_state_dict()` **之前**用线程池并发预读全部分片，
  把"冷读盘"变成"页缓存热读"；随后顺序逐张量读命中页缓存。

| 开关 | 默认 | 说明 |
|---|---|---|
| `AL_SHARD_PREWARM` | `1`（开） | `0` 关闭 |
| `AL_SHARD_PREWARM_THREADS` | `min(6, CPU)` | 线程数（不超过分片数） |

**失败绝不挡加载**（只记一条 `[prewarm] ⚠️ 并行预读跳过（…）`），因为它是纯加速手段。
日志形如：`[prewarm] 并行预读 6 片 / 24.0 GiB / 21.3 s（6 线程 ⇒ 随后的顺序读命中页缓存）`。

## 3. 手段二：BF16 权重副本（`tools/make_bf16_ckpt.py`）

思路：**只读一半体积**。该 ckpt 是 F32（24 GB），加载后一律转 BF16（12 GB）用
（`torch_dtype=bfloat16` 时 `copy_` 会做转换）⇒ 准备一份 BF16 副本，读盘量近似减半，并省掉 CPU 转换。

```bash
PY=/data/miniconda3/envs/lingbotvla/bin/python
SRC=/data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt
DST=/data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt_bf16

# ① 先看计划（不写盘）
$PY tools/make_bf16_ckpt.py --src $SRC --dst $DST --dry-run
# ② 转换（默认**逐张量全量校验**：浮点断言 dst == src.to(bf16) 逐位相等；非浮点原样）
$PY tools/make_bf16_ckpt.py --src $SRC --dst $DST --verify full
# ③ 用它启动（只改 model_path）
... MODEL_PATH=$DST ...
```

行为与安全：
* 浮点 → `bfloat16`；**非浮点（int64/bool）原样保留**；
* 分片文件名与 `model.safetensors.index.json` 的 `weight_map` **不变**（仅 `metadata.total_size` 更新）；
* 其余文件（config / processor / tokenizer / *.py）按字节复制；
* **校验失败 ⇒ 非零退出**（`--verify full|sample:N|none`）；
* 拒绝：目标目录非空、目标=源或位于源内部、源里有 `.bin`；
* 磁盘成本 **+12 GB**；**首次仍需读满一遍**（收益在读盘量，不是"零成本"）。

## 4. 手段三：页缓存预热

并行预读已覆盖"本次进程内"的预热。若还想让**下一次启动**也快（重启/续训场景），
可在启动脚本前置一句（幂等、无副作用）：

```bash
cat $SRC/*.safetensors > /dev/null &     # 后台预热；与前面的导入/构建重叠
```

⚠️ 预热**本身**也要读一遍盘（不减少总 I/O），它的价值在"**重复启动**"与"**与其它启动阶段重叠**"。

## 5. 怎么验证（实测结果见下）

```bash
CK=/data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt
time cat $CK/*.safetensors > /dev/null                                   # ① 冷读（顺序）
time cat $CK/*.safetensors > /dev/null                                   # ② 热读（页缓存）
time (for f in $CK/*.safetensors; do cat $f > /dev/null & done; wait)     # ③ 并行
```

**2026-10-09 实测（6 片 / 23.75 GiB）**：

| 方式 | 耗时 | 有效带宽 |
|---|---|---|
| 冷读（顺序） | **84.2 s** | 0.28 GiB/s（≈290 MB/s）—— 与训练日志的 84 s 完全吻合 |
| 热读（顺序） | **4.2 s** | ~5.7 GiB/s |
| 并行读 | **0.7 s** | ⚠️ 见下 |

⚠️ **第三条不是"并行冷读"**：三种读法连着测，前两次已把 23.75 GiB 读进页缓存 ⇒ 第三条实际测的是
**并行热读**（这正是"顺序 4.2 s → 并行 0.7 s"的 6× 差异来源）。**真正的并行冷读尚未测**。
要拿真数字：① 给脚本加 `--drop-cache`（`sync; echo 3 > /proc/sys/vm/drop_caches`，容器里可能被拒）
或逐文件 `posix_fadvise(DONTNEED)`；② **更简单**：重启机器后跑一次真实启动，看
`[prewarm] 并行预读 6 片 / 11.9 GiB / X s` —— 那就是冷读的真实值。

**端到端等价性（两条独立证据）**：
1. 冒烟 A3：用**仓库真实加载器**（`_resolve_weight_files` + `StateDictIterator`）把 BF16 副本读回，
   10 张量**逐张量一致**、浮点全 bf16；
2. **生产实测**：同一 step500 权重，一次用 F32 原版加载、一次用 BF16 副本加载，bootstrap 阶段
   逐任务评测差异 **0.01–0.63%**（模型自身噪声量级）⇒ **BF16 副本没有引入系统偏差**。

## 6. 未做 / 待实测

| 项 | 说明 |
|---|---|
| 并行**逐张量加载**（不只是预读） | 预读已把数据放进页缓存，顺序读命中缓存后不再受盘限制 ⇒ 收益有限，先不做 |
| BF16 副本的**端到端数值校验** | 工具做了逐张量逐位校验；"跑一次评测比对 MSE"仍需 GPU（可选） |
| 那 ~45 s 的构建段 | 需 profile（Depth 模型 / AdaNorm / optimizer 各占多少），暂未动 |
| **并行冷读**的真实数字 | 未测（前两次读已污染页缓存）⇒ 用 `--drop-cache` 或重启后测 |
| 并行**装权进模型**（多线程 `copy_`） | 未做，**判断不值得**：页缓存命中后这段仅 3.4 s（占 20 s 的 17%），改动风险高于收益 |

## 7. 相关文件

| 路径 | 作用 |
|---|---|
| `lingbotvla/models/module_utils.py` | `_resolve_weight_files()` / `_parallel_prewarm_shards()` + 接线 |
| `tools/make_bf16_ckpt.py` | BF16 副本工具（含逐张量校验、fail-closed） |
| `tests/test_shard_prewarm.py` | 预读 6 项 CPU 测试（开关/线程/失败不阻塞/接线） |
| `tests/test_make_bf16_ckpt.py` | 副本 6 项 CPU 测试（转换正确/篡改必被抓/拒绝覆盖） |

## 8. 一键体检（推荐先跑这个）

```bash
PY=/data/miniconda3/envs/lingbotvla/bin/python
$PY tools/load_accel_smoke.py                                   # A) 合成端到端（秒级、无副作用）
$PY tools/load_accel_smoke.py --ckpt <hf_ckpt 目录>              # B) 真机实测（只读原目录）
$PY tools/load_accel_smoke.py --ckpt <hf_ckpt> --write-copy <新目录>   # B+) 顺带生成 BF16 副本
```

**A 模式**（全部用仓库真实实现，不是替身）：
1. 造 2 分片 F32 小 ckpt（含 int64 / bool / 0 维 / 空张量）；
2. `make_bf16_ckpt` 转换 + **逐张量全量校验**；
3. 用仓库自己的 `_resolve_weight_files()` + `StateDictIterator` **把副本读回来**核对 key/形状/dtype；
4. 调 `_parallel_prewarm_shards()` 报告字节数/线程/耗时。

**B 模式**：冷读 / 热读 / 并行冷读三条计时 + 判定"并行是否有收益"；`--write-copy` 才写盘。

**退出码**：`0` 全部通过；`2` 失败（转换/校验/读回不一致）；`3` **部分完成** ——
本机 transformers 版本与训练机不同、无法导入仓库加载器时，A 只验到"转换+校验"，
**端到端读回必须在训练机上跑**（Mac 上没有 `transformers.utils.import_utils.is_safetensors_available`）。

## 9. 已知边界与防御（单元测试覆盖）

| 边界 | 行为 |
|---|---|
| 非连续张量 | safetensors `save_file` 会 `ValueError: non contiguous tensor` ⇒ 工具自动 `.contiguous()` |
| 两个 key 共享存储（tied weights） | safetensors 会 `RuntimeError: Some tensors share memory` ⇒ 工具自动 `clone()` 去重 |
| 跨分片重复 key | **拒绝转换**（HF 索引会歧义） |
| 源含 `.bin` / 目标非空 / 目标=源或在其内部 | 拒绝（fail-closed） |
| 分片缺失 / 打不开 | 校验返回 `ok=False` 且 CLI 非零退出（**不抛 traceback**） |
| 0 维 / 空张量 | 正常处理（已实测可存可取） |
| 无 `index.json` 的单文件 ckpt | 正常转换，不凭空造索引 |
| `AL_SHARD_PREWARM_THREADS=abc` 等非法值 | 预读整体跳过并记日志，**不影响加载** |
| 预读解析失败（路径不存在等） | 同上，只记 `[prewarm] ⚠️ …` |
