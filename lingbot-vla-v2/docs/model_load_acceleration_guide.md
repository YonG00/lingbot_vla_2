# 训练/评测的模型加载加速（并行分片预读 + BF16 副本）

> 起因：2026-10-09 50-task GMean200 正式跑，`Prepare model → Start training` 共 **129 s**，
> 其中 `Loading checkpoint shards 0/6→6/6` 占 **84 s**（16 s/片 ≈ 290 MB/s）。

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

## 5. 怎么验证（机器上，约 5 分钟）

```bash
CK=/data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt
time cat $CK/*.safetensors > /dev/null                                   # ① 冷读（顺序）
time cat $CK/*.safetensors > /dev/null                                   # ② 热读（页缓存）
time (for f in $CK/*.safetensors; do cat $f > /dev/null & done; wait)     # ③ 并行冷读
```
判定：**③ 明显快于 ①** ⇒ 并行预读有效；**② << ①** ⇒ 页缓存价值大（重启场景）。
再跑一次真实启动，看日志里 `[prewarm] …` 与 `Prepare model → Start training` 的总时长，
与基线 **129 s** 对比。

## 6. 未做 / 待实测

| 项 | 说明 |
|---|---|
| 并行**逐张量加载**（不只是预读） | 预读已把数据放进页缓存，顺序读命中缓存后不再受盘限制 ⇒ 收益有限，先不做 |
| BF16 副本的**端到端数值校验** | 工具做了逐张量逐位校验；"跑一次评测比对 MSE"仍需 GPU（可选） |
| 那 ~45 s 的构建段 | 需 profile（Depth 模型 / AdaNorm / optimizer 各占多少），暂未动 |
| 真实加速倍数 | **待开机实测**（本次只交付了实现与验证方法） |

## 7. 相关文件

| 路径 | 作用 |
|---|---|
| `lingbotvla/models/module_utils.py` | `_resolve_weight_files()` / `_parallel_prewarm_shards()` + 接线 |
| `tools/make_bf16_ckpt.py` | BF16 副本工具（含逐张量校验、fail-closed） |
| `tests/test_shard_prewarm.py` | 预读 6 项 CPU 测试（开关/线程/失败不阻塞/接线） |
| `tests/test_make_bf16_ckpt.py` | 副本 6 项 CPU 测试（转换正确/篡改必被抓/拒绝覆盖） |
