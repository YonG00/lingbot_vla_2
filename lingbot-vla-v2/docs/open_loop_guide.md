# 开环（离线）诊断 使用文档

> 适用版本：`YonG00/lingbot_vla_2`（LingBot-VLA-v2 post-training）
> 相关文件：`scripts/open_loop_eval.py`（评测本体，仓库自带）、`tools/collect_open_loop.py`（结果汇总，新增）、`tools/task_split.py`（取轨迹索引）
> 产物：`/data/eval_results/open_loop/<tag>/<task>/`（`eval.log` + 每轨 `{traj_id}.png`）、`/data/eval_results/open_loop/registry.json`

---

## 1. 这个功能解决什么问题

闭环评测（起推理 server + RoboTwin 仿真 + curobo）**贵且慢**，跑一次十几分钟到几十分钟，
而且失败时只知道"0/12"，不知道**为什么**失败 —— 是策略完全乱动，还是接近但差一点。

开环（离线）诊断便宜得多：**无仿真、单卡、几分钟**，而且给出**可量化的误差**和**逐维曲线图**，
能回答「预测是不是退化成噪声/平线」。

> ⚠️ **定位**：项目 spec 明确「当前不把 open-loop 指标作为模型选择依据，核心指标是 closed-loop success rate」。
> 所以开环是**诊断**，不是成绩，不能替代闭环。

---

## 2. 一句话原理

> 取若干条轨迹，在每个采样点让策略预测动作块，和真值比 `MSE / MAE`；
> 关键是**和「只输出常数均值」的下界比** —— 低于它才算学到东西。

`open_loop_eval.py` 的调用链（详见 `knowledge/codebase.md` 的流程图）：
解析参数 → 加载 `LingbotVLAv2Server`（读 ckpt 上两级的 `lingbotvla_cli.yaml` → 配置/权重）
→ 建 `LeRobotDataset` → 逐轨 `prepare_eval_observation` → `policy.infer` → 逐轨算 MSE/MAE 并出图。

---

## 3. 快速开始（本项目实测可用的两条命令）

**两个必须**（踩错就崩）：
- `--model_path` 必须是**能取到 `lingbotvla_cli.yaml` 的 ckpt 目录**（脚本读它上两级）
- 若该 yaml 的 `tokenizer_path` 是占位符，**必须 `export QWEN3VL_PATH`**

### 3.1 base 基线

```bash
cd /data/code/lingbot-vla-v2
source /data/miniconda3/etc/profile.d/conda.sh && conda activate lingbotvla
export QWEN3VL_PATH=/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct
export CUDA_VISIBLE_DEVICES=0

DATA=/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30
BASE=/data/models/lingbot-vla-v2-6b-base-eval/checkpoints/global_step_0/hf_ckpt
OUT=/data/eval_results/open_loop/base/click_bell; mkdir -p "$OUT"

python scripts/open_loop_eval.py \
  --model_path "$BASE" --robo_name robotwin --data_path "$DATA" \
  --traj_ids $(python tools/task_split.py --task click_bell --pick-train 5) \
             $(python tools/task_split.py --task click_bell --pick-val 10) \
  --use_length 50 --chunk_ret true --save_plot_path "$OUT" 2>&1 | tee "$OUT/eval.log"
```

> ⚠️ **base 只能传 `-base-eval/.../global_step_0/hf_ckpt`**，**不能**传 `-6b-base/lingbot-vla-v2-6b`
> —— 后者的上两级是 `/data/models/`，那里没有 `lingbotvla_cli.yaml` ⇒ 直接 FileNotFoundError。
> `-base-eval` 的 `hf_ckpt` 全是软链指向 base 权重，是同一份模型。

### 3.2 某个 ckpt（换 `CKPT` 即可）

```bash
CKPT=/data/outputs/phase1_L1_vit_frozen/checkpoints/global_step_1558/hf_ckpt
OUT=/data/eval_results/open_loop/vit_frozen_1558/click_bell; mkdir -p "$OUT"
# 其余同上
```

---

## 4. 汇总结果

```bash
python tools/collect_open_loop.py            # 扫描 open_loop/*/*/eval.log → 写 registry.json
python tools/collect_open_loop.py --print    # 只打印表格，不写文件
```

输出示例（`click_bell`，下界 0.2688）：

```
tag                task          floor   train MSE   val MSE   val SEM    val R²
vit_frozen_1558    click_bell    0.2688     0.5267    0.5040    0.0479    -0.875
base               click_bell    0.2688     0.4873    0.5548    0.0383    -1.064
```

`registry.json` 里每条 run 含：`per_traj`（逐条 MSE/MAE）、`groups.train/val`（含 SEM）、`floor_mse`、`r2_val`、`log` 路径。
**以后每跑完一个 ckpt 重跑一次即可自动并入。**

---

## 5. 判据（怎么读这些数）

`floor_mse` = 该任务「只输出常数均值」的 MSE = 该任务全部回合**各维 `std²` 的均值**。

| MSE 相对 floor | 含义 |
|---|---|
| **< floor** | 比"什么都不做"好 ⇒ **才算学到东西**（R² > 0） |
| **≈ floor** | 退化成条件均值（预测一条平线） |
| **≫ floor** | 输出的是噪声，没学到 |

**噪声底**：base 自身的 train/val 差就有 **14%**（base 无 train/val 概念 ⇒ 纯抽样噪声）
⇒ **以后 ckpt 的 train/val 差 <14% 都不算过拟合**。
另外 val 内单条方差很大（n=10 时均值标准误约 ±7%），小差异别过度解读。

**逐条配对**比看均值灵敏得多 —— `collect_open_loop.py` 存了 `per_traj`，可以自己配对做符号检验。

---

## 6. 口径必须一致（否则不可比）

同一批对比里，以下四项**一个都不能变**：

1. **`--traj_ids` 完全相同**（建议硬编码，或固定用 `--pick-*` 的同一策略）
2. **`--use_length 50 --chunk_ret true`**
3. **精度**：**不要加 `--use_bf16`**（官方 release 验证用 FP32，训练也是 F32）
4. **归一化同源**：`--norm_path` 留空即可（自动取 ckpt yaml 的 `norm_stats_file`；训练/评测同源才可比）

`--chunk_ret` 的默认值取决于 ckpt yaml 的 `data.video_enabled`（未设置 ⇒ `True`=chunk 模式），**显式写死更稳**。

---

## 7. 注意事项

1. **不需要 4 次运行**：脚本本就**逐条打印** `MSE for trajectory {id}`，所以把 train 5 条 + val 10 条**合成一次**跑，事后按 id 拆分即可。
2. **成本**：1 次模型加载（1–2 分）+ 每条约 2 次前向（`sample_actions` ~0.6 s）⇒ 15 条 **约 3 分钟**。
3. **`--num_gpus` 不存在** —— 这是单进程单卡脚本（`device='cuda'`）。开 1 张卡就够，结果与卡数无关。
4. **`--ckpt-root` 是另一个脚本（调度器）的参数**，不是本脚本的；本脚本用 `--model_path` 指单个 `hf_ckpt`。
5. **看 `{traj_id}.png` 比看数字有用** —— 图上 GT 与 pred 的逐维曲线：**pred 是高频抖动噪声 / 一条平线 / 贴合 GT**，一眼能分。
6. 本脚本**不产闭环评测的 task list**；要跑闭环请另建 `<task>.eval.txt` 给 launcher 的 `--task_list_file`。
