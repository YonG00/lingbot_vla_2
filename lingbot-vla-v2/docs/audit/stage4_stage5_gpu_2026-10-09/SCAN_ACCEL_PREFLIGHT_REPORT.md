# 扫描加速集成补丁 —— 无卡预检报告（2026-10-09）

补丁 `AL_SCAN_ACCEL_GMEAN200_INTEGRATED`（zip sha256 `44075f34…`）｜基线 = `2650590` + 上一轮 GMean200 选课补丁

## 一、合入核验：全部通过

| 项 | 结果 |
|---|---|
| `SHA256_MANIFEST.txt` | **8/8 逐位一致** ✓ |
| `git apply --check`（真实 HEAD） | **rc=0**；**10/10 文件干净应用** ✓，无整文件覆盖 |
| **后像核验** | 应用后 10 个文件 SHA256 与 `POSTIMAGE_SHA256.txt` **10/10 一致** ✓ |
| 改动范围 | 5 个生产文件（`hardness.py`、`scheduler.py`、`real/backend.py`、`real/build.py`、`utils/open_loop_validation.py`）+ 4 个新增（`scan_accel.py`、`scout_cache.py`、`tools/scan_accel_preflight.py`、`tests/test_scan_accel_integrated.py`）+ 1 文档。**无任何 YAML / 阈值表被改动** ✓ |

## 二、要求 4：正式配置与关键策略未变

| 项 | 结论 |
|---|---|
| 正式 `formal_50task_4pass.yaml` | **改动数 0** ✓（仍 `pass_metric: nmse`） |
| GMean200 阈值表 | 未改动 ✓（预检只读校验，`table_sha256` 见下） |
| NEW/Replay 70:30 | 未改动 ✓（补丁未触碰 dynamic GBS / sampler） |
| DCP / HF 策略 | 未改动 ✓（补丁未触碰 `train_lingbotvla.py` / `direct_hf_checkpoint.py` / `dcp_retention.py`） |

## 三、要求 5：三项安全默认（代码级核实）

| 项 | 默认 | 强制门控 |
|---|---|---|
| **Eval Batch** | `AL_EVAL_BATCH_MODE` 默认 **`serial`** ✓（`open_loop_validation.py:1079`） | 切 `auto` 必须 **`AL_EVAL_BATCH_APPROVED=1`**，否则 **RuntimeError** ✓（:1082-1083）；`probe` 只记录**影子**结果、**PASS 仍走串行** ✓（:1070） |
| **Hardness Batch** | `AL_HARDNESS_BATCH_MODE` 默认 **`fixed`**（Batch8 行为不变）✓（`real/backend.py:101`） | 切 `auto` 必须 **`AL_HARDNESS_BATCH_APPROVED=1`**，否则拒绝 ✓（:102）；上限 `AL_HARDNESS_BATCH_MAX=16`、预留 `AL_HARDNESS_RESERVE_GIB=10` ✓ |
| **Scout 缓存** | **OFF** ✓（`real/build.py:317` 默认 `None`；仅当设置 `AL_SCOUT_CACHE_ROOT` 才创建 ✓） | 不自动复用历史 NMSE 记录 ✓ |

**未设置任何 `*_APPROVED` 环境变量、未启用自动批量推理** ✓

## 四、机器无卡预检（要求 3）

### 4.1 PLAN ONLY（默认，不读权重）⇒ 退出码 0

```json
{"status":"PLAN_ONLY","checkpoint":null,"eval_mode":"serial","hardness_mode":"fixed",
 "scout_cache_mode":"off",
 "note":"Serial is unchanged by default. Probe never affects PASS. Auto must not be enabled before real GPU parity approval."}
```

### 4.2 CPU `--verify`（读真实权重分片）⇒ **READY**

```bash
$PY tools/scan_accel_preflight.py --verify \
  --checkpoint /data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt \
  --manifest /data/train/task_splits_50/manifest.json \
  --norm assets/norm_stats/robotwin_competition_clean.json \
  --thresholds /data/eval_results/open_loop/ref50k/pass_thresholds_gmean200_warn.json \
  --baseline /data/train/task_splits_50/task_baseline.json
```

| 字段 | 值 |
|---|---|
| status | **READY** ✓ |
| checkpoint | `/data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt` ✓（**6 个分片**） |
| **provenance_sha256** | **`a09e194d792652b4b682aa510b7c5be557a83486de26285e15fbc90093ea13fe`** |
| proof 定义 | *"SHA256 of raw checkpoint shard bytes + code + norm + split + thresholds + baseline"* ⇒ 把 **checkpoint/代码/norm/split/50任务阈值/baseline** 全部绑进一个来源指纹 ✓ |
| cache_namespace | `null` ✓（Scout 缓存关闭 ⇒ 无命名空间） |
| 模式 | `eval_mode=serial` / `hardness_mode=fixed` / `scout_cache_mode=off` ✓ |

## 五、测试

| 环境 | 项 | 结果 |
|---|---|---|
| 本机 | 新增专项（scan_accel + ratio-priority + preflight） | **37 passed**（与提供方一致 ✓） |
| 本机 | 相关回归（scan_accel/pipeline/resume/动态GBS/DCP/HF/精度/验收工具/hardness 计时） | **162 passed / 8 skipped** |
| 本机 | **全量 CPU 回归** | **782 passed / 18 skipped / 0 failed** |
| 机器 | 同批新增专项 | **37 passed** |
| 本机 / 机器 | CPU Gate | **✅(9/0/9)** 两边均通过 |

## 六、GPU 下一步（**仅计划，未执行；需你批准**）

1. **Eval Batch 数值对拍（probe）**：`AL_EVAL_BATCH_MODE=probe`（**只记录影子结果，PASS 仍串行**）⇒ 采集 Batch1/2/4 的 action 数值差、GMean/NMSE、峰值显存与吞吐；
2. **Hardness 独立验收**：逐样本 Loss 一致性（per-sample RNG/parity）+ 显存安全；必要时再谈 `AL_HARDNESS_BATCH_MODE=auto`（**必须先获批 `AL_HARDNESS_BATCH_APPROVED=1`**）；
3. 未完成上述真实模型验收前：**不启用自动批量、不声称已获得加速** ✓

## 七、遗留（按要求未动）

- 未启动 GPU、未跑正式 50-task 训练、未自动关机、**未清理任何历史权重/测试产物** ✓
- 历史测试产物仍在：`/data/outputs/gpu96_acceptance/hf_bf16_micro10_gas1/hf_milestones`（**12 GiB**）等（SHA256 已存于前一报告，随时可清 ✓）；`/data` 当前 **192 GB 可用** ✓
