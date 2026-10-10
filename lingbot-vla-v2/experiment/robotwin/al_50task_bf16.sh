#!/usr/bin/env bash
# =============================================================================
# Auto Learning 训练：支持当前 PASS 总数目标与无存档 Smoke 模式（48G 单卡 + BF16）
# -----------------------------------------------------------------------------
# 与 single_task_train.sh 的区别（那个是"单任务探针"）：
#   ① 数据 = **50 个任务**的训练划分（`task_splits_50/combined.train_ids.json`）
#   ② 开启 Auto Learning（`--train.auto_learning` + manifest + baseline 三件套）
#   ③ 精度 = **BF16**（`MIXED=false`）—— 48G 单卡 F32 放不下（权重25.5+梯度23.8+优化器23.7≈73G）
#   ④ 自带 TensorBoard（默认 6006），并在结束时打印查看方式
#
# 用法（任意目录均可，脚本自定位仓库根）：
#   bash experiment/robotwin/al_50task_bf16.sh                  # 真跑
#   DRY_RUN=1 bash experiment/robotwin/al_50task_bf16.sh        # 只打印计划 + 完整命令
#   TRAIN_OUT=/data/outputs/al_50task_v2 bash experiment/...sh  # 换输出目录
#   TB=0 bash experiment/robotwin/al_50task_bf16.sh             # 不起 TensorBoard
#
# 前置（本脚本会自检）：
#   python tools/task_split.py --task all --out /data/train/task_splits_50
#   python -m lingbotvla.auto_learning.tools.compute_task_baseline \
#       --manifest /data/train/task_splits_50/manifest.json \
#       --config   /data/outputs/single/click_bell/lingbotvla_cli.yaml \
#       --out      /data/train/task_splits_50/task_baseline.json
#
# 收工条件（配置里写死，见 configs/auto_learning/formal_50task_4pass.yaml）：
#   target_total_passed_tasks = 4     ← 包含 Bootstrap PASS；当前累计满 4 即停
#   max_global_steps             = 20000 ← 钱的安全带
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- 参数 -------------------------------------------------------------------
SPLIT_DIR=${SPLIT_DIR:-/data/train/task_splits_50}
PHASES=${PHASES:-/data/train/phases}
TRAIN_OUT=${TRAIN_OUT:-/data/outputs/al_50task_bf16}
AL_CFG=${AL_CFG:-$REPO/configs/auto_learning/formal_50task_4pass.yaml}

MICRO=${MICRO:-1}
GAS=${GAS:-10}
N_GPU=${N_GPU:-1}
GBS=$(( MICRO * GAS * N_GPU ))

MAX_STEPS=${MAX_STEPS:-20000}
SAVE_EVERY=${SAVE_EVERY:-1000}
SAVE_EPOCHS=0                      # 只按步存档，不开轮末存档
SMOKE_NO_CHECKPOINT=${SMOKE_NO_CHECKPOINT:-0}
if [ "$SMOKE_NO_CHECKPOINT" = "1" ]; then
    if [ "${RESUME:-0}" = "1" ]; then
        echo "❌ SMOKE_NO_CHECKPOINT=1 不支持 RESUME" >&2; exit 2
    fi
    SAVE_EVERY=0
    SAVE_EPOCHS=0
    PRUNE=0
fi

# 精度：**BF16**（48G 单卡必须）。true=F32 权重（≈73G，48G 放不下）
MIXED=${MIXED:-false}
AUGMENT=false                      # Auto Learning 要求 image_augment=false（开了会 fail-fast）

MODEL_PATH=${MODEL_PATH:-/data/models/lingbot-vla-v2-6b-base/lingbot-vla-v2-6b}
#: 训练配置（各机器路径不同；ROCm/cpu1 用 configs/rocm/robotwin_official_paths_rocm.yaml 覆盖）
CONFIG=${CONFIG:-/data/train/configs/robotwin_official_paths.yaml}
STEP_OFFSET=${STEP_OFFSET:-0}      # >0 时编号接着旧模型（见 train_lingbotvla.py 的注释）
RESUME=${RESUME:-0}
RESUME_BOOL=$([ "$RESUME" = "1" ] && echo true || echo false)

# 剪枝看门狗：每份 hf_ckpt 校验通过后剪掉 DCP。
# ⚠️ RESUME=1 时强制 keep-last≥1，否则看门狗会把要恢复的那份 DCP 剪掉。
PRUNE=${PRUNE:-0}   # DCP 要可 Resume；不运行按 HF 完成情况剪 DCP 的旧看门狗
PRUNE_KEEP=${PRUNE_KEEP:-1}
PRUNE_MIN_AGE=${PRUNE_MIN_AGE:-300}

# 新正式策略：每 1000 optimizer steps DCP；新增每 2 个非 Bootstrap PASS 直接 HF；收尾仅 DCP。
# 独立 HF 不读取 DCP；禁用原有每次 DCP 之后的自动 HF 转换。
DCP_MODE=${DCP_MODE:-always}
HF_PASS_INTERVAL=${HF_PASS_INTERVAL:-2}
DCP_FINAL_GB=${DCP_FINAL_GB:-0}
# 安全保留：只留最近 N 份**完整** DCP（新 DCP 校验成功后才删最旧；保存失败则一份都不删）。
# HF 里程碑在 TRAIN_OUT/hf_milestones/ 独立管理，不受本策略影响；Smoke 无存档模式不启用。
DCP_KEEP_LAST=${DCP_KEEP_LAST:-2}
# HF 直出存储精度（与训练/评测精度解耦）：BF16 训练默认导出 bf16（约 12G），F32 训练默认 fp32。
# 注意：若源权重是 fp32 而这里选 bf16，属于**有意降低存储精度、并非无损**（日志会明确标注）。
HF_EXPORT_DTYPE=${HF_EXPORT_DTYPE:-$([ "$MIXED" = "true" ] && echo fp32 || echo bf16)}
#: 是否在**每个存档点**同时导出 HF（默认关：DCP 更快省盘；HF 约 +12G、多花 1–2 分钟）。
#: 🔴 2026-10-10 教训：这里写死 false ⇒ r2 run 只有 DCP、**一个 safetensors 都没有**，
#:    闭环评测（要求 `<CKPT_ROOT>/checkpoints/global_step_N/hf_ckpt`）直接没法跑。
#:    需要 HF 时：`SAVE_HF=1`（导出精度见上面的 HF_EXPORT_DTYPE，默认 bf16）。
SAVE_HF_BOOL=$([ "${SAVE_HF:-0}" = "1" ] && echo true || echo false)
DISK_GUARD_BOOL=$([ "$SMOKE_NO_CHECKPOINT" = "1" ] && echo false || echo true)

# TensorBoard
TB=${TB:-1}
TB_PORT=${TB_PORT:-6006}

DRY_RUN=${DRY_RUN:-0}

if [ "$RESUME" = "1" ] && [ "$PRUNE" = "1" ] && [ "$PRUNE_KEEP" -lt 1 ]; then
    echo "[al50] ⚠️  RESUME=1 ⇒ 强制 PRUNE_KEEP=1" >&2
    PRUNE_KEEP=1
fi

PY=${PY:-/data/miniconda3/envs/lingbotvla/bin/python}

# ---- torchrun 端口 ----------------------------------------------------------
# 🔴 `train.sh` 里 `MASTER_PORT` 默认写死 **62500**。若同时有别的训练在跑，
#    torchrun 会直接崩：
#      DistNetworkError: ... port: 62500 ... EADDRINUSE, address already in use
#    （2026-10-07 实测：探针和一个正在跑的回归撞了端口。）
#    ⇒ 默认端口被占就**自动换一个空闲端口**。
MASTER_PORT=${MASTER_PORT:-62500}
if ! "$PY" -c "import socket,sys; s=socket.socket(); s.bind(('127.0.0.1', int(sys.argv[1]))); s.close()" "$MASTER_PORT" 2>/dev/null; then
    _FREE_PORT=$("$PY" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); p=s.getsockname()[1]; s.close(); print(p)")
    echo "[al50] ⚠️  MASTER_PORT=$MASTER_PORT 已被占用（可能有别的训练在跑）⇒ 自动改用 $_FREE_PORT"
    MASTER_PORT=$_FREE_PORT
fi
export MASTER_PORT

# ---- 前置自检 ---------------------------------------------------------------
FAIL=0
[ -x "$PY" ]                   || { echo "❌ 找不到 $PY" >&2; FAIL=1; }
[ -f "$AL_CFG" ]               || { echo "❌ 找不到 AL 配置 $AL_CFG" >&2; FAIL=1; }
[ -f "$SPLIT_DIR/manifest.json" ]       || { echo "❌ 缺 $SPLIT_DIR/manifest.json（先跑 tools/task_split.py --task all --out ${SPLIT_DIR}）" >&2; FAIL=1; }
[ -f "$SPLIT_DIR/combined.train_ids.json" ] || { echo "❌ 缺 $SPLIT_DIR/combined.train_ids.json" >&2; FAIL=1; }
[ -f "$SPLIT_DIR/task_baseline.json" ]  || { echo "❌ 缺 $SPLIT_DIR/task_baseline.json（P1-5 起 baseline 默认必填）" >&2; FAIL=1; }
[ -d "${QWEN3VL:-/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct}" ] || {
    echo "❌ QWEN3VL_PATH 不存在或未设（export QWEN3VL_PATH=/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct）" >&2; FAIL=1; }
[ "$FAIL" = "0" ] || { echo "❌ 前置未通过，退出"; exit 2; }

# ---- 从 manifest 估总步数，推导 epoch 数 ------------------------------------
read -r N_TASKS TOTAL_FRAMES < <("$PY" - "$SPLIT_DIR/manifest.json" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
ts = m["tasks"]
print(len(ts), sum(t["train_frames"] for t in ts.values()))
PYEOF
)
STEPS_PER_EPOCH=$(( TOTAL_FRAMES / GBS ))
[ "$STEPS_PER_EPOCH" -lt 1 ] && { echo "❌ 每轮步数为 0（train_frames=$TOTAL_FRAMES < gbs=${GBS}）" >&2; exit 1; }
EPOCHS=$(( (MAX_STEPS + STEPS_PER_EPOCH - 1) / STEPS_PER_EPOCH + 1 ))

# save_steps 的真实语义为全局 optimizer step 的模数；无需强行整除 MAX_STEPS。
# 例如 MAX_STEPS=19500 时必须仍按 1000、2000… 保存，而不是改成 975。
if [ "${SAVE_EVERY}" -le 0 ] 2>/dev/null; then
    SAVE_STEPS=0; N_SAVES=0
else
    SAVE_STEPS=$SAVE_EVERY
    N_SAVES=$(( MAX_STEPS / SAVE_STEPS ))
fi

MANIFEST="$SPLIT_DIR/manifest.json"
TRAIN_IDS="$SPLIT_DIR/combined.train_ids.json"
BASELINE="$SPLIT_DIR/task_baseline.json"

# ---- 从 AL 配置里读出真实收工条件（别在计划里写死）--------------------------
read -r AL_PASS_CAP AL_ATT_CAP AL_MAXSTEPS AL_PASS_NMSE AL_TOTAL_CAP < <("$PY" - "$AL_CFG" <<'PYEOF'
import sys, yaml
raw = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
b = raw.get("auto_learning", raw) or {}
def g(k):
    v = b.get(k)
    return "null" if v is None else str(v)
print(g("max_new_tasks_passed_this_run"), g("max_new_tasks_attempted_this_run"),
      g("max_global_steps"), g("pass_nmse"), g("target_total_passed_tasks"))
PYEOF
)

# ---- 计划 -------------------------------------------------------------------
cat <<EOF
================================================================================
  Auto Learning 运行计划 — 当前 PASS 目标 $AL_TOTAL_CAP 个
================================================================================
  仓库根      = $REPO
  数据划分    = $SPLIT_DIR
    任务数    = $N_TASKS 个（train 帧合计 ${TOTAL_FRAMES}）
  AL 配置     = $AL_CFG
  AL manifest = $MANIFEST
  AL baseline = $BASELINE
  训练输出    = $TRAIN_OUT
  冻结配置    = train_expert_only=false + freeze_vision_encoder=true
  精度        = MIXED=${MIXED}（true=F32 / false=**bf16**）
  image_augment = ${AUGMENT}（AL 要求 false）
  训练规模    = ${EPOCHS} epoch × ${STEPS_PER_EPOCH} 步/轮，max_steps=${MAX_STEPS}（绝对）
  批大小      = micro ${MICRO} × gas ${GAS} × ${N_GPU} 卡 = gbs ${GBS}
                （AL new_ratio 模式按 DP local batch 动态分配；无 new_ratio 时仍用静态 slots）
  存档计划    = $([ "$SMOKE_NO_CHECKPOINT" = "1" ] && echo 'NO_CHECKPOINT (无 DCP/HF，不支持 Resume)' || echo "每 ${SAVE_STEPS} 步一份 × ${N_SAVES} 份；DCP_MODE=${DCP_MODE}")
  剪枝看门狗  = PRUNE=$PRUNE keep-last=$PRUNE_KEEP min-age=${PRUNE_MIN_AGE}s
  DCP 保留    = 最近 ${DCP_KEEP_LAST} 份完整 DCP（新份校验成功后才删最旧；HF 里程碑独立管理）
  续训        = RESUME=${RESUME}（${RESUME_BOOL}）
  编号起点    = STEP_OFFSET=${STEP_OFFSET}（0=从零计）
  初始权重    = $MODEL_PATH
  torchrun 端口 = ${MASTER_PORT}（被占会自动换）
  TensorBoard = $([ "$TB" = "1" ] && echo "端口 ${TB_PORT}（logdir=$TRAIN_OUT/runs）" || echo "关闭")
  收工条件    = 当前 PASS 总数达到 ${AL_TOTAL_CAP}（含 Bootstrap，通过 Registry 当前状态计算）
                新增 PASS 上限 ${AL_PASS_CAP}（旧配置 max_new_tasks_passed_this_run）
                最多主动尝试 $AL_ATT_CAP 个任务（max_new_tasks_attempted_this_run）
                总步数上限 ${AL_MAXSTEPS}（max_global_steps，兜底）
  及格线      = pass_nmse=${AL_PASS_NMSE}（nmse ≤ 该值判 PASS）
================================================================================
EOF

if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] 只打印计划，不训练。完整命令："
    echo
    cat <<EOF
cd $REPO
export QWEN3VL_PATH=${QWEN3VL:-/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct}
# 🔴 不能写死单卡：多卡时按 N_GPU 推导可见卡（否则 train.sh 数出 1 卡 ⇒ torchrun 只起 1 进程；
#    AMD 上还踩过 nvidia-smi 缺失被 wc -l 数成 1 的坑）
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if [ "${N_GPU:-1}" -gt 1 ] 2>/dev/null; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPU - 1)))
    export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
  else
    export CUDA_VISIBLE_DEVICES=0
  fi
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash train.sh tasks/vla/train_lingbotvla.py "$CONFIG" \\
  --model.model_path       $MODEL_PATH \\
  --data.train_path        $PHASES/datasets.txt \\
  --data.episode_ids_file  $TRAIN_IDS \\
  --data.image_augment     false \\
  --train.output_dir       $TRAIN_OUT \\
  --train.micro_batch_size $MICRO \\
  --train.gradient_accumulation_steps $GAS \\
  --train.global_batch_size $GBS \\
  --train.num_train_epochs $EPOCHS \\
  --train.max_steps        $MAX_STEPS \\
  --train.save_steps       $SAVE_STEPS \\
  --train.save_epochs      $SAVE_EPOCHS \\
  --train.save_hf_weights  $SAVE_HF_BOOL \\
  --train.async_save_hf_weights false \\
  --train.hf_pass_interval $([ "$SMOKE_NO_CHECKPOINT" = "1" ] && echo 0 || echo "$HF_PASS_INTERVAL") \\
  --train.hf_export_dtype $HF_EXPORT_DTYPE \\
  --train.enable_resume    $RESUME_BOOL \\
  --train.train_expert_only false \\
  --train.freeze_vision_encoder true \\
  --train.enable_mixed_precision $MIXED \\
  --train.rmpad            false \\
  --train.rmpad_with_pos_ids false \\
  --train.step_offset      $STEP_OFFSET \\
  --train.disk_guard       $DISK_GUARD_BOOL \\
  --train.disk_guard_margin 1.1 \\
  --train.disk_check_interval 50 \\
  --train.smoke_no_checkpoint $([ "$SMOKE_NO_CHECKPOINT" = "1" ] && echo true || echo false) \\
  --train.dcp_save_mode    $DCP_MODE \\
  --train.dcp_final_size_gb $DCP_FINAL_GB \\
  --train.dcp_keep_last $([ "$SMOKE_NO_CHECKPOINT" = "1" ] && echo 0 || echo "$DCP_KEEP_LAST") \\
  --train.auto_learning          $AL_CFG \\
  --train.auto_learning_manifest $MANIFEST \\
  --train.auto_learning_baseline $BASELINE
EOF
    echo
    echo "[dry-run] TensorBoard: $PY -m tensorboard.main --logdir $TRAIN_OUT/runs --port $TB_PORT --host 0.0.0.0"
    exit 0
fi

# ---- 磁盘预检 ---------------------------------------------------------------
mkdir -p "$TRAIN_OUT"
AV_GB=$(df -BG --output=avail "$TRAIN_OUT" 2>/dev/null | tail -1 | tr -dc '0-9' || true)
echo "[al50] 可用磁盘 ${AV_GB:-?}G（bf16 单份完整 DCP ≈31G；final_only 时更省）"

# ---- 剪枝看门狗 -------------------------------------------------------------
PRUNE_PID=""
if [ "$PRUNE" = "1" ]; then
    setsid nohup "$PY" -u "$REPO/tools/prune_dcp.py" \
        --ckpt-root "$TRAIN_OUT" --keep-last "$PRUNE_KEEP" \
        --min-age-seconds "$PRUNE_MIN_AGE" --interval 120 \
        > "$TRAIN_OUT/prune_dcp.log" 2>&1 < /dev/null &
    PRUNE_PID=$!
    echo "[al50] 剪枝看门狗已启动 (pid $PRUNE_PID)，日志 $TRAIN_OUT/prune_dcp.log"
fi

# ---- TensorBoard ------------------------------------------------------------
TB_PID=""
if [ "$TB" = "1" ]; then
    mkdir -p "$TRAIN_OUT/runs"
    setsid nohup "$PY" -m tensorboard.main \
        --logdir "$TRAIN_OUT/runs" --port "$TB_PORT" --host 0.0.0.0 \
        --reload_interval 5 --samples_per_plugin scalars=100000 \
        > "$TRAIN_OUT/tensorboard.log" 2>&1 < /dev/null &
    TB_PID=$!
    echo "[al50] TensorBoard 已启动 (pid $TB_PID, 端口 $TB_PORT)，日志 $TRAIN_OUT/tensorboard.log"
    echo "[al50] ⚠️ 要在浏览器看，需在 AutoDL 控制台把端口 $TB_PORT 配成「自定义服务」"
fi

cleanup() {
    [ -n "$PRUNE_PID" ] && kill "$PRUNE_PID" 2>/dev/null || true
    [ -n "$TB_PID" ]    && kill "$TB_PID"    2>/dev/null || true
}
trap cleanup EXIT

# ---- 训练 -------------------------------------------------------------------
cd "$REPO"
export PATH="$(dirname "$PY"):$PATH"
# 🔴 不能写死单卡：多卡时按 N_GPU 推导可见卡（否则 train.sh 数出 1 卡 ⇒ torchrun 只起 1 进程；
#    AMD 上还踩过 nvidia-smi 缺失被 wc -l 数成 1 的坑）
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if [ "${N_GPU:-1}" -gt 1 ] 2>/dev/null; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPU - 1)))
    export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
  else
    export CUDA_VISIBLE_DEVICES=0
  fi
fi
export QWEN3VL_PATH=${QWEN3VL:-/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
#: 额外参数透传（空格分隔）。用于不改脚本就切换并行模式等，例如：
#:   EXTRA_ARGS="--train.data_parallel_mode ddp"
EXTRA_ARGS=${EXTRA_ARGS:-}

set +e
bash train.sh tasks/vla/train_lingbotvla.py \
    "$CONFIG" \
    --model.model_path       "$MODEL_PATH" \
    --data.train_path        "$PHASES/datasets.txt" \
    --data.episode_ids_file  "$TRAIN_IDS" \
    --data.image_augment     false \
    --train.output_dir       "$TRAIN_OUT" \
    --train.micro_batch_size "$MICRO" \
    --train.gradient_accumulation_steps "$GAS" \
    --train.global_batch_size "$GBS" \
    --train.num_train_epochs "$EPOCHS" \
    --train.max_steps        "$MAX_STEPS" \
    --train.save_steps       "$SAVE_STEPS" \
    --train.save_epochs      "$SAVE_EPOCHS" \
    --train.save_hf_weights  $SAVE_HF_BOOL \
    --train.async_save_hf_weights false \
    --train.hf_pass_interval "$([ "$SMOKE_NO_CHECKPOINT" = "1" ] && echo 0 || echo "$HF_PASS_INTERVAL")" \
    --train.hf_export_dtype "$HF_EXPORT_DTYPE" \
    --train.enable_resume    "$RESUME_BOOL" \
    --train.train_expert_only false \
    --train.freeze_vision_encoder true \
    --train.enable_mixed_precision "$MIXED" \
    --train.rmpad            false \
    --train.rmpad_with_pos_ids false \
    --train.step_offset      "$STEP_OFFSET" \
    --train.disk_guard       "$DISK_GUARD_BOOL" \
    --train.disk_guard_margin 1.1 \
    --train.disk_check_interval 50 \
    --train.smoke_no_checkpoint "$([ "$SMOKE_NO_CHECKPOINT" = "1" ] && echo true || echo false)" \
    --train.dcp_save_mode    "$DCP_MODE" \
    --train.dcp_final_size_gb "$DCP_FINAL_GB" \
    --train.dcp_keep_last "$([ "$SMOKE_NO_CHECKPOINT" = "1" ] && echo 0 || echo "$DCP_KEEP_LAST")" \
    --train.data_parallel_mode           "${DP_MODE:-fsdp2}" \
    --train.data_parallel_replicate_size 1 \
    --train.data_parallel_shard_size     "$N_GPU" \
    --train.auto_learning          "$AL_CFG" \
    --train.auto_learning_manifest "$MANIFEST" \
    $EXTRA_ARGS \
    --train.auto_learning_baseline "$BASELINE"
TRAIN_RC=$?
set -e

echo
echo "================================================================================"
echo "  训练结束 — Auto Learning"
echo "================================================================================"
echo "  存档      : $([ "$SMOKE_NO_CHECKPOINT" = "1" ] && echo "本轮不保存 DCP/HF" || echo "$TRAIN_OUT/checkpoints/global_step_*/")"
echo "  AL 事件   : $TRAIN_OUT/auto_learning_events.jsonl"
echo "  决策摘要  : grep '\"action\": \"finish\"' $TRAIN_OUT/auto_learning_events.jsonl | tail -1"
echo "  收工原因  : target_total_passed_reached(N) = 当前总 PASS 达标；all_tasks_resolved = 池子跑完了"
echo "  TensorBoard（本地隧道）:"
echo "    ssh -N -L ${TB_PORT}:127.0.0.1:${TB_PORT} <user>@<host> -p <port>"
echo "    然后浏览器打开 http://127.0.0.1:${TB_PORT}/"
echo "  TensorBoard（AutoDL 自定义服务）: 控制台 → 自定义服务 → 添加端口 ${TB_PORT}"
echo "================================================================================"

exit "${TRAIN_RC:-0}"
