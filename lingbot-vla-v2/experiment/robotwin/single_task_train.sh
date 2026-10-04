#!/usr/bin/env bash
# =============================================================================
# 单任务训练（最小探针）—— 一次只训一个 task，**不串联闭环评测**（闭环用户自己手动跑）
# -----------------------------------------------------------------------------
# 与昨晚 `phase1_L1_vit_frozen_train_then_eval.sh` 的关系：
#   冻结配置完全一致（train_expert_only=false + freeze_vision_encoder=true），
#   差异只有三点，全部是用户明确指定的：
#     ① 数据：14 个 L1 任务(700 回合) → **单个 task**(40 回合，来自 tools/task_split.py 的 train 划分)
#     ② 卡数：4 卡 micro 14 / gbs 112 → **单卡 micro 16 / gas 1 ⇒ gbs 16**
#     ③ `image_augment`：true → **false**（与官方一致）
#   ⚠️ 因为同时动了两处（数据 + image_augment），若成功**无法单独归因**；
#      补救：探针便宜，第二轮用 `AUGMENT=true` 反跑一次即可分离。
#
# 用法（任意目录均可，脚本自定位仓库根）：
#   TASK=click_bell bash experiment/robotwin/single_task_train.sh
#   bash experiment/robotwin/single_task_train.sh click_bell        # 位置参数亦可
#   DRY_RUN=1 TASK=click_bell bash experiment/robotwin/single_task_train.sh   # 只打印计划
#   MAX_STEPS=3 TASK=click_bell bash experiment/robotwin/single_task_train.sh # 显存探测（不存档）
#   MICRO=12 TASK=click_bell bash experiment/robotwin/single_task_train.sh    # OOM 时降 micro
#   PRUNE=0 TASK=click_bell bash experiment/robotwin/single_task_train.sh     # 关掉剪枝看门狗
#
# 前置：必须先有划分
#   python tools/task_split.py --task click_bell            # 默认 --strategy quantile
#
# 显存：单卡要把**全部**权重+梯度+优化器放进一张卡（FSDP2 在单卡上不分片）：
#   权重 25.5G + 梯度 23.8G + 优化器 23.7G ≈ 73G ⇒ 只剩 ~23G 给激活值。
#   micro 16 若 OOM：① 降 MICRO ② 加 `--train.enable_gradient_checkpointing true`（会变慢）
#   建议先 `MAX_STEPS=3` 跑一次确认能起来。
#
# 存档与磁盘：MAX_STEPS=1500、SAVE_EVERY=600 ⇒ 自动取「≤600 的 1500 的最大约数」= **500**
#   ⇒ 存 3 份（500/1000/1500），末份对齐不丢。单份完整 72G，3 份 = 216G > 现有 134G，
#   所以**默认开剪枝看门狗**（PRUNE=1）：每份 hf_ckpt 校验通过后剪掉 DCP ⇒ 24G/份，3 份 72G。
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- 参数 -------------------------------------------------------------------
TASK="${TASK:-${1:-}}"
[ -z "$TASK" ] && { echo "❌ 必须指定任务：TASK=<名字> bash $0   （或 $0 <名字>）" >&2; exit 1; }

SPLIT_DIR=${SPLIT_DIR:-/data/train/task_splits}
PHASES=${PHASES:-/data/train/phases}
TRAIN_OUT=${TRAIN_OUT:-/data/outputs/single/$TASK}

MICRO=${MICRO:-16}
GAS=${GAS:-1}
N_GPU=${N_GPU:-1}
GBS=$(( MICRO * GAS * N_GPU ))

MAX_STEPS=${MAX_STEPS:-1500}
SAVE_EVERY=${SAVE_EVERY:-600}
SAVE_EPOCHS=0                      # 用户指定：开了步存档就不开轮末存档
AUGMENT=${AUGMENT:-false}          # 与官方一致
PRUNE=${PRUNE:-1}                  # 剪枝看门狗（默认开：3 份完整存档放不下）
PRUNE_KEEP=${PRUNE_KEEP:-0}        # 保留几份完整 DCP（0 = 全剪，放弃续训）
DRY_RUN=${DRY_RUN:-0}

PY=/data/miniconda3/envs/lingbotvla/bin/python
[ -x "$PY" ] || { echo "❌ 找不到 $PY" >&2; exit 1; }

# ---- 读划分 -----------------------------------------------------------------
MANIFEST="$SPLIT_DIR/manifest.json"
TRAIN_IDS="$SPLIT_DIR/$TASK.train_ids.json"
[ -f "$MANIFEST" ]  || { echo "❌ 缺 $MANIFEST；先跑: python tools/task_split.py --task $TASK" >&2; exit 1; }
[ -f "$TRAIN_IDS" ] || { echo "❌ 缺 $TRAIN_IDS；先跑: python tools/task_split.py --task $TASK" >&2; exit 1; }

read -r N_TRAIN TRAIN_FRAMES N_VAL VAL_FRAMES STRATEGY VAL_RATIO < <(
    "$PY" - "$MANIFEST" "$TASK" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
t = m["tasks"].get(sys.argv[2])
if t is None:
    sys.exit(f"manifest 里没有任务 {sys.argv[2]}；先重跑 tools/task_split.py")
print(t["n_train"], t["train_frames"], t["n_val"], t["val_frames"], m["strategy"], m["val_ratio"])
PYEOF
)

STEPS_PER_EPOCH=$(( TRAIN_FRAMES / GBS ))
[ "$STEPS_PER_EPOCH" -lt 1 ] && { echo "❌ 每轮步数为 0（train_frames=$TRAIN_FRAMES < gbs=$GBS）" >&2; exit 1; }

# 目标轮数：保证 train_steps × epochs > max_steps，让 max_steps 真正驱动总步数
EPOCHS=$(( (MAX_STEPS + STEPS_PER_EPOCH - 1) / STEPS_PER_EPOCH + 1 ))

# SAVE_STEPS：取「≤ SAVE_EVERY 的 MAX_STEPS 的最大约数」⇒ 末步一定有存档
SAVE_STEPS=$("$PY" -c "
m, want = $MAX_STEPS, $SAVE_EVERY
d = [x for x in range(1, m + 1) if m % x == 0 and x <= want]
print(max(d))
")
N_SAVES=$(( MAX_STEPS / SAVE_STEPS ))

# ---- 计划 -------------------------------------------------------------------
cat <<EOF
================================================================================
  单任务训练探针 — $TASK
================================================================================
  仓库根      = $REPO
  数据划分    = $SPLIT_DIR   (strategy=$STRATEGY, val_ratio=$VAL_RATIO)
    train     = $N_TRAIN 回合 / $TRAIN_FRAMES 帧
    val       = $N_VAL 回合 / $VAL_FRAMES 帧（开环用，不参与训练）
  训练输出    = $TRAIN_OUT
  冻结配置    = train_expert_only=false + freeze_vision_encoder=true（同昨晚对照组）
  image_augment = $AUGMENT  （官方 false）
  训练规模    = ${EPOCHS} epoch × ${STEPS_PER_EPOCH} 步/轮，max_steps=${MAX_STEPS} ⇒ 总 ${MAX_STEPS} 步
  批大小      = micro ${MICRO} × gas ${GAS} × ${N_GPU} 卡 = gbs ${GBS}
  存档计划    = 每 ${SAVE_STEPS} 步一份（目标 ${SAVE_EVERY}，已对齐 max_steps）× ${N_SAVES} 份；轮末不存
  剪枝看门狗  = PRUNE=$PRUNE  keep-last=$PRUNE_KEEP  ⇒ 单份 72G → 24G
  评测        = 本脚本**不跑评测**（闭环请手动执行）
================================================================================
EOF

if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] 只打印计划，不训练。"
    exit 0
fi

# ---- 磁盘预检 ---------------------------------------------------------------
AVAIL_GB=$(df -BG --output=avail "$(dirname "$TRAIN_OUT")" 2>/dev/null | tail -1 | tr -dc '0-9')
echo "[single] 可用磁盘 ${AVAIL_GB}G；单份完整存档 ≈72G，disk_guard 门槛 ≈78.3G"
if [ "${AVAIL_GB:-0}" -lt 80 ] && [ "$PRUNE" != "1" ]; then
    echo "[single] ⚠️  空间偏紧且未开剪枝；建议 PRUNE=1 或先清理旧存档" >&2
fi

mkdir -p "$TRAIN_OUT"

# ---- 剪枝看门狗（训练期间常驻，每份 hf_ckpt 校验通过后剪 DCP）----------------
PRUNE_PID=""
if [ "$PRUNE" = "1" ]; then
    setsid nohup "$PY" -u "$REPO/tools/prune_dcp.py" \
        --ckpt-root "$TRAIN_OUT" --keep-last "$PRUNE_KEEP" \
        --min-age-seconds 300 --interval 120 \
        > "$TRAIN_OUT/prune_dcp.log" 2>&1 < /dev/null &
    PRUNE_PID=$!
    echo "[single] 剪枝看门狗已启动 (pid $PRUNE_PID)，日志 $TRAIN_OUT/prune_dcp.log"
fi
cleanup() {
    if [ -n "$PRUNE_PID" ]; then
        kill "$PRUNE_PID" 2>/dev/null || true
        echo "[single] 剪枝看门狗已停止"
    fi
}
trap cleanup EXIT

# ---- 训练 -------------------------------------------------------------------
cd "$REPO"
export PATH=/data/miniconda3/envs/lingbotvla/bin:$PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export QWEN3VL_PATH=${QWEN3VL:-/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct}

bash train.sh tasks/vla/train_lingbotvla.py \
    /data/train/configs/robotwin_official_paths.yaml \
    --model.model_path       /data/models/lingbot-vla-v2-6b-base/lingbot-vla-v2-6b \
    --data.train_path        "$PHASES/datasets.txt" \
    --data.episode_ids_file  "$TRAIN_IDS" \
    --train.output_dir       "$TRAIN_OUT" \
    --train.micro_batch_size "$MICRO" \
    --train.gradient_accumulation_steps "$GAS" \
    --train.global_batch_size "$GBS" \
    --train.num_train_epochs "$EPOCHS" \
    --train.max_steps        "$MAX_STEPS" \
    --train.save_steps       "$SAVE_STEPS" \
    --train.save_epochs      "$SAVE_EPOCHS" \
    --train.save_hf_weights  true \
    --train.async_save_hf_weights true \
    --train.enable_resume    false \
    --train.train_expert_only false \
    --train.freeze_vision_encoder true \
    --data.image_augment     "$AUGMENT" \
    --train.disk_guard true \
    --train.disk_guard_margin 1.1 \
    --train.disk_check_interval 50

echo
echo "================================================================================
  训练结束 — $TASK
================================================================================
  产出: $TRAIN_OUT/checkpoints/global_step_*/
  开环评测（手动，base 基线只跑一次）:
    python tools/task_split.py --task $TASK --pick-train 5     # train 5 条
    python tools/task_split.py --task $TASK --pick-val 10      # val 全 10 条
    python scripts/open_loop_eval.py \\
        --model_path $TRAIN_OUT/checkpoints/global_step_${MAX_STEPS}/hf_ckpt \\
        --robo_name robotwin \\
        --data_path /data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30 \\
        --traj_ids \$(python tools/task_split.py --task $TASK --pick-train 5) \$(python tools/task_split.py --task $TASK --pick-val 10) \\
        --use_length 50 --chunk_ret true \\
        --save_plot_path /data/eval_results/open_loop/single_${TASK}_${MAX_STEPS}
================================================================================"
