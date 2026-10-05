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
# 精度：true = F32 权重（**现状**；`train_lingbotvla.py:422` 里 enable_mixed_precision=True ⇒ torch_dtype=float32）
#       false = bf16 权重（显存约省一半，但**数值设置变了** ⇒ 与 F32 跑的结果不可直接比）
MIXED=${MIXED:-true}
PRUNE=${PRUNE:-1}                  # 剪枝看门狗（默认开：3 份完整存档放不下）
PRUNE_KEEP=${PRUNE_KEEP:-0}        # 保留几份完整 DCP（0 = 全剪，放弃续训）
# 剪枝看门狗的「静默期」：文件 mtime 距今小于它就不剪（怕剪到正在写的）。
# ⚠️ 2026-10-05 实测：300s 太慢 —— disk_guard 会在 prune 动手前就判定空间不足而停训
#    （差 11 秒）。急着回收空间时设小一点（如 60）。
PRUNE_MIN_AGE=${PRUNE_MIN_AGE:-300}
# 从 <TRAIN_OUT>/checkpoints/global_step_* 里**最大**的那份续训（train_lingbotvla.py:722）
RESUME=${RESUME:-0}
RESUME_BOOL=$([ "$RESUME" = "1" ] && echo true || echo false)
# 初始权重。默认 base 模型；指向某个 `hf_ckpt` 即可「接着那份权重继续训」
# （⚠️ 这不是 resume —— optimizer / LR 调度会重置，见 docs）
MODEL_PATH=${MODEL_PATH:-/data/models/lingbot-vla-v2-6b-base/lingbot-vla-v2-6b}

# 🔴 2026-10-05 血的教训：看门狗**先于训练启动**，而 resume 要读的正是
#   `<TRAIN_OUT>/checkpoints` 里**最大**那份 ckpt。若 PRUNE_KEEP=0（全剪）且 min_age 很小，
#   看门狗会在启动那一秒就把那份 DCP 剪掉 ⇒ 训练 load 时
#   `FileNotFoundError: .../global_step_N/model/.metadata`（实测踩过）。
#   ⇒ RESUME=1 时强制保留最新一份完整 DCP。
if [ "$RESUME" = "1" ] && [ "$PRUNE" = "1" ] && [ "$PRUNE_KEEP" -lt 1 ]; then
    echo "[single] ⚠️  RESUME=1 ⇒ 强制 PRUNE_KEEP=1" >&2
    echo "[single]     （否则看门狗会剪掉正要恢复的那份 ckpt，导致 load 失败）" >&2
    PRUNE_KEEP=1
fi
DRY_RUN=${DRY_RUN:-0}

# 训练中原地 open-loop validation（见 lingbotvla/utils/open_loop_validation.py）
OPEN_LOOP_EVAL_STEPS=${OPEN_LOOP_EVAL_STEPS:-0}   # 0 = 关闭；如 250
OPEN_LOOP_TRAIN_IDS=${OPEN_LOOP_TRAIN_IDS:-}      # 空 = 用模块内置的 5 条 train-monitor
OPEN_LOOP_VAL_IDS=${OPEN_LOOP_VAL_IDS:-}          # 空 = 用模块内置的 10 条 held-out val
STOP_AND_SAVE_FILE=${STOP_AND_SAVE_FILE:-}        # 空 = <TRAIN_OUT>/STOP_AND_SAVE
SKIP_FINAL_SAVE=${SKIP_FINAL_SAVE:-0}             # ⚠️ 仅供 smoke test：1 = max_steps 到顶时跳过收尾存档

PY=/data/miniconda3/envs/lingbotvla/bin/python
[ -x "$PY" ] || { echo "❌ 找不到 $PY" >&2; exit 1; }

# ---- 训练中 open-loop validation 的透传参数 ---------------------------------
OPEN_LOOP_ARGS=()
if [ "${OPEN_LOOP_EVAL_STEPS}" -gt 0 ] 2>/dev/null; then
    OPEN_LOOP_ARGS+=(--train.open_loop_eval_steps "${OPEN_LOOP_EVAL_STEPS}")
    [ -n "${OPEN_LOOP_TRAIN_IDS}" ] && OPEN_LOOP_ARGS+=(--train.open_loop_train_ids "${OPEN_LOOP_TRAIN_IDS}")
    [ -n "${OPEN_LOOP_VAL_IDS}" ]   && OPEN_LOOP_ARGS+=(--train.open_loop_val_ids "${OPEN_LOOP_VAL_IDS}")
fi
[ -n "${STOP_AND_SAVE_FILE}" ] && OPEN_LOOP_ARGS+=(--train.stop_and_save_file "${STOP_AND_SAVE_FILE}")
[ "${SKIP_FINAL_SAVE}" = "1" ] && OPEN_LOOP_ARGS+=(--train.skip_final_save_on_max_steps true)

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

# SAVE_STEPS：取「≤ SAVE_EVERY 的 MAX_STEPS 的最大约数」⇒ 末步一定有存档。
# ⚠️ SAVE_EVERY<=0 ⇒ 关掉步存档（SAVE_STEPS=0）。smoke test 需要「零存档」时用这个 +
#    SKIP_FINAL_SAVE=1（拦收尾存档）：否则 MAX_STEPS=3 会让推导式取到 3，在 step 3 白存 72G。
if [ "${SAVE_EVERY}" -le 0 ] 2>/dev/null; then
    SAVE_STEPS=0
    N_SAVES=1
else
    SAVE_STEPS=$("$PY" -c "
m, want = $MAX_STEPS, $SAVE_EVERY
d = [x for x in range(1, m + 1) if m % x == 0 and x <= want]
print(max(d) if d else 0)
")
    N_SAVES=$(( SAVE_STEPS > 0 ? MAX_STEPS / SAVE_STEPS : 1 ))
fi

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
  精度        = MIXED=$MIXED（true=F32 权重 / false=bf16 权重）
  image_augment = $AUGMENT  （官方 false）
  训练规模    = ${EPOCHS} epoch × ${STEPS_PER_EPOCH} 步/轮，max_steps=${MAX_STEPS} ⇒ 总 ${MAX_STEPS} 步
  批大小      = micro ${MICRO} × gas ${GAS} × ${N_GPU} 卡 = gbs ${GBS}
  存档计划    = 每 ${SAVE_STEPS} 步一份（目标 ${SAVE_EVERY}，已对齐 max_steps）× ${N_SAVES} 份；轮末不存
  剪枝看门狗  = PRUNE=$PRUNE  keep-last=$PRUNE_KEEP  min-age=${PRUNE_MIN_AGE}s  ⇒ 单份 72G → 24G
  续训        = RESUME=$RESUME（$RESUME_BOOL）⇒ 从 $TRAIN_OUT/checkpoints 里最大那份接着跑
  初始权重    = $MODEL_PATH
  开环验证    = 每 ${OPEN_LOOP_EVAL_STEPS} 步一次（0=关）；train_ids=${OPEN_LOOP_TRAIN_IDS:-<内置5条>}  val_ids=${OPEN_LOOP_VAL_IDS:-<内置10条>}
  STOP_AND_SAVE = ${STOP_AND_SAVE_FILE:-<TRAIN_OUT>/STOP_AND_SAVE}
  跳过收尾存档 = ${SKIP_FINAL_SAVE}（1=跳过；⚠️ 仅供 smoke test，正式训练保持 0）
  评测        = 本脚本**不跑闭环评测**（闭环请手动执行）
================================================================================
EOF

if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] 只打印计划，不训练。"
    exit 0
fi

# ---- 磁盘预检 ---------------------------------------------------------------
# ⚠️ 必须先建出 TRAIN_OUT 再 df：`df <不存在的路径>` 返回非 0，配合 `set -e` + `pipefail`
#    会把整个脚本静默干掉（smoke test 踩过：TRAIN_OUT=/data/outputs/smoke/s1 的父目录不存在）
mkdir -p "$TRAIN_OUT"
AVAIL_GB=$(df -BG --output=avail "$TRAIN_OUT" 2>/dev/null | tail -1 | tr -dc '0-9' || true)
echo "[single] 可用磁盘 ${AVAIL_GB:-?}G；单份完整存档 ≈72G，disk_guard 门槛 ≈78.3G"
if [ -n "${AVAIL_GB}" ] && [ "${AVAIL_GB}" -lt 80 ] && [ "$PRUNE" != "1" ]; then
    echo "[single] ⚠️  空间偏紧且未开剪枝；建议 PRUNE=1 或先清理旧存档" >&2
fi

# ---- 剪枝看门狗（训练期间常驻，每份 hf_ckpt 校验通过后剪 DCP）----------------
PRUNE_PID=""
if [ "$PRUNE" = "1" ]; then
    setsid nohup "$PY" -u "$REPO/tools/prune_dcp.py" \
        --ckpt-root "$TRAIN_OUT" --keep-last "$PRUNE_KEEP" \
        --min-age-seconds "$PRUNE_MIN_AGE" --interval 120 \
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
# 单卡上可训参数多（对照组 5.96B ⇒ 权重+梯度+优化器 ≈73G），碎片会吃掉最后几个 G。
# micro 16 实测 OOM（2026-10-04 smoke），打开 expandable_segments 减少碎片后再看是否需要降 micro。
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

bash train.sh tasks/vla/train_lingbotvla.py \
    /data/train/configs/robotwin_official_paths.yaml \
    --model.model_path       "$MODEL_PATH" \
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
    --train.enable_resume    "$RESUME_BOOL" \
    --train.train_expert_only false \
    --train.freeze_vision_encoder true \
    --train.enable_mixed_precision "$MIXED" \
    --data.image_augment     "$AUGMENT" \
    --train.disk_guard true \
    --train.disk_guard_margin 1.1 \
    --train.disk_check_interval 50 \
    "${OPEN_LOOP_ARGS[@]+"${OPEN_LOOP_ARGS[@]}"}"
TRAIN_RC=$?      # 🔴 必须立刻接住：后面还有 echo/banner，否则脚本会返回 0 把失败吞掉
                 #    （2026-10-05 实测：训练因 resume 失败当场死，脚本却报 rc=0）

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

# 🔴 把训练的真实退出码传出去。不加这句的话，脚本的退出码 = 上面最后一个 echo 的（永远 0），
#    调用方/自动化会把「训练当场崩掉」误判成成功（2026-10-05 实测）。
exit "${TRAIN_RC:-0}"
