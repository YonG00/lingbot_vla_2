#!/usr/bin/env bash
# =============================================================================
# 阶段 2（L1+L2）训练【直接基于 base 模型】 → 训练结束后自动串联 RoboTwin 闭环评测
# -----------------------------------------------------------------------------
# 起点：**base 预训练模型**（不是阶段 1 的产出！）
# 产出：/data/outputs/phase2_L1_L2_from_base
#
#   ⚠️ 命名约定：`phase2_L1_L2`（无后缀）留给「base → 阶段1 → 阶段2」的续训链；
#      本脚本是「base → 阶段2」，所以目录名带 `_from_base` 后缀，两者不会互相覆盖。
#
# 「基于 base」由三个条件共同保证（详见 docs/phase_train_then_eval_guide.md 第 7 节）：
#   ① 全新的 output_dir        ② --train.enable_resume false
#   ③ --model.model_path 指向 base（本脚本显式写出）
#   注意：`enable_resume` 只控制「是否续训 DCP」，**不参与初始权重**；
#         初始权重只来自 model.model_path。
#
# 训练：官方 train.sh + torchrun，4 卡，expert-only，micro=28 / gbs=112
#       1450 回合 / 207,872 帧 → 1856 步 / 1 epoch，约 1.7 小时
# 评测：多 ckpt 调度器，L1+L2 sentinel 8 任务 × (clean 3 + randomized 3) 回合
#       约 14 分钟/ckpt；3 个 ckpt 两波并发 ≈ 30 分钟
#
# 用法（在任意目录均可，脚本自己定位仓库根）：
#   bash experiment/robotwin/phase2_from_base_train_then_eval.sh              # 真跑
#   DRY_RUN=1 bash experiment/robotwin/phase2_from_base_train_then_eval.sh    # 只预览
#   SAVE_STEPS=928 bash experiment/robotwin/phase2_from_base_train_then_eval.sh  # 2 个 ckpt
#   SAVE_STEPS=0   bash experiment/robotwin/phase2_from_base_train_then_eval.sh  # 只存轮末 1 个
#   TB=1           bash experiment/robotwin/phase2_from_base_train_then_eval.sh  # 顺带起 TensorBoard(:6006)
#
# 磁盘预算（1856 步 / 1 epoch，单次存档 ≈ 55G = hf_ckpt 24G + DCP ~30G，无轮转不清理）：
#   | save_steps | 存档点               | 份数 | 约占用 |
#   |    0       | 1856                 |  1   |  ~55G  |
#   |   928      | 928, 1856            |  2   | ~110G  |
#   |   619      | 619, 1238, 1856      |  3   | ~165G  |  ← 默认（= ceil(1856/3)）
#   |   464      | 464, 928, 1392, 1856 |  4   | ~224G  |  ← 330G 盘也偏紧
#   注意：阶段 1 用的 545 在这里会变成 4 份（~224G），因为步数从 779 涨到 1856。
#
# 详见 docs/phase_train_then_eval_guide.md
# =============================================================================
set -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/../.." && pwd)

PHASES=/data/train/phases
TRAIN_OUT=${TRAIN_OUT:-/data/outputs/phase2_L1_L2_from_base}
EVAL_OUT=${EVAL_OUT:-/data/eval_results/phase2_L1_L2_from_base}
MODEL_PATH=${MODEL_PATH:-/data/models/lingbot-vla-v2-6b-base/lingbot-vla-v2-6b}
QWEN3VL=${QWEN3VL:-/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct}

PHASE=2
EPOCHS=1
TOTAL_STEPS=1856
EPISODE_IDS=phase2_L1_L2.episode_ids.json
# 每多少 step 存一次（0 = 只在轮末存一次）
SAVE_STEPS=${SAVE_STEPS:-619}

# 轮末存档周期。🔴 **不能恒为 1**：源码里「按步存档」(`train_lingbotvla.py:1131`) 与
#   「轮末存档」(`:1253`) 互不去重（`:1211` 的 already_saved **只保护 reached_max_steps 那条路**，
#   而 max_steps=50000 永远走不到），所以 SAVE_STEPS 整除 TOTAL_STEPS 时会把**同一个目录写两遍**。
#   规则：步存档已覆盖末步 ⇒ 关掉轮末存档；否则只在**最后一个** epoch 末补一份，保证有收尾存档。
#   本脚本默认 619 不整除 1856 ⇒ 行为与旧的硬编码 1 完全一致；但 SAVE_STEPS=928 / 464 时
#   （两者都整除 1856）旧写法会在 1856 处重复写一遍，现在不会了。
SAVE_EPOCHS=$EPOCHS
if [ "$SAVE_STEPS" -gt 0 ] && [ $(( TOTAL_STEPS % SAVE_STEPS )) -eq 0 ]; then
    SAVE_EPOCHS=0
fi

cd "$REPO" || exit 1
export PATH=/data/miniconda3/envs/lingbotvla/bin:$PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
# 评测侧必需：launcher 的默认值是占位符 /path/to/your/checkpoints/...
export QWEN3VL_PATH="$QWEN3VL"

echo "[phase${PHASE}] 仓库根    = $REPO"
echo "[phase${PHASE}] 起点权重  = $MODEL_PATH"
echo "[phase${PHASE}] 训练输出  = $TRAIN_OUT"
echo "[phase${PHASE}] 评测输出  = $EVAL_OUT"

# ---------------------------------------------------------------------------
# 只预览评测计划（不训练、不启动子进程、不初始化 CUDA）
# ---------------------------------------------------------------------------
if [ -n "${DRY_RUN:-}" ]; then
    python experiment/robotwin/robotwin_multi_ckpt_eval.py \
        --ckpt-root "$TRAIN_OUT" \
        --phase "$PHASE" --episodes 3 \
        --conditions clean,randomized \
        --max-parallel-checkpoints 2 \
        --num-gpus 4 --num-per-gpu 1 \
        --output-base "$EVAL_OUT" \
        --dry-run
    exit $?
fi

# ---------------------------------------------------------------------------
# 预检 ①：output_dir 已存在时会覆盖同名 global_step_* —— 提醒一下
# ---------------------------------------------------------------------------
if [ -d "$TRAIN_OUT/checkpoints" ]; then
    echo "[phase${PHASE}] ⚠️  $TRAIN_OUT/checkpoints 已存在："
    ls -1 "$TRAIN_OUT/checkpoints" | sed "s/^/[phase${PHASE}]    /"
    echo "[phase${PHASE}] ⚠️  enable_resume=false 会从 base 重新训练并覆盖同名目录。" >&2
    echo "[phase${PHASE}] ⚠️  想接着上次跑就把 --train.enable_resume 改成 true。" >&2
fi

# ---------------------------------------------------------------------------
# 预检 ②：当前可用空间 vs 本次要存几份 checkpoint（不阻塞，只提醒）
# ---------------------------------------------------------------------------
AVAIL_GB=$(df -BG --output=avail "$(dirname "$TRAIN_OUT")" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$AVAIL_GB" ]; then
    if [ "$SAVE_STEPS" -gt 0 ] 2>/dev/null; then
        if [ $(( TOTAL_STEPS % SAVE_STEPS )) -eq 0 ]; then
            N_SAVES=$(( TOTAL_STEPS / SAVE_STEPS ))      # 轮末恰好撞上步存档, 不额外多一份
        else
            N_SAVES=$(( TOTAL_STEPS / SAVE_STEPS + 1 ))
        fi
    else
        N_SAVES=1
    fi
    echo "[phase${PHASE}] /data 可用 ${AVAIL_GB}G；本次计划存档 ${N_SAVES} 份，约需 $(( N_SAVES * 56 ))G"
    if [ "$AVAIL_GB" -lt $(( N_SAVES * 56 )) ]; then
        echo "[phase${PHASE}] ⚠️  空间可能不足：disk_guard 会在放不下时优雅停止训练，" >&2
        echo "[phase${PHASE}] ⚠️  届时最后一个 checkpoint 会缺失。请调大 SAVE_STEPS 或先扩容。" >&2
    fi
fi

# ---------------------------------------------------------------------------
# 可选：后台启动 TensorBoard（TB=1 时；默认关）
#   训练**只会写** <TRAIN_OUT>/runs/ 下的事件文件（rank0 写），不会自己起 tensorboard，
#   所以这里给个显式开关。不设 TB 时行为与不加这段完全一致。
# ---------------------------------------------------------------------------
TB_PORT=${TB_PORT:-6006}
if [ -n "${TB:-}" ]; then
    mkdir -p "$TRAIN_OUT/runs"
    # 端口探测：本机没有 ss / netstat，用 bash 内建 /dev/tcp（零依赖）
    _tb_busy() { (exec 3<>"/dev/tcp/127.0.0.1/${TB_PORT}") 2>/dev/null; }
    if _tb_busy; then
        echo "[phase${PHASE}] TensorBoard 已在 127.0.0.1:${TB_PORT} 运行，跳过启动"
    else
        setsid nohup tensorboard --logdir "$TRAIN_OUT/runs" \
            --port "$TB_PORT" --host 127.0.0.1 \
            > "$TRAIN_OUT/tensorboard.log" 2>&1 < /dev/null &
        for _ in $(seq 15); do _tb_busy && break; sleep 1; done
        if _tb_busy; then
            echo "[phase${PHASE}] TensorBoard 已启动: http://127.0.0.1:${TB_PORT}  (logdir=$TRAIN_OUT/runs)"
        else
            echo "[phase${PHASE}] ⚠️  TensorBoard 似乎没起来，看 $TRAIN_OUT/tensorboard.log" >&2
        fi
        echo "[phase${PHASE}] 本机浏览器访问需先做端口转发:"
        echo "[phase${PHASE}]   ssh -L ${TB_PORT}:127.0.0.1:${TB_PORT} -p <SSH端口> root@<主机>"
    fi
fi

# ---------------------------------------------------------------------------
# ① 训练（官方 train.sh；从 base 起 + 课程数据配比 + 磁盘保护）
#    && 串联：训练非 0 退出就**不会**进入评测
# ---------------------------------------------------------------------------
bash train.sh tasks/vla/train_lingbotvla.py \
    /data/train/configs/robotwin_official_paths.yaml \
    --model.model_path       "$MODEL_PATH" \
    --data.train_path        "$PHASES/datasets.txt" \
    --data.episode_ids_file  "$PHASES/$EPISODE_IDS" \
    --train.output_dir       "$TRAIN_OUT" \
    --train.micro_batch_size 28 \
    --train.gradient_accumulation_steps 1 \
    --train.global_batch_size 112 \
    --train.num_train_epochs "$EPOCHS" \
    --train.max_steps        50000 \
    --train.save_steps       "$SAVE_STEPS" \
    --train.save_epochs      "$SAVE_EPOCHS" \
    --train.save_hf_weights  true \
    --train.async_save_hf_weights true \
    --train.enable_resume    false \
    --train.train_expert_only true \
    --data.image_augment     true \
    --train.disk_guard true \
    --train.disk_guard_margin 1.1 \
    --train.disk_check_interval 50 \
&& \
mkdir -p "$EVAL_OUT" \
&& \
python experiment/robotwin/robotwin_multi_ckpt_eval.py \
    --ckpt-root "$TRAIN_OUT" \
    --phase "$PHASE" --episodes 3 \
    --conditions clean,randomized \
    --max-parallel-checkpoints 2 \
    --num-gpus 4 --num-per-gpu 1 \
    --output-base "$EVAL_OUT" \
    2>&1 | tee "$EVAL_OUT/scheduler.log"
