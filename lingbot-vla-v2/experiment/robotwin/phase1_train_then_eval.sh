#!/usr/bin/env bash
# =============================================================================
# 阶段 1（L1）训练 → 训练结束后自动串联 RoboTwin 闭环评测
# -----------------------------------------------------------------------------
# 起点：**base 预训练模型**（阶段 1 天然从 base 起，目录名不带后缀）
# 产出：/data/outputs/phase1_L1
#
# 训练：官方 train.sh + torchrun，4 卡，expert-only，micro=28 / gbs=112
#       700 回合 / 87,266 帧 → 779 步 / 1 epoch，约 43 分钟
# 评测：多 ckpt 调度器，L1 sentinel 4 任务 × clean 3 回合 = 12 回合，约 3.5 分钟
#       ⚠️ **默认只测 clean**（randomized 关）—— 见下方 CONDITIONS 说明
#
# 用法（在任意目录均可，脚本自己定位仓库根）：
#   bash experiment/robotwin/phase1_train_then_eval.sh              # 真跑
#   DRY_RUN=1 bash experiment/robotwin/phase1_train_then_eval.sh    # 只预览评测计划
#   SAVE_STEPS=260 bash experiment/robotwin/phase1_train_then_eval.sh   # 3 个 ckpt
#   SAVE_STEPS=0   bash experiment/robotwin/phase1_train_then_eval.sh   # 只存轮末 1 个
#   CONDITIONS=clean,randomized bash experiment/robotwin/phase1_train_then_eval.sh  # 恢复官方口径
#   TB=1           bash experiment/robotwin/phase1_train_then_eval.sh   # 顺带后台起 TensorBoard(:6006)
#
# 磁盘预算（779 步 / 1 epoch，**单份存档 ≈ 55G**，无轮转不清理）：
#   单份 = model/ 23.75G + optimizer/ 7.2G + hf_ckpt/ 23.75G + extra ~0.2G
#   （expert-only 只训 1.938B ⇒ 优化器状态比对照组的 23.7G 小得多）
#   | save_steps | 存档点        | 份数 | 约占用 |
#   |    0       | 779           |  1   |  ~55G  |
#   |   545      | 545, 779      |  2   | ~110G  |  ← 默认
#   |   260      | 260, 520, 779 |  3   | ~165G  |
#
# 详见 docs/phase_train_then_eval_guide.md
# =============================================================================
set -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/../.." && pwd)

PHASES=/data/train/phases
TRAIN_OUT=${TRAIN_OUT:-/data/outputs/phase1_L1}
EVAL_OUT=${EVAL_OUT:-/data/eval_results/phase1_L1}
QWEN3VL=${QWEN3VL:-/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct}

PHASE=1
TOTAL_STEPS=779
EPISODE_IDS=phase1_L1.episode_ids.json
# 每多少 step 存一次（0 = 只在轮末存一次）
SAVE_STEPS=${SAVE_STEPS:-545}

# 评测条件。默认**只测 clean**：
#   当前阶段要回答的是「换冻结范围到底有没有效果」，不是泛化能力。
#   randomized 会把回合数翻倍，而 L1 sentinel 只有 4 个任务 × 3 回合 = 12 回合，
#   样本本来就小，再加一路条件只会把信号摊薄、把噪声放大。
#   等 clean 上看出方向了，再用 CONDITIONS=clean,randomized 补官方口径。
CONDITIONS=${CONDITIONS:-clean}

cd "$REPO" || exit 1
export PATH=/data/miniconda3/envs/lingbotvla/bin:$PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
# 评测侧必需：launcher 的默认值是占位符 /path/to/your/checkpoints/...
export QWEN3VL_PATH="$QWEN3VL"

echo "[phase${PHASE}] 仓库根    = $REPO"
echo "[phase${PHASE}] 训练输出  = $TRAIN_OUT"
echo "[phase${PHASE}] 评测输出  = $EVAL_OUT"
echo "[phase${PHASE}] 评测条件  = $CONDITIONS  (randomized 关掉时只看「有没有效果」，不看泛化)"

# ---------------------------------------------------------------------------
# 只预览评测计划（不训练、不启动子进程、不初始化 CUDA）
# ---------------------------------------------------------------------------
if [ -n "${DRY_RUN:-}" ]; then
    python experiment/robotwin/robotwin_multi_ckpt_eval.py \
        --ckpt-root "$TRAIN_OUT" \
        --phase "$PHASE" --episodes 3 \
        --conditions "$CONDITIONS" \
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
#   单份 54.9G = model/ 23.75 + optimizer/ 7.2 + hf_ckpt/ 23.75 + extra ~0.2
#   （expert-only 只训 1.938B，优化器状态远小于对照组的 23.7G）
#   ⚠️ disk_guard 判据是 max_used × margin，不是总和：存第 k 份前需
#      avail >= 单份 × 1.1，所以「能存下 N 份」的条件是 avail >= (N-1)×单份 + 单份×1.1
# ---------------------------------------------------------------------------
SAVE_GB=55
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
    NEED_GB=$(( N_SAVES * SAVE_GB ))
    GUARD_GB=$(( (N_SAVES - 1) * SAVE_GB + SAVE_GB * 11 / 10 ))
    echo "[phase${PHASE}] /data 可用 ${AVAIL_GB}G；本次计划存档 ${N_SAVES} 份 × ${SAVE_GB}G = ${NEED_GB}G"
    echo "[phase${PHASE}] disk_guard 实际门槛：存最后一份前需可用 ≥ ${GUARD_GB}G"
    if [ "$AVAIL_GB" -lt "$GUARD_GB" ]; then
        echo "[phase${PHASE}] ⚠️  差 $(( GUARD_GB - AVAIL_GB ))G：disk_guard 会在放不下时优雅停止训练，" >&2
        echo "[phase${PHASE}] ⚠️  届时最后一个 checkpoint 会缺失。三条出路：" >&2
        echo "[phase${PHASE}] ⚠️    a) 调大 SAVE_STEPS 少存几份" >&2
        echo "[phase${PHASE}] ⚠️    b) 先扩容" >&2
        echo "[phase${PHASE}] ⚠️    c) PRUNE_DCP=1 剪掉旧存档里评测用不到的 DCP（单份降到 23.9G）" >&2
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
    --model.model_path       /data/models/lingbot-vla-v2-6b-base/lingbot-vla-v2-6b \
    --data.train_path        "$PHASES/datasets.txt" \
    --data.episode_ids_file  "$PHASES/$EPISODE_IDS" \
    --train.output_dir       "$TRAIN_OUT" \
    --train.micro_batch_size 28 \
    --train.gradient_accumulation_steps 1 \
    --train.global_batch_size 112 \
    --train.num_train_epochs 1 \
    --train.max_steps        50000 \
    --train.save_steps       "$SAVE_STEPS" \
    --train.save_epochs      1 \
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
    --conditions "$CONDITIONS" \
    --max-parallel-checkpoints 2 \
    --num-gpus 4 --num-per-gpu 1 \
    --output-base "$EVAL_OUT" \
    2>&1 | tee "$EVAL_OUT/scheduler.log"
