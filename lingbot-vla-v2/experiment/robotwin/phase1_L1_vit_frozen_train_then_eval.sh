#!/usr/bin/env bash
# =============================================================================
# 阶段 1（L1）**对照组**：只冻结视觉编码器（ViT）训练 → 训练结束后自动串联 RoboTwin 闭环评测
# -----------------------------------------------------------------------------
# 与 phase1_train_then_eval.sh（expert-only）构成 A/B 对照，**唯一变量是「冻结范围」**：
#
#   | 目录                     | train_expert_only | freeze_vision_encoder | 在训参数        |
#   |--------------------------|-------------------|-----------------------|-----------------|
#   | phase1_L1        (实验组)| true              | false                 | 1.938B (30.4%)  |
#   | phase1_L1_vit_frozen (本) | false            | true                  | 5.961B (93.5%)  |
#
# 其余一切（数据配比、gbs=112、lr/optimizer/调度、image_augment、存档策略）完全一致。
# 冻结语义（modeling_lingbot_vla_v2.py L197-205 set_requires_grad）：
#   freeze_vision_encoder=true → self.qwenvl.visual.eval() + 其参数 requires_grad=False
#   （self.qwenvl.visual 是 Qwen3VLForConditionalGeneration 的 property → self.model.visual）
#   ⇒ 只冻 ViT 0.415B(6.5%)，LLM backbone 4.022B + 动作专家 1.787B + 动作头 0.151B 全部在训
#
# 起点：base 预训练模型
# 产出：/data/outputs/phase1_L1_vit_frozen
#
# 训练：官方 train.sh + torchrun，4 卡，micro=14 / gas=2 / gbs=112，**3 个 epoch**
#       700 回合 / 87,266 帧 → 779 步/epoch × 3 = 2337 步，约 1.7~2.1 小时
# 评测：多 ckpt 调度器，L1 sentinel 4 任务 × (clean 3 + randomized 3) 回合
#
# ⚠️ micro 为什么从 28 降到 14（实验组的 28 在这个配置下会 OOM）：
#   可训参数 1.938B → 5.961B（3.08×），优化器状态/梯度显存同步上涨，实测基准 85.6G 放不下。
#   **gbs 保持 112 不变** —— arguments.py:661 `dataloader_batch_size = global_batch_size // data_parallel_size`
#   只跟 gbs 与卡数有关，**与 micro 无关** ⇒ 每 epoch 步数恒为 779、LR 调度 horizon 恒为 2337，
#   与实验组逐项可比。micro×gas 恒 = 28 ⇒ (28,1) (14,2) (7,4) (4,7) 都是 gbs=112。
#
# 用法（在任意目录均可，脚本自己定位仓库根）：
#   bash experiment/robotwin/phase1_L1_vit_frozen_train_then_eval.sh            # 真跑
#   DRY_RUN=1 bash experiment/robotwin/phase1_L1_vit_frozen_train_then_eval.sh  # 只预览评测计划
#   MICRO=7 GAS=4 bash experiment/robotwin/phase1_L1_vit_frozen_train_then_eval.sh   # OOM 时降 micro
#   SAVE_STEPS=1169 bash experiment/robotwin/phase1_L1_vit_frozen_train_then_eval.sh # 只存 2 份（省 55G）
#   SAVE_STEPS=0    bash experiment/robotwin/phase1_L1_vit_frozen_train_then_eval.sh # 只存轮末 1 份
#   TB=1            bash experiment/robotwin/phase1_L1_vit_frozen_train_then_eval.sh # 顺带后台起 TensorBoard(:6006)
#
# 磁盘预算（2337 步 / 3 epoch，单次存档 ≈ 55G = hf_ckpt 24G + DCP ~30G，无轮转不清理）：
#   | save_steps | 存档点               | 份数 | 约占用 |
#   |    0       | 2337                 |  1   |  ~55G  |
#   |   1169     | 1169, 2337           |  2   | ~110G  |
#   |    779     | 779, 1558, 2337      |  3   | ~165G  |  ← 默认（每 epoch 一份，779 与实验组同预算可直接比）
#   |    584     | 584,1168,1752,2337   |  4   | ~220G  |  ⚠️ 需先扩容
#
# 详见 docs/phase1_vit_frozen_guide.md
# =============================================================================
set -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/../.." && pwd)

PHASES=/data/train/phases
TRAIN_OUT=${TRAIN_OUT:-/data/outputs/phase1_L1_vit_frozen}
EVAL_OUT=${EVAL_OUT:-/data/eval_results/phase1_L1_vit_frozen}
QWEN3VL=${QWEN3VL:-/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct}

PHASE=1
EPOCHS=${EPOCHS:-3}
STEPS_PER_EPOCH=779
TOTAL_STEPS=$(( STEPS_PER_EPOCH * EPOCHS ))
EPISODE_IDS=phase1_L1.episode_ids.json
# 每多少 step 存一次（0 = 只在轮末存一次）。默认 779 = 每个 epoch 末存一份。
SAVE_STEPS=${SAVE_STEPS:-779}

# micro × gas 恒 = 28 ⇒ gbs 恒 = 28 × 4 卡 = 112，与实验组一致
MICRO=${MICRO:-14}
GAS=${GAS:-2}
GBS=$(( MICRO * GAS * 4 ))

cd "$REPO" || exit 1
export PATH=/data/miniconda3/envs/lingbotvla/bin:$PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
# 评测侧必需：launcher 的默认值是占位符 /path/to/your/checkpoints/...
export QWEN3VL_PATH="$QWEN3VL"

echo "[phase${PHASE}-vitfrozen] 仓库根    = $REPO"
echo "[phase${PHASE}-vitfrozen] 训练输出  = $TRAIN_OUT"
echo "[phase${PHASE}-vitfrozen] 评测输出  = $EVAL_OUT"
echo "[phase${PHASE}-vitfrozen] 冻结配置  = train_expert_only=false + freeze_vision_encoder=true"
echo "[phase${PHASE}-vitfrozen] 在训参数  = 5.961B / 6.376B (93.5%)  ← 实验组是 1.938B (30.4%)"
echo "[phase${PHASE}-vitfrozen] 训练规模  = ${EPOCHS} epoch × ${STEPS_PER_EPOCH} 步 = ${TOTAL_STEPS} 步"
echo "[phase${PHASE}-vitfrozen] 批大小    = micro ${MICRO} × gas ${GAS} × 4 卡 = gbs ${GBS}"
if [ "$GBS" -ne 112 ]; then
    echo "[phase${PHASE}-vitfrozen] ⚠️  gbs=${GBS} ≠ 112，与实验组不可比（micro×gas 应恒等于 28）" >&2
fi

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
    echo "[phase${PHASE}-vitfrozen] ⚠️  $TRAIN_OUT/checkpoints 已存在："
    ls -1 "$TRAIN_OUT/checkpoints" | sed "s/^/[phase${PHASE}-vitfrozen]    /"
    echo "[phase${PHASE}-vitfrozen] ⚠️  enable_resume=false 会从 base 重新训练并覆盖同名目录。" >&2
    echo "[phase${PHASE}-vitfrozen] ⚠️  想接着上次跑就把 --train.enable_resume 改成 true。" >&2
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
    echo "[phase${PHASE}-vitfrozen] /data 可用 ${AVAIL_GB}G；本次计划存档 ${N_SAVES} 份，约需 $(( N_SAVES * 56 ))G"
    if [ "$AVAIL_GB" -lt $(( N_SAVES * 56 )) ]; then
        echo "[phase${PHASE}-vitfrozen] ⚠️  空间可能不足：disk_guard 会在放不下时优雅停止训练，" >&2
        echo "[phase${PHASE}-vitfrozen] ⚠️  届时最后一个 checkpoint 会缺失。请调大 SAVE_STEPS 或先扩容。" >&2
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
        echo "[phase${PHASE}-vitfrozen] TensorBoard 已在 127.0.0.1:${TB_PORT} 运行，跳过启动"
    else
        setsid nohup tensorboard --logdir "$TRAIN_OUT/runs" \
            --port "$TB_PORT" --host 127.0.0.1 \
            > "$TRAIN_OUT/tensorboard.log" 2>&1 < /dev/null &
        for _ in $(seq 15); do _tb_busy && break; sleep 1; done
        if _tb_busy; then
            echo "[phase${PHASE}-vitfrozen] TensorBoard 已启动: http://127.0.0.1:${TB_PORT}  (logdir=$TRAIN_OUT/runs)"
        else
            echo "[phase${PHASE}-vitfrozen] ⚠️  TensorBoard 似乎没起来，看 $TRAIN_OUT/tensorboard.log" >&2
        fi
        echo "[phase${PHASE}-vitfrozen] 本机浏览器访问需先做端口转发:"
        echo "[phase${PHASE}-vitfrozen]   ssh -L ${TB_PORT}:127.0.0.1:${TB_PORT} -p <SSH端口> root@<主机>"
    fi
fi

# ---------------------------------------------------------------------------
# ① 训练（官方 train.sh；从 base 起 + 课程数据配比 + 磁盘保护）
#    && 串联：训练非 0 退出就**不会**进入评测
#    与实验组唯一的差异：train_expert_only false + freeze_vision_encoder true
#                        + num_train_epochs 3 + micro/gas（保 gbs 不变）
# ---------------------------------------------------------------------------
bash train.sh tasks/vla/train_lingbotvla.py \
    /data/train/configs/robotwin_official_paths.yaml \
    --model.model_path       /data/models/lingbot-vla-v2-6b-base/lingbot-vla-v2-6b \
    --data.train_path        "$PHASES/datasets.txt" \
    --data.episode_ids_file  "$PHASES/$EPISODE_IDS" \
    --train.output_dir       "$TRAIN_OUT" \
    --train.micro_batch_size "$MICRO" \
    --train.gradient_accumulation_steps "$GAS" \
    --train.global_batch_size 112 \
    --train.num_train_epochs "$EPOCHS" \
    --train.max_steps        50000 \
    --train.save_steps       "$SAVE_STEPS" \
    --train.save_epochs      1 \
    --train.save_hf_weights  true \
    --train.async_save_hf_weights true \
    --train.enable_resume    false \
    --train.train_expert_only false \
    --train.freeze_vision_encoder true \
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
