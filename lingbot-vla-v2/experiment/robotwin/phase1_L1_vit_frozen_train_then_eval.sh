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
# 评测：多 ckpt 调度器，L1 sentinel 4 任务 × clean 3 回合 = 12 回合
#       ⚠️ **默认只测 clean**（randomized 关）—— 与实验组保持完全一致的口径
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
#   SAVE_STEPS=1169 bash experiment/robotwin/phase1_L1_vit_frozen_train_then_eval.sh # 只存 2 份（省 71G）
#   SAVE_STEPS=0    bash experiment/robotwin/phase1_L1_vit_frozen_train_then_eval.sh # 只存轮末 1 份
#   CONDITIONS=clean,randomized bash experiment/robotwin/phase1_L1_vit_frozen_train_then_eval.sh # 恢复官方口径
#   TB=1            bash experiment/robotwin/phase1_L1_vit_frozen_train_then_eval.sh # 顺带后台起 TensorBoard(:6006)
#
# 磁盘预算（2337 步 / 3 epoch，**单份存档 ≈ 71.4G**，无轮转不清理）：
#   单份构成（对照组算值，与实验组 55G 实测交叉验证，误差 0.2%）：
#     model/      23.75G  DCP fp32 权重       ← 只服务续训
#     optimizer/  23.70G  Muon+AdamW 状态      ← 只服务续训（实验组只有 7.2G）
#     hf_ckpt/    23.75G  HF 格式 fp32 权重    ← **评测唯一需要的**
#     extra_state/ ~0.2G  调度器/RNG/dataloader
#   注：权重是 F32（config `enable_fp32: true`，safetensors 头部实测），不是 bf16。
#   | save_steps | 存档点                  | 份数 | 合计占用 | disk_guard 门槛 | 210G 可用时 |
#   |    0       | 2337                    |  1   |  ~72G   |      ~79G      | 富余 131G   |
#   |   1169     | 1169, 2337              |  2   | ~143G   |     ~150G      | 富余  60G   |
#   |    779     | 779, 1558, 2337         |  3   | ~214G   |     ~221G      | ⚠️ 差 11G   |  ← 默认（779 与实验组同预算可直接比）
#   |    584     | 584,1168,1752,2336,2337 |  5   | ~357G   |     ~364G      | ⚠️ 差 154G  |
#   份数规则：N = floor(2337 / save_steps) + (2337 % save_steps ? 1 : 0)；
#     save_steps=0 ⇒ N=1。不整除时轮末会**再补一份**，所以最后两份可能挨得很近
#     （如 584 ⇒ 2336 与 2337），这是公式的固有行为，不是 bug。
#
# 省盘：PRUNE_DCP=1 会拉起 tools/prune_dcp.py，剪掉旧存档里评测用不到的 DCP，
#   单份从 71.4G 降到 23.9G ⇒ 3 份 hf_ckpt + 最新 1 份完整 DCP = 119.5G（富余 90G）。
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

# 轮末存档的 epoch 周期。
# 🔴 **不能恒为 1**：源码里「按步存档」(`train_lingbotvla.py:1131`) 与「轮末存档」(`:1253`)
#    互不去重（`:1211` 的 already_saved **只保护 reached_max_steps 那条路**，而 max_steps=50000 走不到）。
#    本脚本默认 SAVE_STEPS=779 恰好 = 779 步/epoch ⇒ 两个分支在 779/1558/2337 **全部命中同一目录**，
#    每个 epoch 边界把 DCP 重写一遍 —— 实测多花 385s / 378s（约 6.3 分钟），3 轮白烧 ~19 分钟。
#    （HF 侧有去重，日志 `[async_hf] skip duplicate checkpoint`；只有 DCP 重复写。）
# 规则：步存档已覆盖轮末 ⇒ 关掉轮末存档；否则只在**最后一个** epoch 末补一份，保证一定有收尾存档。
SAVE_EPOCHS=$EPOCHS
if [ "$SAVE_STEPS" -gt 0 ] && [ $(( TOTAL_STEPS % SAVE_STEPS )) -eq 0 ]; then
    SAVE_EPOCHS=0
fi

# 本次一共会存几份（与下面 disk_guard 预检共用，避免两处各算一遍算歪）
if [ "$SAVE_STEPS" -gt 0 ] 2>/dev/null; then
    if [ $(( TOTAL_STEPS % SAVE_STEPS )) -eq 0 ]; then
        N_SAVES=$(( TOTAL_STEPS / SAVE_STEPS ))      # 末步已被步存档覆盖，轮末不再额外存
    else
        N_SAVES=$(( TOTAL_STEPS / SAVE_STEPS + 1 ))  # 轮末补最后一份
    fi
else
    N_SAVES=1                                        # 只存轮末一份
fi

# 评测条件。默认**只测 clean**，与实验组 phase1_train_then_eval.sh 完全一致：
#   A/B 要回答的是「换冻结范围有没有效果」，不是泛化能力。randomized 会把回合数翻倍，
#   而 L1 sentinel 只有 4 任务 × 3 回合 = 12 回合，样本本就小，再加一路只会摊薄信号。
#   等 clean 上看出方向了，再用 CONDITIONS=clean,randomized 补官方口径。
CONDITIONS=${CONDITIONS:-clean}

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
echo "[phase${PHASE}-vitfrozen] 评测条件  = $CONDITIONS  (randomized 关掉时只看「有没有效果」，不看泛化)"
echo "[phase${PHASE}-vitfrozen] 批大小    = micro ${MICRO} × gas ${GAS} × 4 卡 = gbs ${GBS}"
echo "[phase${PHASE}-vitfrozen] 存档计划  = 每 ${SAVE_STEPS} 步一次(0=关) + 轮末每 ${SAVE_EPOCHS} 轮一次 ⇒ 共 ${N_SAVES} 份"
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
    echo "[phase${PHASE}-vitfrozen] ⚠️  $TRAIN_OUT/checkpoints 已存在："
    ls -1 "$TRAIN_OUT/checkpoints" | sed "s/^/[phase${PHASE}-vitfrozen]    /"
    echo "[phase${PHASE}-vitfrozen] ⚠️  enable_resume=false 会从 base 重新训练并覆盖同名目录。" >&2
    echo "[phase${PHASE}-vitfrozen] ⚠️  想接着上次跑就把 --train.enable_resume 改成 true。" >&2
fi

# ---------------------------------------------------------------------------
# 预检 ②：当前可用空间 vs 本次要存几份 checkpoint（不阻塞，只提醒）
#   单份 71.4G = model/ 23.75 + optimizer/ 23.70 + hf_ckpt/ 23.75 + extra ~0.2
#   （对照组可训 5.961B，优化器状态比实验组的 7.2G 大三倍多）
#   ⚠️ 真正的门槛不是「总和」，是 disk_guard 的逐步判据：
#      required = max_used × 1.1，存第 k 份前需 avail >= 单份 × 1.1
#      ⇒ 能存下 N 份的条件是 avail >= (N - 1) × 单份 + 单份 × 1.1
#      3 份 ⇒ 需可用 ≥ 2×71.4 + 78.5 = 221.3G（**不是** 214G）
#   N_SAVES 已在上面算好，这里直接用。
# ---------------------------------------------------------------------------
SAVE_GB=72
AVAIL_GB=$(df -BG --output=avail "$(dirname "$TRAIN_OUT")" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$AVAIL_GB" ]; then
    NEED_GB=$(( N_SAVES * SAVE_GB ))
    GUARD_GB=$(( (N_SAVES - 1) * SAVE_GB + SAVE_GB * 11 / 10 ))
    echo "[phase${PHASE}-vitfrozen] /data 可用 ${AVAIL_GB}G；本次计划存档 ${N_SAVES} 份 × ${SAVE_GB}G = ${NEED_GB}G"
    echo "[phase${PHASE}-vitfrozen] disk_guard 实际门槛：存最后一份前需可用 ≥ ${GUARD_GB}G"
    if [ "$AVAIL_GB" -lt "$GUARD_GB" ]; then
        echo "[phase${PHASE}-vitfrozen] ⚠️  差 $(( GUARD_GB - AVAIL_GB ))G：disk_guard 会在放不下时优雅停止训练，" >&2
        echo "[phase${PHASE}-vitfrozen] ⚠️  届时最后一个 checkpoint 会缺失。三条出路：" >&2
        echo "[phase${PHASE}-vitfrozen] ⚠️    a) 调大 SAVE_STEPS（如 1169 ⇒ 2 份，门槛 150G）" >&2
        echo "[phase${PHASE}-vitfrozen] ⚠️    b) 先扩容" >&2
        echo "[phase${PHASE}-vitfrozen] ⚠️    c) PRUNE_DCP=1 剪掉旧存档的 DCP（3 份只需 119.5G）" >&2
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
    --train.save_epochs      "$SAVE_EPOCHS" \
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
    --conditions "$CONDITIONS" \
    --max-parallel-checkpoints 2 \
    --num-gpus 4 --num-per-gpu 1 \
    --output-base "$EVAL_OUT" \
    2>&1 | tee "$EVAL_OUT/scheduler.log"
