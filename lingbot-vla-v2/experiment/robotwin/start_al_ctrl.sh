#!/usr/bin/env bash
# =============================================================================
# ctrl（AutoDL / NVIDIA）专用 Auto Learning 启动脚本（2026-10-14）
#
# 作用：把「从 al_v76 存档（HF 权重 + pool_state.json）续训 50 任务 AL」的整条命令固化成
#       可复现入口；**不含任何训练逻辑**，只做：预检 → 设环境 → 后台拉起 al_50task_bf16.sh。
# 取代：旧 AMD 机的 `al_launch.py … --gpus 0,2,3,…` 长命令（那套路径全是 /workspace、/opt/robotwin-env）。
#
# 用法（在 ctrl 上，仓库根 = /data/code/lingbot-vla-v2）：
#   DRY_RUN=1 bash experiment/robotwin/start_al_ctrl.sh     # 只预检 + 打印计划，不训练（上卡前必跑）
#   bash experiment/robotwin/start_al_ctrl.sh               # 后台启动（setsid nohup），立即返回
#   RUN=al_v78 ARCH=/path/to/archive bash experiment/robotwin/start_al_ctrl.sh
#
# 可覆盖变量（默认值见下）：RUN / ARCH / STEP_OFFSET / MAX_STEPS / SAVE_EVERY / MICRO / GAS / N_GPU /
#   AL_CFG / OUT / LOG / TB_PORT / MIN_FREE_GB / ALLOW_BUSY / DRY_RUN
#
# 与旧 AMD 命令的差异（每条都有理由，别随手改回去）：
#   1. 不走 al_launch.py：它的默认路径全是 AMD 的，且预扫 worker 在 NVIDIA 单机上没有收益；
#      缓存没有就由训练进程 bootstrap 现扫（用户 2026-10-14：「缓存没有到时重扫就行」）。
#   2. `AL_POOL_STATE_FILE` 必须显式给：pool_state.json 在 hf_ckpt 目录**外面**（存档根目录），
#      按 model_path 同目录查找找不到。
#   3. `STEP_OFFSET` 必须显式给：路径里没有 `global_step_N`，不会自动推断；且 `--steps` 是绝对步号。
#   4. 存档组合 = `DCP_MODE=final_only` + `SAVE_HF=1` + `HF_PASS_INTERVAL=0`（HF 内存直出，
#      不碰 DCP）。⚠️ `HF_PASS_INTERVAL` 在 al_50task_bf16.sh 里默认是 2，与 save_hf_weights=true 互斥
#      ⇒ 不传会 ValueError 启动即崩。四象限见技能 lingbot-auto-learning-logic §13 附。
#   5. 去掉了 `AL_EVAL_BATCH_STOCHASTIC_APPROVED`（代码里已无此开关）；评测批大小 = micro_batch。
#   6. 不传 `--train.enable_fp32`：ctrl 的 /data/train/configs/robotwin_official_paths.yaml 里本来就是 true。
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

# ---- 可覆盖参数 -------------------------------------------------------------
RUN="${RUN:-al_v77}"
ARCH="${ARCH:-/root/autodl-tmp/data/lingbot_archive/al_v76_step4050}"
STEP_OFFSET="${STEP_OFFSET:-4050}"
MAX_STEPS="${MAX_STEPS:-20000}"            # 绝对步号（= offset + 要跑的步数的上限）；AL 自己会按 max_rounds 收工
SAVE_EVERY="${SAVE_EVERY:-150}"
GAS="${GAS:-1}"
TB_PORT="${TB_PORT:-6007}"
AL_CFG="${AL_CFG:-configs/auto_learning/al_eval2_target50_m60_step100_h400_noreview.yaml}"
OUT="${OUT:-/data/outputs/$RUN}"
LOG="${LOG:-/data/tmp/${RUN}.log}"
PY="${PY:-/data/miniconda3/envs/lingbotvla/bin/python}"
MIN_FREE_GB="${MIN_FREE_GB:-120}"           # 输出盘最少空闲：每份 HF 直出约 12G，要留出多份 + 余量
DRY_RUN="${DRY_RUN:-0}"

MODEL_PATH="$ARCH/hf_ckpt"
POOL_FILE="$ARCH/pool_state.json"
THRESH="$REPO/refs/al_artifacts/eval_results/open_loop/ref50k/pass_thresholds_gmean60_warn.json"

FAIL=0
say_fail() { echo "❌ $*" >&2; FAIL=1; }

# ---- GPU 探测（ctrl 无卡模式下 nvidia-smi 会报 No devices）------------------
N_DETECT=0; MIN_MEM_MIB=0
if command -v nvidia-smi >/dev/null 2>&1; then
    MEMS="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | grep -E '^[0-9]+' || true)"
    if [ -n "$MEMS" ]; then
        N_DETECT="$(echo "$MEMS" | wc -l | tr -d ' ')"
        MIN_MEM_MIB="$(echo "$MEMS" | sort -n | head -1)"
    fi
fi
N_GPU="${N_GPU:-$N_DETECT}"
[ "${N_GPU:-0}" -ge 1 ] 2>/dev/null || say_fail "没有可见 GPU（nvidia-smi 报无设备？当前是无卡模式——先开卡再跑）"

# micro 默认值随显存走；显存不足 80G 时不替用户猜，要求显式给
if [ -z "${MICRO:-}" ]; then
    if [ "$MIN_MEM_MIB" -ge 80000 ] 2>/dev/null; then
        MICRO=24
    else
        say_fail "显存 ${MIN_MEM_MIB} MiB < 80G：请显式给 MICRO（96G 卡上用 24；48G 卡 fp32 训练需另行标定）"
        MICRO=0
    fi
fi
GBS=$(( MICRO * GAS * ${N_GPU:-0} ))

# ---- 预检 -------------------------------------------------------------------
[ -x "$PY" ] || say_fail "解释器不存在：$PY"
[ -f "$REPO/experiment/robotwin/al_50task_bf16.sh" ] || say_fail "找不到 al_50task_bf16.sh（REPO=$REPO）"
[ -f "$REPO/$AL_CFG" ] || say_fail "AL 配置不存在：$REPO/$AL_CFG"
[ -f "$THRESH" ] || say_fail "阈值表不存在：$THRESH（应随仓库入库）"
[ -f /data/train/task_splits_50/manifest.json ] || say_fail "缺 /data/train/task_splits_50/manifest.json"
[ -f /data/train/task_splits_50/task_baseline.json ] || say_fail "缺 /data/train/task_splits_50/task_baseline.json"
[ -d /data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct ] || say_fail "缺 Qwen3-VL tokenizer 目录"

# 存档完整性：3 片 safetensors + config + 池子；且 rsync 必须已结束（否则第 3 片可能只传了一半）
for f in model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors config.json; do
    [ -s "$MODEL_PATH/$f" ] || say_fail "存档缺文件：$MODEL_PATH/$f"
done
[ -s "$POOL_FILE" ] || say_fail "池子文件缺失：$POOL_FILE"
if pgrep -f "[r]sync .*lingbot_archive" >/dev/null 2>&1; then
    say_fail "存档 rsync 仍在进行，等传完再启动（否则 safetensors 可能不完整）"
fi
if [ -s "$POOL_FILE" ]; then
    POOL_STEP="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('global_step'))" "$POOL_FILE" 2>/dev/null || echo '?')"
    if [ "$POOL_STEP" != "$STEP_OFFSET" ]; then
        say_fail "pool_state.global_step=$POOL_STEP 与 STEP_OFFSET=$STEP_OFFSET 不一致（编号会脱节）"
    fi
fi
# 输出目录：不覆盖已有产物
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    say_fail "输出目录非空：$OUT（换 RUN 名，别覆盖旧产物）"
fi
# 空间：先建父目录再量
mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")" /data/al_cache /data/outputs/al_pool_mirror 2>/dev/null
FREE_GB="$(df -BG --output=avail "$(dirname "$OUT")" 2>/dev/null | tail -1 | tr -dc '0-9')"
if [ -n "$FREE_GB" ] && [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
    say_fail "输出盘空闲 ${FREE_GB}G < ${MIN_FREE_GB}G：先清理旧存档（HF 直出无自动保留策略，每份约 12G）"
fi
# 不与别的训练并跑
if [ "${ALLOW_BUSY:-0}" != "1" ] && pgrep -f "[t]asks/vla/train_lingbotvla.py" >/dev/null 2>&1; then
    say_fail "检测到训练进程在跑（ALLOW_BUSY=1 可强制）"
fi

# ---- 计划 -------------------------------------------------------------------
cat <<EOF
================================================================================
  ctrl AL 启动计划（${RUN}）  DRY_RUN=${DRY_RUN}
================================================================================
  仓库        = $REPO   （HEAD $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)，分支 $(git -C "$REPO" branch --show-current 2>/dev/null)）
  GPU         = ${N_GPU} 张，最小显存 ${MIN_MEM_MIB} MiB
  批大小      = micro ${MICRO} × gas ${GAS} × ${N_GPU} 卡 = GBS ${GBS}
  起点模型    = $MODEL_PATH
  池子        = $POOL_FILE（step=${POOL_STEP:-?}）
  编号起点    = STEP_OFFSET=${STEP_OFFSET}；MAX_STEPS=${MAX_STEPS}（绝对步号）
  AL 配置     = $AL_CFG
  阈值表      = $THRESH
  存档        = 每 ${SAVE_EVERY} 步 HF 直出 → $OUT/checkpoints/global_step_N/hf_ckpt（final_only，不写 DCP）
  输出 / 日志 = $OUT  /  $LOG  （输出盘空闲 ${FREE_GB:-?}G）
  TensorBoard = 端口 ${TB_PORT}
================================================================================
EOF
[ "$FAIL" = "0" ] || { echo "❌ 预检未通过，退出（未启动任何东西）" >&2; exit 2; }

# ---- 环境 -------------------------------------------------------------------
export QWEN3VL_PATH=/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct
export AL_POOL_STATE_FILE="$POOL_FILE"
export AL_POOL_STATE_MIRROR=/data/outputs/al_pool_mirror   # 默认值 /workspace/al/pool_state 在 ctrl 上不存在
export AL_EVAL_BATCH_MODE=auto
export AL_EVAL_BATCH_APPROVED=1
export AL_MODEL_NAME="${RUN}_from_$(basename "$ARCH")"
export AL_HARDNESS_CACHE_FILE=/data/al_cache/hardness.json
export TMPDIR=/data/tmp
if [ "$N_GPU" -gt 1 ]; then
    export AL_HARDNESS_SHARD=1      # hardness 帧级分片（单卡时自动退回全量，这里仅多卡打开）
    export AL_EVAL_SHARD=1          # 开环评测轨迹分片（world>1 才生效；单卡口径不受影响）
fi

# 传给 al_50task_bf16.sh 的变量（该脚本全部走环境变量）
ENVS=(
  "TB_PORT=$TB_PORT" "MICRO=$MICRO" "GAS=$GAS" "N_GPU=$N_GPU"
  "AL_CFG=$AL_CFG" "TRAIN_OUT=$OUT" "MODEL_PATH=$MODEL_PATH"
  "STEP_OFFSET=$STEP_OFFSET" "MAX_STEPS=$MAX_STEPS" "SAVE_EVERY=$SAVE_EVERY"
  "DCP_MODE=final_only" "SAVE_HF=1" "HF_PASS_INTERVAL=0"
)

cd "$REPO" || exit 2
if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] 预检通过。下面由 al_50task_bf16.sh 打印它自己的计划（DRY_RUN=1，不训练）："
    env "${ENVS[@]}" DRY_RUN=1 bash experiment/robotwin/al_50task_bf16.sh 2>&1 | head -80
    exit 0
fi

setsid nohup env "${ENVS[@]}" bash experiment/robotwin/al_50task_bf16.sh > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "✅ 已后台启动：pid=$PID  日志=$LOG"
echo "   看日志：tail -f $LOG"
echo "   面板  ：bash tools/auto_learning_pipeline/al_watch.sh $RUN   （AL_ROOT 视 ctrl 目录而定）"
