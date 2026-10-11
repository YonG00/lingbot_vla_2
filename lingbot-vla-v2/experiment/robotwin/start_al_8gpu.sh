#!/usr/bin/env bash
# =============================================================================
# 标准启动脚本：50 任务 Auto Learning（8 卡，ROCm / cpu1）
#
# 与文档配套：`docs/auto_learning_launch_guide_zh.md` §0.2「标准启动命令」。
# 本脚本只是把那条命令固化成可复现入口；**不含任何训练逻辑**。
#
# 用法：
#   bash experiment/robotwin/start_al_8gpu.sh                # 用默认值启动 al_v36
#   RUN_NAME=al_v37 STEPS=200 bash experiment/robotwin/start_al_8gpu.sh
#   DRY_RUN=1 bash experiment/robotwin/start_al_8gpu.sh      # 只打印命令，不启动
#   NO_CACHE=1 bash experiment/robotwin/start_al_8gpu.sh     # 全量重扫（慎用）
#
# 设计要点（每一条都是真机踩出来的，改前先看指南 §0.2）：
#   1. **AITER_USE_SYSTEM_TRITON=1 必须用 `--env` 传**：launcher 给 worker 的环境只透传
#      `AL_` 前缀 + 显式 `--env`；父进程 export 到不了训练进程 ⇒ aiter 导入失败 ⇒
#      模型注册表为空 ⇒ `Unrecognized configuration class`。
#   2. **`--workers 1` 与 `AL_HARDNESS_SHARD=1` 必须配套**：1 片 ⇒ 8 卡同一进程
#      （world_size=8），hardness 才能按 rank 切样本（`ids[rank::8]`，约 8× 加速）；
#      分片数 >1 时每片 N_GPU=1，world_size==1 会让分片自动退回单卡全量。
#   3. **输出/缓存必须落 overlay**：`/workspace` 仅约 98 G，训练日志与 checkpoint 会撑爆它。
#   4. **默认不加 `--no-cache`**：launcher 语义是「有缓存用缓存 / 缺就补扫 / 显式才全量重扫」。
#   5. 运行前必须无残留训练进程（否则 launcher 自锁，退出码 2）。
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

# ---- 可覆盖参数 -------------------------------------------------------------
RUN_NAME="${RUN_NAME:-al_v36}"
STEPS="${STEPS:-5000}"
MICRO="${MICRO:-5}"
GAS="${GAS:-1}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"          # 8 卡；某卡异常时从列表里去掉
N_CARDS=$(( $(tr -cd ',' <<<"$GPUS" | wc -c) + 1 ))   # 卡数 = 逗号数 + 1
WORKERS="${WORKERS:-1}"                   # 1 片 = 8 卡同进程（配合 AL_HARDNESS_SHARD）
PY="${PY:-/opt/robotwin-env/bin/python}"

RUNS_ROOT="${RUNS_ROOT:-/models/robotwin-persistent/al_runs}"
CACHE_ROOT="${CACHE_ROOT:-/models/robotwin-persistent/al_cache}"
TMPDIR_AL="${TMPDIR_AL:-/models/robotwin-persistent/tmp}"
SCOUT_CACHE="${SCOUT_CACHE:-/workspace/al/scout_cache/scout.json}"
HARDNESS_CACHE="${HARDNESS_CACHE:-/workspace/al/hardness_cache/hardness.json}"
MODEL_NAME="${MODEL_NAME:-robbyant_lingbot-vla-v2-6b-bf16}"

LOG="${LOG:-$RUNS_ROOT/logs/al_launch_${RUN_NAME}.log}"
NO_CACHE_FLAG=""
[ "${NO_CACHE:-0}" = "1" ] && NO_CACHE_FLAG="--no-cache"

[ -x "$PY" ] || { echo "❌ 解释器不存在：$PY" >&2; exit 2; }
[ -f "$REPO/experiment/robotwin/al_launch.py" ] || { echo "❌ 找不到 al_launch.py（REPO=$REPO）" >&2; exit 2; }

mkdir -p "$RUNS_ROOT/logs" "$CACHE_ROOT/triton" "$CACHE_ROOT/torchinductor" "$TMPDIR_AL" \
         "$(dirname "$SCOUT_CACHE")" "$(dirname "$HARDNESS_CACHE")"

# ---- 自锁：有残留训练进程就拒绝启动（与 launcher 的退出码 2 语义一致）--------
if pgrep -f "tasks/vla/train_lingbotvla.py" >/dev/null 2>&1; then
    echo "❌ 检测到残留训练进程，拒绝启动（先确认并按 PID 清理）：" >&2
    ps -eo pid,etime,args | grep "[t]rain_lingbotvla" | head -5 >&2
    exit 2
fi

CMD=(
  "$PY" -u experiment/robotwin/al_launch.py
  --run-name "$RUN_NAME" --steps "$STEPS" --micro "$MICRO" --gas "$GAS"
  --gpus "$GPUS"
  --worker-out-root     "$RUNS_ROOT"
  --triton-cache        "$CACHE_ROOT/triton"
  --torchinductor-cache "$CACHE_ROOT/torchinductor"
  --hardness-cache-file "$HARDNESS_CACHE"
  --scout-cache-file    "$SCOUT_CACHE"
  --model-name          "$MODEL_NAME"
  --env "PRUNE=1"
  --env "AITER_USE_SYSTEM_TRITON=1"      # 见文件头第 1 条：必须 --env
  --env "AL_HARDNESS_SHARD=1"            # 见文件头第 2 条：与 --workers 1 配套
  --workers "$WORKERS" --no-tb
)
[ -n "$NO_CACHE_FLAG" ] && CMD+=("$NO_CACHE_FLAG")

echo "==================== 标准启动（8 卡 Auto Learning）===================="
echo "  仓库      : $REPO"
echo "  run 名称  : $RUN_NAME     步数: $STEPS   micro=$MICRO gas=$GAS"
echo "  卡        : $GPUS   （workers=$WORKERS）"
echo "  批大小    : GBS = micro $MICRO × gas $GAS × ${N_CARDS} 卡 = $(( MICRO * GAS * N_CARDS ))"
echo "  输出根    : $RUNS_ROOT"
echo "  缓存      : scout=$SCOUT_CACHE"
echo "              hardness=$HARDNESS_CACHE"
echo "  launcher 日志: $LOG"
echo "  训练日志     : $RUNS_ROOT/logs/train_${RUN_NAME}.log"
echo "  监控         : tail -f \$训练日志"
echo "======================================================================"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY_RUN] 将要执行："
    printf '  %q' "${CMD[@]}"; echo
    exit 0
fi

cd "$REPO" || exit 2
export TMPDIR="$TMPDIR_AL"
setsid nohup "${CMD[@]}" < /dev/null > "$TMPDIR_AL/al_launch_${RUN_NAME}.out" 2>&1 &
echo "已后台启动（脱离终端）。等待 launcher 写日志：$LOG"
