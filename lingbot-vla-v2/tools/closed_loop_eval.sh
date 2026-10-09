#!/usr/bin/env bash
# =============================================================================
# 单任务闭环评测一键脚本（单 ckpt）
#
# 用法（在 /data/code/lingbot-vla-v2 下）：
#   bash tools/closed_loop_eval.sh                                    # 默认 click_bell / demo_clean / 10 回合
#   CONFIG=demo_randomized EPISODES=5 bash tools/closed_loop_eval.sh  # 随机化 5 回合
#   STEP=1000 bash tools/closed_loop_eval.sh                          # 评另一份 ckpt
#   CKPT_ROOT=/data/outputs/single/click_bell_cont500 STEP=1000 bash tools/closed_loop_eval.sh
#
# 环境变量：
#   TASK       任务名（默认 click_bell；必须在 RoboTwin 的 50 个官方任务里）
#   TASKS      多任务（逗号或空格分隔），**一次调用**跑完 ⇒ inference server 常驻、
#              模型**只加载一次**（底层 launcher 的 queue 模式：跑完一个任务腾出槽位、
#              直接接下一个，不重启 server）。与 TASK 二选一；给了 TASKS 就忽略 TASK。
#              例：TASKS="place_fan,stack_blocks_two" EPISODES=10 CKPT_ROOT=... bash ...
#   CONFIG     RoboTwin task_config（默认 demo_clean）—— 合法值 = task_config/*.yml：
#              demo_clean / demo_randomized（**不是** clean/randomized）
#   EPISODES   每个任务的回合数（默认 10；官方客户端默认是 100）
#   CKPT_ROOT  训练输出目录（默认 /data/outputs/single/$TASK）
#   STEP       评哪一步的 ckpt（默认 500）
#   TAG        输出子目录名（默认 step${STEP}_${CONFIG#demo_}）；
#              单任务落 ${TAG}/${TASK}/，多任务落 ${TAG}/${各任务}/（布局一致）
#   PRECISION  推理精度：fp32（默认，与历史一致）| bf16。低于 fp32 时必须与权重精度一致
#              （bf16 权重要用 bf16 服务；本项目 bf16 训练的 HF 导出默认就是 bf16）
#   PORT       起始端口（默认 9330）
#   DRY_RUN    1 = 只打印计划
#
# 🔴 两个必须遵守的约定（都实测踩过）：
#   1. `--model_path` 必须是 **<CKPT_ROOT>/checkpoints/global_step_N/hf_ckpt**。
#      `deploy/lingbot_vla_v2_policy.py:276` 找的是 `<model_path>/../../../lingbotvla_cli.yaml`
#      ⇒ 把 ckpt 挪到别的目录（如 run1_step_500/）会直接挂。本脚本会**前置断言**这一点。
#   2. 必须 **export QWEN3VL_PATH**（launcher 默认值是占位符 `/path/to/your/...`）
#      ⇒ 不导出会让 4 个推理 server 全 DEAD、sim 侧 0 步。
# =============================================================================
set -euo pipefail

REPO=/data/code/lingbot-vla-v2
TASK=${TASK:-click_bell}
TASKS=${TASKS:-}          # 非空 ⇒ 多任务一次调用（server 常驻、模型只加载一次）
CONFIG=${CONFIG:-demo_clean}
EPISODES=${EPISODES:-10}
CKPT_ROOT=${CKPT_ROOT:-/data/outputs/single/$TASK}
STEP=${STEP:-500}
PORT=${PORT:-9330}
PRECISION=${PRECISION:-fp32}
QWEN3VL=${QWEN3VL:-/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct}
ROBOTWIN_DIR=${ROBOTWIN_DIR:-/data/code/RoboTwin-lingbot}
CONDA_SH=${CONDA_SH:-/data/miniconda3/etc/profile.d/conda.sh}
TAG=${TAG:-step${STEP}_${CONFIG#demo_}}
DRY_RUN=${DRY_RUN:-0}

[[ "$PRECISION" == "bf16" || "$PRECISION" == "fp32" ]] \
    || die "PRECISION 只支持 fp32|bf16，收到 $PRECISION"
if [[ "$PRECISION" == "bf16" ]]; then USE_BF16=true; USE_FP32=false; else USE_BF16=false; USE_FP32=true; fi

hr() { printf '%.0s─' {1..78}; echo; }
die() { echo "❌ $*" >&2; exit 1; }

cd "$REPO" || die "仓库目录不存在: $REPO"
[[ -f "$CONDA_SH" ]] || die "找不到 $CONDA_SH"
source "$CONDA_SH" && conda activate lingbotvla

# ---- ① ckpt 层级断言（最容易踩）------------------------------------------------
CKPT="$CKPT_ROOT/checkpoints/global_step_${STEP}/hf_ckpt"
[[ -f "$CKPT/model.safetensors.index.json" ]] \
    || die "ckpt 不完整或缺 index.json: $CKPT
   （若 ckpt 被挪到别处，先移回：mkdir -p $CKPT_ROOT/checkpoints &&
     mv <旧位置> $CKPT_ROOT/checkpoints/global_step_${STEP}）"
YAML="$(dirname "$(dirname "$(dirname "$CKPT")")")/lingbotvla_cli.yaml"
[[ -f "$YAML" ]] || die "找不到 $YAML
   ⇒ launcher（deploy/lingbot_vla_v2_policy.py:276）要求 --model_path 是
     <CKPT_ROOT>/checkpoints/global_step_N/hf_ckpt，当前层级不对"

# ---- ② 任务清单（多任务时全部写进一个清单 ⇒ launcher 排队跑，server 不重启）----
if [[ -n "$TASKS" ]]; then
    TASK_LIST_STR="$(printf '%s' "$TASKS" | tr ',' ' ')"
    read -r -a _task_arr <<< "$TASK_LIST_STR"
    N_TASKS=${#_task_arr[@]}
    TL=/data/train/task_splits/${N_TASKS}task.eval.txt
    printf '%s\n' "${_task_arr[@]}" > "$TL"
    TASK_DESC="$TASK_LIST_STR
  任务个数    : $N_TASKS（一次调用 ⇒ 模型只加载一次、server 常驻排队换任务）"
else
    N_TASKS=1
    TL=/data/train/task_splits/${TASK}.eval.txt
    printf '%s\n' "$TASK" > "$TL"
    TASK_DESC="$TASK"
fi

# ---- ③ Qwen3-VL backbone（必须 export）---------------------------------------
[[ -d "$QWEN3VL" ]] || die "QWEN3VL 路径不存在: $QWEN3VL"
export QWEN3VL_PATH="$QWEN3VL"
[[ -d "$ROBOTWIN_DIR" ]] || die "RoboTwin 仓库不存在: $ROBOTWIN_DIR"

if [[ "$N_TASKS" -gt 1 ]]; then
    OUT=/data/eval_results/closed_loop/${TAG}      # launcher 会按任务建子目录
else
    OUT=/data/eval_results/closed_loop/${TAG}/${TASK}
fi
mkdir -p "$OUT"

hr
cat <<EOF
  任务        : $TASK_DESC
  条件        : $CONFIG   （每任务回合数 $EPISODES）
  ckpt        : $CKPT
  输出        : $OUT
  端口        : $PORT  （1 GPU × 1 server）
  推理精度    : $PRECISION  （use_bf16=$USE_BF16 / use_fp32=$USE_FP32）
  QWEN3VL_PATH: $QWEN3VL
  视频        : 关（--no_video）
EOF
hr
if [ "$DRY_RUN" = "1" ]; then echo "[dry-run] 只打印计划"; exit 0; fi

# ---- ④ 启动（前台，日志同时进 $OUT/launch.log）--------------------------------
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
    --model_path "$CKPT" \
    --output_base "$OUT" \
    --start_port "$PORT" \
    --task_list_file "$TL" \
    --task_config "$CONFIG" \
    --num_gpus 1 --num_per_gpu 1 \
    --use_fp32 "$USE_FP32" --use_bf16 "$USE_BF16" --use_compile false \
    --no_video \
    --test_num "$EPISODES" \
    --eval_workdir "$ROBOTWIN_DIR" \
    --conda_sh "$CONDA_SH" \
    2>&1 | tee "$OUT/launch.log"

RC=${PIPESTATUS[0]}
hr
echo "评测结束 rc=$RC"
find "$OUT" -name '_result.txt' -exec sh -c 'echo "--- $1 ---"; cat "$1"' _ {} \; 2>/dev/null || true
exit "$RC"
