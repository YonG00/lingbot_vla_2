#!/usr/bin/env bash
# =============================================================================
# 用官方脚本(train_full_sft.sh)跑「我们的代码 + 我们的权重」——零拷贝软链骨架
#
# 背景：官方启动器把一切路径从 ROBOTWIN_ROOT 派生：data / experiments/.../source /
#       experiments/.../models / training/CONFIG。因此只要把 ROBOTWIN_ROOT 指向一个
#       自建骨架（全软链），就能在不改官方脚本的前提下换成我们的仓库与 bf16 权重。
#
# 用法：
#   bash tools/rocm/make_official_scaffold.sh                     # 只建骨架 + 自检
#   ROBOTWIN_ROOT_SRC=/RoboTwin GPU_COUNT=8 MICRO=12 GBS=96 MAX_STEPS=12 \
#     bash tools/rocm/make_official_scaffold.sh --run              # 建好并直接启动官方脚本
#   ... --run --dry-run                                            # 只打印将要执行的命令
#
# 关键环境变量（都有默认值，按需覆盖）：
#   SCAFFOLD=/workspace/rt_scaffold      骨架位置
#   OFFICIAL_ROOT=/RoboTwin              官方镜像内的根（提供 training/ 与 data/）
#   OUR_REPO=/workspace/lingbot_vla_2/lingbot-vla-v2      我们的代码（SOURCE_DIR 指向它）
#   BF16_DIR=/workspace/models/robbyant_lingbot-vla-v2-6b-bf16   bf16 权重（BASE_MODEL）
#   QWEN_DIR=/workspace/models/Qwen3-VL-4B-Instruct-config-tokenizer
#   F32_DIR=/models/robotwin-persistent/models/robbyant_lingbot-vla-v2-6b
#   MODEL_ENV=/opt/robotwin-env
#   GPU_COUNT / MICRO / GBS / MAX_STEPS / SAVE_STEPS / TEACHER_MODE / OPTIMIZER / OUTPUT_DIR
# =============================================================================
set -euo pipefail

SCAFFOLD="${SCAFFOLD:-/workspace/rt_scaffold}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/RoboTwin}"
OUR_REPO="${OUR_REPO:-/workspace/lingbot_vla_2/lingbot-vla-v2}"
BF16_DIR="${BF16_DIR:-/workspace/models/robbyant_lingbot-vla-v2-6b-bf16}"
QWEN_DIR="${QWEN_DIR:-/workspace/models/Qwen3-VL-4B-Instruct-config-tokenizer}"
F32_DIR="${F32_DIR:-/models/robotwin-persistent/models/robbyant_lingbot-vla-v2-6b}"
MOGE_FILE="${MOGE_FILE:-/models/robotwin-persistent/models/moge-2-vitb-normal/model.pt}"
MODEL_ENV="${MODEL_ENV:-/opt/robotwin-env}"

GPU_COUNT="${GPU_COUNT:-8}"
MICRO="${MICRO:-12}"
GAS="${GAS:-1}"
GBS="${GBS:-$((MICRO * GAS * GPU_COUNT))}"
MAX_STEPS="${MAX_STEPS:-12}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
TEACHER_MODE="${TEACHER_MODE:-full}"
OPTIMIZER="${OPTIMIZER:-adamw}"
OUTPUT_DIR="${OUTPUT_DIR:-/models/robotwin-persistent/outputs/official_scaffold_run}"
LOG_FILE="${LOG_FILE:-/workspace/runtime/outputs/logs/official_scaffold.log}"

E="${SCAFFOLD}/experiments/lingbot_vla_v2_6b_robotwin"

# ---------- ① 补齐 bf16 模型目录（加载器需要 assets/depth/dino_video 同时存在）----------
for d in assets depth dino_video; do
  [ -e "${BF16_DIR}/${d}" ] || ln -s "${F32_DIR}/${d}" "${BF16_DIR}/${d}"
done

# ---------- ② 搭骨架（全软链，零拷贝）----------
mkdir -p "${E}/source" "${E}/models/moge-2-vitb-normal" "${E}/training"
ln -sfn "${OFFICIAL_ROOT}/data"                                   "${SCAFFOLD}/data"
ln -sfn "${OUR_REPO}"                                             "${E}/source/lingbot-vla-v2"
ln -sfn "${BF16_DIR}"                                             "${E}/models/robbyant_lingbot-vla-v2-6b"
ln -sfn "${QWEN_DIR}"                                             "${E}/models/Qwen3-VL-4B-Instruct-config-tokenizer"
ln -sfn "${MOGE_FILE}"                                            "${E}/models/moge-2-vitb-normal/model.pt"
ln -sfn "${OFFICIAL_ROOT}/experiments/lingbot_vla_v2_6b_robotwin/training/lingbotvla_cli.yaml" "${E}/training/lingbotvla_cli.yaml"
ln -sfn "${OFFICIAL_ROOT}/experiments/lingbot_vla_v2_6b_robotwin/training/train_full_sft.sh"    "${E}/training/train_full_sft.sh"

# ---------- ③ 自检（官方脚本会 test -e 这些）----------
echo "=== 骨架自检（$SCAFFOLD）==="
fail=0
check() { if [ -e "$1" ]; then echo "  OK   ${1#$SCAFFOLD/}"; else echo "  MISS $1"; fail=1; fi; }
check "${SCAFFOLD}/data/robotwin_demo_clean_joint_v30.txt"
check "${E}/source/lingbot-vla-v2/tasks/vla/train_lingbotvla.py"
check "${E}/models/robbyant_lingbot-vla-v2-6b/model.safetensors.index.json"
check "${E}/models/robbyant_lingbot-vla-v2-6b/depth/model.pt"
check "${E}/models/robbyant_lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth"
check "${E}/models/robbyant_lingbot-vla-v2-6b/dino_video/config.yaml"
check "${E}/models/robbyant_lingbot-vla-v2-6b/assets"
check "${E}/models/Qwen3-VL-4B-Instruct-config-tokenizer/config.json"
check "${E}/models/moge-2-vitb-normal/model.pt"
check "${E}/training/lingbotvla_cli.yaml"
check "${E}/training/train_full_sft.sh"
echo "  python: $([ -x "${MODEL_ENV}/bin/python" ] && echo OK || echo MISS)"
echo "  我们的代码: $(readlink -f "${E}/source/lingbot-vla-v2")"
echo "  权重:       $(readlink -f "${E}/models/robbyant_lingbot-vla-v2-6b")"
[ "$fail" = 0 ] || { echo "❌ 骨架不完整"; exit 2; }

# ---------- ④ 启动官方脚本 ----------
LAUNCH=(env ROBOTWIN_ROOT="${SCAFFOLD}" MODEL_ENV="${MODEL_ENV}" GPU_COUNT="${GPU_COUNT}"
  MICRO_BATCH_SIZE="${MICRO}" GLOBAL_BATCH_SIZE="${GBS}" MAX_STEPS="${MAX_STEPS}"
  SAVE_STEPS="${SAVE_STEPS}" TEACHER_MODE="${TEACHER_MODE}" OPTIMIZER="${OPTIMIZER}"
  OUTPUT_DIR="${OUTPUT_DIR}" LOG_FILE="${LOG_FILE}"
  bash "${E}/training/train_full_sft.sh")

echo
echo "=== 将要执行 ==="
printf '  %q' "${LAUNCH[@]}"; echo
echo "  日志: ${LOG_FILE}"
[ "${1:-}" = "--run" ] || { echo "（未加 --run，只建骨架）"; exit 0; }
[ "${2:-}" = "--dry-run" ] && exit 0
mkdir -p "$(dirname "${LOG_FILE}")" "${OUTPUT_DIR}"
"${LAUNCH[@]}"
