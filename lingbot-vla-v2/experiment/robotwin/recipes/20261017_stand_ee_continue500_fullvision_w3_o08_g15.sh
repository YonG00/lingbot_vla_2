#!/usr/bin/env bash
# place_object_stand / step500 -> +500; launch timestamp is generated at runtime.
#
# 调优策略：
#   1. 从本次 step500 HF 继续，不重置任务数据口径。
#   2. 解冻视觉编码器（修正上一轮 freeze_vision_encoder=true 的配置错误）。
#   3. 保持 low_noise_fm + target_change_window + 独立 val。
#   4. EE loss = position 3.0 / orientation 0.8 / gripper 1.5。
#   5. micro=24, GAS=1, 新增500步，每250步保存BF16 HF，不写DCP。
#
# 默认只预览命令；EXECUTE=1 才会后台启动。SHUTDOWN=1 才允许正常结束后关机。
set -euo pipefail

ROOT="${ROOT:-/data/experiments/learn_skill_mix_task/lingbot-vla-v2}"
FROM_HF="${FROM_HF:-/data/outputs/al_stand_ee_lnfm_9680_add500_20261017_184252/checkpoints/global_step_500/hf_ckpt}"
SOURCE_RUN="${SOURCE_RUN:-/data/outputs/al_stand_ee_lnfm_9680_add500_20261017_184252}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
DEFAULT_RUN_NAME="al_stand_ee_fullvision_w3_o08_g15_from500_add500_${RUN_STAMP}"
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
OUT="${OUT:-/data/outputs/${RUN_NAME}}"

CMD=(
  "${PYTHON:-/data/miniconda3/envs/lingbotvla/bin/python}"
  "$ROOT/experiment/robotwin/continue_skill_training.py"
  --from-hf "$FROM_HF"
  --steps 500
  --micro 24
  --gas 1
  --run-name "$RUN_NAME"
  --out "$OUT"
  --runtime-dir "$OUT/launch_runtime"
  --al-config "$SOURCE_RUN/al_config.json"
  --learning-tasks place_object_stand
  --ee-fk-loss
  --ee-fk-eval
  --ee-position-weight 3.0
  --ee-orientation-weight 0.8
  --ee-gripper-weight 1.5
  --no-freeze-vision-encoder
)

# The training environment path is explicit on the server; the fallback keeps
# this file usable when copied into an activated environment.
if [[ ! -x "${CMD[0]}" ]]; then
  CMD[0]="${PYTHON:-python}"
fi
if [[ "${SHUTDOWN:-0}" == "1" ]]; then
  CMD+=(--shutdown)
fi

printf 'Strategy: full vision + EE weights position=3.0 orientation=0.8 gripper=1.5\n'
printf 'Source HF: %s\nOutput: %s\n' "$FROM_HF" "$OUT"
printf 'Command:\n  '
printf '%q ' "${CMD[@]}"
printf '\n'

if [[ "${EXECUTE:-0}" == "1" ]]; then
  exec "${CMD[@]}" --execute
fi
