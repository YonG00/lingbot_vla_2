#!/usr/bin/env bash
# Continue the morning full-vision step500 checkpoint for 750 steps.
# Default is preview-only. The launcher prepares a fresh CPU cache before torchrun.
set -euo pipefail

ROOT="${ROOT:-/data/experiments/learn_skill_mix_task/lingbot-vla-v2}"
SOURCE_RUN="${SOURCE_RUN:-/data/outputs/al_stand_ee_fullvision_w3_o08_g15_from500_add500_20261018_092225}"
FROM_HF="${FROM_HF:-$SOURCE_RUN/checkpoints/global_step_500/hf_ckpt}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-al_stand_ee_contiguous_grasp_fromfull500_add750_${RUN_STAMP}}"
OUT="${OUT:-/data/outputs/$RUN_NAME}"
RUNTIME_DIR="${RUNTIME_DIR:-/data/outputs/skill_launches}"
WINDOW_CACHE_DIR="${WINDOW_CACHE_DIR:-/data/al_cache/${RUN_NAME}}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-source}"
PYTHON_BIN="${PYTHON:-/data/miniconda3/envs/lingbotvla/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python}"
fi

CMD=(
  "$PYTHON_BIN" "$ROOT/experiment/robotwin/continue_skill_training.py"
  --from-hf "$FROM_HF"
  --steps 750
  --micro 20
  --gas 1
  --run-name "$RUN_NAME"
  --out "$OUT"
  --runtime-dir "$RUNTIME_DIR"
  --al-config "$SOURCE_RUN/al_config.json"
  --learning-tasks place_object_stand
  --sample-scan
  --sample-scan-method target_change_window
  --low-noise-fm
  --al-set "target_change_cache_dir=\"$WINDOW_CACHE_DIR\""
  --al-set target_change_grasp_lift_max_delay_seconds=2.0
  --ee-fk-loss
  --ee-fk-eval
  --ee-position-weight 3.0
  --ee-orientation-weight 0.8
  --ee-gripper-weight 1.5
  --no-freeze-vision-encoder
)
case "$GRADIENT_CHECKPOINTING" in
  source) ;;
  1|true|TRUE) CMD+=(--gradient-checkpointing) ;;
  0|false|FALSE) CMD+=(--no-gradient-checkpointing) ;;
  *) echo "GRADIENT_CHECKPOINTING must be source, 0 or 1" >&2; exit 2 ;;
esac
if [[ "${SHUTDOWN:-0}" == "1" ]]; then
  CMD+=(--shutdown)
else
  CMD+=(--no-shutdown)
fi

printf 'Strategy: full vision, contiguous grasp windows, +750 steps, EE weights=3.0/0.8/1.5\n'
printf 'Source HF: %s\nWindow cache: %s\nOutput: %s\n' "$FROM_HF" "$WINDOW_CACHE_DIR" "$OUT"
printf 'Gradient checkpointing: %s\nTraining command:\n  ' "$GRADIENT_CHECKPOINTING"
printf '%q ' "${CMD[@]}"
printf '\n'

if [[ "${EXECUTE:-0}" == "1" ]]; then
  exec "${CMD[@]}" --execute
fi
