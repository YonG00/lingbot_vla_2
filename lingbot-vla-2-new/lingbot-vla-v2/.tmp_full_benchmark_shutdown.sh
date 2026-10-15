#!/usr/bin/env bash

cd /data/code/lingbot-vla-v2 || exit 2
python experiment/robotwin/run_clean_benchmark.py \
  --model-path /data/outputs/al_v79_poolfix_7300/checkpoints/global_step_9680/hf_ckpt \
  --robotwin-root /data/code/RoboTwin-lingbot \
  --qwen3vl-path /data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct \
  --conda-sh /data/miniconda3/etc/profile.d/conda.sh \
  --inference-env lingbotvla \
  --sim-env RoboTwin \
  --task-config both \
  --num-tasks 50 \
  --episodes 10 \
  --precision bf16 \
  --run-name step9680_bf16_unseen_full50_both_10ep_20261014 \
  --runtime-dir /data/eval_results/clean_benchmark; /usr/bin/shutdown
