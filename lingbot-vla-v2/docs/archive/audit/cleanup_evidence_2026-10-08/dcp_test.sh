#!/usr/bin/env bash
# 存档专项验收：15 步 → DCP@505/510/515，keep_last=2 ⇒ 第三次保存后应删掉 505
set -u
cd /data/code/lingbot-vla-v2
PY=/data/miniconda3/envs/lingbotvla/bin/python
CFG=configs/auto_learning/dcp_retention_test.yaml
cp configs/auto_learning/smoke_gbs4_4task_tb5.yaml "$CFG"
"$PY" - "$CFG" <<'PYEOF'
import re, sys, pathlib, yaml
p = pathlib.Path(sys.argv[1]); t = p.read_text(encoding="utf-8")
t = re.sub(r"^hardness_probe_fraction:.*$", "hardness_probe_fraction: 0.01   # 测试用：压缩扫描耗时", t, flags=re.M)
t = re.sub(r"^target_total_passed_tasks:.*$", "target_total_passed_tasks: null # 测试用：不早停", t, flags=re.M)
t = re.sub(r"^max_global_steps:.*$", "max_global_steps: 200", t, flags=re.M)
p.write_text(t, encoding="utf-8")
c = yaml.safe_load(t)
print("  ✅ 测试 YAML:", "unit=%s" % c["eval_interval_steps"], "hardness=%s" % c["hardness_probe_fraction"],
      "target=%s" % c["target_total_passed_tasks"], "batch=%s(%s+%s)" % (c["batch_size"], c["new_slots"], c["replay_slots"]))
PYEOF
OUT=/data/outputs/al_dcp_retention_test_$(date +%m%d_%H%M)
echo "$OUT" > /data/tmp/dcp_test_outdir.txt
setsid nohup env TORCH_COMPILE_DISABLE=1 MICRO=1 GAS=4 N_GPU=1 MIXED=false \
  SMOKE_NO_CHECKPOINT=0 SAVE_EVERY=5 DCP_MODE=always DCP_KEEP_LAST=2 PRUNE=0 HF_PASS_INTERVAL=0 TB=1 TB_PORT=6006 \
  MAX_STEPS=515 STEP_OFFSET=500 \
  MODEL_PATH=/data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt \
  TRAIN_OUT="$OUT" AL_CFG="$PWD/$CFG" SPLIT_DIR=/data/train/task_splits_50 \
  bash experiment/robotwin/al_50task_bf16.sh > /data/tmp/dcp_test_job.log 2>&1 < /dev/null &
sleep 20
echo "  OUT=$OUT"
echo "  --- 启动器确认的关键参数 ---"
grep -E "save_steps |dcp_save_mode|dcp_keep_last|hf_pass_interval|save_hf_weights|DCP 保留|存档计划" /data/tmp/dcp_test_job.log | sed 's/^/    /' | head -8
echo "  --- 进程/GPU ---"
echo "    train 进程: $(pgrep -c -f '[t]rain_lingbotvla' || echo 0)  GPU: $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)"
