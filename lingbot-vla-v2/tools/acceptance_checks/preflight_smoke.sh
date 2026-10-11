#!/usr/bin/env bash
# 96G 单卡 Smoke 预检：GPU 身份 / CUDA 环境 / /data 资产完整性 / 配置与步数
set -uo pipefail
PY=/data/miniconda3/envs/lingbotvla/bin/python
echo "=== ① 时间/主机 ==="; date "+%F %T"; hostname
echo "=== ② GPU 身份与驱动 ==="
nvidia-smi -i 0 --query-gpu=name,memory.total,memory.used,memory.free,driver_version,compute_cap --format=csv,noheader || echo "  ❌ nvidia-smi 失败（无卡？）"
nvidia-smi -L 2>/dev/null | head -4
echo "=== ③ CUDA / torch ==="
$PY -c "
import torch
print(f'  torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} 卡数={torch.cuda.device_count()}')
if torch.cuda.is_available():
    p=torch.cuda.get_device_properties(0)
    print(f'  {p.name} | total {p.total_memory/2**30:.2f} GiB | cc {p.major}.{p.minor}')" 2>&1 | tail -3
echo "=== ④ /data 资产完整性（新实例的话这里会缺）==="
for p in /data/code/lingbot-vla-v2/train.sh \
         /data/code/lingbot-vla-v2/configs/auto_learning/smoke_gbs4_4task_tb5.yaml \
         /data/train/task_splits_50/manifest.json \
         /data/train/task_splits_50/task_baseline.json \
         /data/train/task_splits_50/combined.train_ids.json \
         /data/eval_results/open_loop/ref50k/pass_thresholds_gmean200_warn.json \
         /data/eval_results/open_loop/ref50k/ref_per_traj.jsonl \
         /data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt/config.json \
         /data/train/phases/datasets.txt ; do
  if [ -e "$p" ]; then echo "  ✅ $p"; else echo "  ❌ 缺 $p"; fi
done
echo "  步500 权重体积: $(du -sh /data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt 2>/dev/null | cut -f1 || echo '-')"
echo "  磁盘: $(df -h /data | tail -1)"
echo "=== ⑤ 关键哈希（与记录对拍）==="
md5sum /data/code/lingbot-vla-v2/configs/auto_learning/smoke_gbs4_4task_tb5.yaml \
       /data/eval_results/open_loop/ref50k/pass_thresholds_gmean200_warn.json \
       /data/train/task_splits_50/task_baseline.json \
       /data/eval_results/open_loop/ref50k/ref_per_traj.jsonl 2>/dev/null
echo "  期望: YAML cf7aa366f107720c2eeb33fc83670c07 | 200× 表 29ea9985c7a36f2ce7dcaa8d84f8b4df"
echo "         baseline fc60f0e8b7cb07b2a411a5d1901f4341 | ref f347eea32abade7a69d6d7ca70dca6e4"
echo "=== ⑥ git 状态（远端 /data/code）==="
cd /data/code 2>/dev/null && { echo "  HEAD=$(git rev-parse --short HEAD)"; git status --short | head -6; echo "  status 行数=$(git status --short | wc -l)"; }
echo "=== ⑦ 动态看门狗阈值（按实际总显存算）==="
T=$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ' | head -1)
echo "  total=${T} MiB ⇒ 阈值 $(( ${T:-0} * 90 / 100 )) MiB (90%)"
