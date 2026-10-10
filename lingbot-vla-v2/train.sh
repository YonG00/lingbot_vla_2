#!/bin/bash

set -x

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1 
export HF_HUB_DISABLE_TELEMETRY=1 
export DISABLE_TELEMETRY=1 

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  # 🔴 不能用 `nvidia-smi -L | wc -l` 兜底：AMD/ROCm 上没有 nvidia-smi，
  #    报错信息会被 wc -l 数成 1 ⇒ 只起 1 个进程、FSDP 网格崩（2026-10-10 实测踩过）。
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
  else
    NPROC_PER_NODE=$("${PY:-python3}" -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 1)
  fi
else
  NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Using NPROC_PER_NODE=$NPROC_PER_NODE GPUs"
NNODES=${NNODES:=1}
NPROC_PER_NODE=${NPROC_PER_NODE:=$NPROC_PER_NODE}
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=62500}


# 🔴 torchrun 不一定在 PATH 上（例如被 `bash train.sh` 直接调用、或没先 conda activate）
#    ⇒ 依次尝试：PATH 上的 torchrun → 与 $PY 同目录的 torchrun → `$PY -m torch.distributed.run`
#    （三者等价；缺了会静默 `command not found` + rc=127，2026-10-07 实测踩过）
if command -v torchrun >/dev/null 2>&1; then
  TORCHRUN=(torchrun)
elif [ -n "${PY:-}" ] && [ -x "${PY%/*}/torchrun" ]; then
  TORCHRUN=("${PY%/*}/torchrun")
else
  TORCHRUN=("${PY:-python}" -m torch.distributed.run)
fi

"${TORCHRUN[@]}" --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT $@ 2>&1 | tee log.txt

# 🔴 用了管道 ⇒ `$?` 是 `tee` 的（永远是 0）。必须取 PIPESTATUS[0]（torchrun 的），
#    否则训练崩溃/退出会被上报成成功，调用方与自动化全部误判（2026-10-05 实测踩过）。
TORCHRUN_RC=${PIPESTATUS[0]}
exit "$TORCHRUN_RC"
