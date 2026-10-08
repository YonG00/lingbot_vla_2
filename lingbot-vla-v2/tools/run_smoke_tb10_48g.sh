#!/usr/bin/env bash
# =============================================================================
# 10-step Auto Learning Smoke —— **单卡（按实际卡型）**版：本机实测为 RTX 4090 49140 MiB
#   BF16 · MICRO=1 · GAS=4 · GBS=4(3 NEW + 1 Replay) · 4 任务 · GMean 200×/warn
#   STEP_OFFSET=500 + MAX_STEPS=510 ⇒ 再训 10 步 · Learning Unit = 5 步（unit 边界 @505/@510）
#   ⚠️ 步时口径：历史 5.99 s/step 是 **gas=10** 的实测；本轮 gas=4 ⇒ **不可**直接比较。
#   自带：GPU 环境记录 / 动态显存看门狗(90% of memory.total, MiB) / TB 自启 / 阶段与显存汇总 / 成功才关机
# =============================================================================
set -uo pipefail
PY=/data/miniconda3/envs/lingbotvla/bin/python
cd /data/code/lingbot-vla-v2
OUT=/data/outputs/al_smoke_gbs4_4task_tb10
SUMMARY=/data/tmp/smoke_tb10_summary.txt
REPOLOG=/data/code/lingbot-vla-v2/log.txt
ALCFG=/data/code/lingbot-vla-v2/configs/auto_learning/smoke_gbs4_4task_tb5.yaml
OFFSET=500; TARGET=510; STEPS=$(( TARGET - OFFSET ))
GPU_INDEX=0

echo "=== 前置校验 $(date '+%F %T') ==="
# ---- GPU 环境（型号/显存/驱动/CUDA）----
GPU_NAME=$(nvidia-smi -i $GPU_INDEX --query-gpu=name --format=csv,noheader | head -1)
TOTAL_MIB=$(nvidia-smi -i $GPU_INDEX --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ' | head -1)
DRIVER=$(nvidia-smi -i $GPU_INDEX --query-gpu=driver_version --format=csv,noheader | head -1)
echo "  GPU: ${GPU_NAME}  显存: ${TOTAL_MIB} MiB ($(awk -v m="$TOTAL_MIB" 'BEGIN{printf "%.2f", m/1024}') GiB)  驱动: ${DRIVER}"
[ "${TOTAL_MIB:-0}" -gt 0 ] 2>/dev/null || { echo "  ❌ 读不到显存总量（无卡/无 nvidia-smi）⇒ 中止"; exit 5; }
$PY -c "
import torch
print(f'  torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} 卡数={torch.cuda.device_count()}')
if torch.cuda.is_available():
    p=torch.cuda.get_device_properties($GPU_INDEX)
    print(f'  device={p.name} total={p.total_memory/2**30:.2f} GiB cc={p.major}.{p.minor}')" 2>&1 | tail -3
# ---- 动态看门狗阈值：总显存 90%（单位统一为 MiB，与采样口径一致）----
VRAM_LIMIT_MIB=$(( TOTAL_MIB * 90 / 100 ))
echo "  显存看门狗阈值: ${VRAM_LIMIT_MIB} MiB = 总显存 90%（采样与阈值同为 MiB）"
# ---- 步数 / 依赖 / YAML ----
echo "  STEP_OFFSET=$OFFSET MAX_STEPS=$TARGET ⇒ 实际训练 $STEPS 步（期望 10）"
[ "$STEPS" -eq 10 ] || { echo "  ❌ 期望 10 步，已中止"; exit 2; }
for f in "$ALCFG" /data/eval_results/open_loop/ref50k/pass_thresholds_gmean200_warn.json \
         /data/train/task_splits_50/task_baseline.json /data/train/task_splits_50/manifest.json \
         /data/train/task_splits_50/combined.train_ids.json; do
  [ -f "$f" ] && echo "  ✅ $f" || { echo "  ❌ 缺文件 $f"; exit 3; }
done
$PY - <<'PY' || exit 4
import yaml
from lingbotvla.auto_learning.config import AutoLearningConfig
c=AutoLearningConfig.from_dict(yaml.safe_load(open("/data/code/lingbot-vla-v2/configs/auto_learning/smoke_gbs4_4task_tb5.yaml",encoding="utf-8")))
assert c.eval_interval_steps==5 and c.min_steps_before_defer==5 and c.defer_retry_steps==5, "unit 必须是 5 步"
assert c.batch_size==4 and c.new_slots==3 and c.replay_slots==1, "AL batch 必须 4(3+1)"
assert c.pass_metric=="mse" and c.task_names and len(c.task_names)==4, "4 任务 + GMean-MSE"
print(f"  ✅ YAML: unit={c.eval_interval_steps} 步 | batch={c.batch_size}({c.new_slots} NEW+{c.replay_slots} Replay) | metric={c.pass_metric} | 任务={c.task_names} | rescan_every={c.rescan_every_n_task_switches}")
PY
mkdir -p "$OUT"

# ---- TensorBoard（仅服务；events 由训练器无条件写）----
setsid nohup $PY -m tensorboard.main --logdir "$OUT/runs" --port 6006 --host 0.0.0.0 > "$OUT/tb.log" 2>&1 < /dev/null &
ln -sfn "$OUT/runs" /root/tf-logs/al_smoke_gbs4_tb10
echo "  TB: http://0.0.0.0:6006 (symlink /root/tf-logs/al_smoke_gbs4_tb10)"

# ---- 显存采样（MiB, GPU0）+ 动态阈值看门狗 ----
: > "$OUT/vram.csv"; : > "$OUT/watchdog.log"
( over=0
  while true; do
    line=$(nvidia-smi -i $GPU_INDEX --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    echo "$line" >> "$OUT/vram.csv"
    m=$(echo "$line" | awk -F',' '{gsub(/[^0-9]/,"",$2); print $2+0}')
    if [ "${m:-0}" -gt "$VRAM_LIMIT_MIB" ]; then
      over=$((over+1)); echo "$(date '+%T') 超限 ${m}MiB (> ${VRAM_LIMIT_MIB}MiB = 90%×${TOTAL_MIB}) 连续 $over 次" >> "$OUT/watchdog.log"
      if [ "$over" -ge 2 ]; then
        echo "$(date '+%T') ⇒ 看门狗触发：杀掉训练进程（避免硬 OOM）" >> "$OUT/watchdog.log"
        pkill -f "[t]rain_lingbotvla"; echo "OVERRUN_KILL limit=${VRAM_LIMIT_MIB}MiB" > "$OUT/watchdog_killed"; break
      fi
    else over=0; fi
    sleep 3
  done ) &
WATCH=$!

START=$(date +%s)
echo "=== 训练启动 $(date '+%F %T') ==="
MICRO=1 GAS=4 N_GPU=1 MIXED=false \
MAX_STEPS=$TARGET SAVE_EVERY=0 DCP_MODE=final_only PRUNE=0 TB=0 \
MODEL_PATH=/data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt STEP_OFFSET=$OFFSET \
TRAIN_OUT="$OUT" AL_CFG="$ALCFG" SPLIT_DIR=/data/train/task_splits_50 \
  bash experiment/robotwin/al_50task_bf16.sh
RC=$?
END=$(date +%s); DUR=$((END-START)); kill $WATCH 2>/dev/null
cp -f "$REPOLOG" "$OUT/log.txt" 2>/dev/null
LOG="$OUT/log.txt"; [ -s "$LOG" ] || LOG="$REPOLOG"
{
echo "=== SMOKE 结束 $(date '+%F %T') rc=$RC 总耗时 ${DUR}s = $((DUR/60))分$((DUR%60))秒 ==="
echo "  硬件: ${GPU_NAME} | 总显存 ${TOTAL_MIB} MiB | 驱动 ${DRIVER} | 看门狗阈值 ${VRAM_LIMIT_MIB} MiB"
echo "  ⚠️ 步时口径：本轮 gas=4；历史 5.99 s/step 为 gas=10 ⇒ 不作严格性能对比"
[ -f "$OUT/watchdog_killed" ] && echo "  ⚠️ 看门狗曾触发（>${VRAM_LIMIT_MIB}MiB）：$(cat $OUT/watchdog_killed)"
echo "--- ① 阶段耗时（日志时间戳）---"
$PY - "$LOG" <<'PY'
import re,sys,datetime
rows=[]
for l in open(sys.argv[1],encoding="utf-8",errors="ignore"):
    m=re.match(r"(\d\d/\d\d/\d{4} \d\d:\d\d:\d\d)",l)
    if m: rows.append((datetime.datetime.strptime(m.group(1),"%m/%d/%Y %H:%M:%S"),l.rstrip()))
if not rows: print("  (无时间戳行)"); raise SystemExit
t0=rows[0][0]
def mark(pat,label,limit=3):
    for t,l in [(t,l) for t,l in rows if re.search(pat,l)][:limit]:
        print(f"  [{t.strftime('%H:%M:%S')} +{(t-t0).total_seconds():7.1f}s] {label}: {l[:150]}")
mark(r"Initializing datasets: 100%","数据集初始化")
mark(r"eval .*n_ids=2","Bootstrap scout 评测(2 轨迹)")
mark(r"第一个 learning unit 的 TrainRequest","首个 unit 发布")
mark(r"Step 50\d/","训练 step")
mark(r"eval .*n_ids=4","单元评测(4 轨迹)")
mark(r"unit 没跑满|收尾","单元收尾")
mark(r"Saved checkpoint|Distributed checkpoint saved","存档")
mark(r"VRAM usage","框架显存埋点(max_memory_allocated)")
PY
echo "--- ② 训练 step 明细（Loss/GradNorm/StepTime）---"
grep -a "StepTime" "$LOG" | tail -12
echo "--- ③ 框架显存埋点（PyTorch peak allocated）---"
grep -a "VRAM usage" "$LOG"
echo "--- ④ Hardness 计时（事件 + TB 标签）---"
$PY - <<'PY'
import json,os
ev="/data/outputs/al_smoke_gbs4_4task_tb10/auto_learning_events.jsonl"
if not os.path.exists(ev): print("  (无 events)"); raise SystemExit
rows=[json.loads(l) for l in open(ev,encoding="utf-8") if l.strip()]
soft=[r for r in rows if r.get("hardness_scan_seconds") is not None]
print("  hardness: (task, seconds, is_first, samples) =")
for r in soft:
    print("   ",r.get("task"),r.get("hardness_scan_seconds"),r.get("hardness_scan_is_first"),r.get("n_scanned") or r.get("n_samples"))
PY
echo "--- ⑤ 决策/事件摘要 ---"
$PY - <<'PY'
import json,collections,os
ev="/data/outputs/al_smoke_gbs4_4task_tb10/auto_learning_events.jsonl"
if not os.path.exists(ev): print("  (无 events)"); raise SystemExit
rows=[json.loads(l) for l in open(ev,encoding="utf-8") if l.strip()]
c=collections.Counter(r.get("action") or r.get("kind") for r in rows)
print("  action 计数:",dict(c))
for r in rows:
    if r.get("action") in ("bootstrap","select","train_unit","decision","rescan","review","finish","round_rollover"):
        print("   ",json.dumps({k:r.get(k) for k in ("action","task","result","steps","pass_source","val_nmse","val_mse","reason","stop_reason","deferred","hardness_scan_seconds") if k in r},ensure_ascii=False))
PY
echo "--- ⑥ 显存峰值（nvidia-smi 采样, MiB）---"
$PY - <<'PY'
import datetime
A="/data/outputs/al_smoke_gbs4_4task_tb10/vram.csv"
rows=[]
for line in open(A,encoding="utf-8",errors="ignore"):
    p=[x.strip() for x in line.split(",")]
    if len(p)<3 or not p[0][:2].isdigit(): continue
    try: rows.append((datetime.datetime.strptime(p[0],"%Y/%m/%d %H:%M:%S.%f"),int(p[1].split()[0]),int(p[2].split()[0])))
    except Exception: pass
if rows:
    pk=max(rows,key=lambda r:r[1])
    print(f"  采样 {len(rows)} 条 | 峰值 {pk[1]} MiB = {pk[1]/1024:.2f} GiB @ {pk[0].strftime('%H:%M:%S')} (util {pk[2]}%)")
PY
echo "--- ⑦ TensorBoard 标量（含 505/510 与 max_memory_*）---"
$PY - <<'PY'
import glob,os
try:
    from tensorboard.backend.event_processing import event_accumulator as ea
except Exception as e:
    print("  (tensorboard event_accumulator 不可用:",e,")"); raise SystemExit
fs=sorted(glob.glob("/data/outputs/al_smoke_gbs4_4task_tb10/runs/events.out.tfevents.*"))
if not fs: print("  (无 events 文件)"); raise SystemExit
acc=ea.EventAccumulator(fs[-1]); acc.Reload()
tags=sorted(acc.Tags().get("scalars",[]))
print(f"  events={os.path.basename(fs[-1])} 标量标签数={len(tags)}")
key=[t for t in tags if t in ("training/loss","auto_learning/unit_loss","current_skill/val_mse","current_skill/pass_threshold_mse",
     "current_skill/val_to_pass_threshold","auto_learning/hardness_scan_seconds","max_memory_allocated(GB)","max_memory_reserved(GB)","steptime")
     or t.startswith("task/") and t.endswith(("val_to_pass_threshold","pass_threshold_mse")) or t.startswith("detailed_loss/")]
for t in key[:40]:
    pts=[(s.step,round(s.value,6)) for s in acc.Scalars(t)]
    sel=[p for p in pts if p[0] in (500,505,510)]
    print(f"  {t:<48} n={len(pts):<4} 505/510: {sel if sel else pts[-3:]}")
PY
echo "--- ⑧ 存档 ---"
CKPT=$(ls -d "$OUT"/checkpoints/global_step_* 2>/dev/null | tail -1)
echo "  ckpt_dir=${CKPT:-<无>}"
[ -n "$CKPT" ] && { du -sh "$CKPT"/* 2>/dev/null | head -4; echo "  hf_ckpt 分片: $(ls "$CKPT"/hf_ckpt/*.safetensors 2>/dev/null | wc -l)  index.json: $([ -f "$CKPT"/hf_ckpt/index.json ] && echo 有 || echo 无)"; }
echo "  events.jsonl 行数: $(wc -l < "$OUT/auto_learning_events.jsonl" 2>/dev/null || echo 0) | runs/: $(ls "$OUT/runs" 2>/dev/null | wc -l) 文件"
} 2>&1 | tee "$SUMMARY"
sync
if [ "$RC" -eq 0 ] && [ -n "$CKPT" ] && [ ! -f "$OUT/watchdog_killed" ]; then
  echo "=== ✅ rc=0 且存档落地且看门狗未触发 ⇒ 8 秒后自动关机 ==="; sleep 8; echo "SHUTDOWN_NOW $(date '+%F %T')"; sync; /usr/bin/shutdown
else
  echo "=== ❌ rc=$RC / 存档缺失 / 看门狗触发 ⇒ 不关机，等诊断 ==="
fi
echo "=== JOB_END $(date '+%F %T') ==="
