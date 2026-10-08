#!/usr/bin/env bash
# =============================================================================
# 10-step Auto Learning Smoke —— **无存档（SMOKE_NO_CHECKPOINT=1）** 单卡版
#
#   BF16 · MICRO=1 · GAS=4 · GBS=4(3 NEW + 1 Replay) · 4 任务 · GMean 200×/warn
#   STEP_OFFSET=500 + MAX_STEPS=510 ⇒ 再训 10 步 · Learning Unit = 5 步（@505/@510）
#
#   v3 相对 v2 的改动（配合 totalpass / no-save / TB 补丁）：
#     1) SMOKE_NO_CHECKPOINT 默认 1：本轮**不产 DCP/HF**，成功条件不再要求存档；
#        反而**检查"没有存档"**作为模式生效的正向证据（模式泄漏 ⇒ 判失败）。
#     2) 成功条件 = rc=0 + 看门狗未触发 + 关键日志/TB 已落盘 +
#        （跑满预期绝对步数 **或** AL 正常收工：target_total_passed_reached 等）。
#        ⇒ **提前达到 PASS 目标不再被误判为异常退出**。
#     3) 关机前先写汇总 + RUN_DONE 标记 + sync，绝不在日志落盘前关机；
#        不自动延长训练步数。
#     4) 路径全部可用 env 覆盖（REPO/OUT/SPLIT_DIR/ALCFG/SUMMARY），便于在临时副本上验证。
#
#   自带：GPU 环境记录 / 动态显存看门狗(99% of memory.total, MiB) / TB 自启 /
#         阶段与显存汇总 / 判定 / 成功才关机
# =============================================================================
set -uo pipefail
# 【用户决策 b】正式/ Smoke 均关闭 torch.compile：config 里 train.use_compile=true
# （robotwin_official_paths.yaml:90），用官方 env 全局关掉 dynamo（实测 dynamo.config.disable=True）。
export TORCH_COMPILE_DISABLE=1

PY=${PY:-/data/miniconda3/envs/lingbotvla/bin/python}
REPO=${REPO:-/data/code/lingbot-vla-v2}
OUT=${OUT:-/data/outputs/al_smoke_gbs4_4task_nosave}
SUMMARY=${SUMMARY:-/data/tmp/smoke_nosave_summary.txt}
REPOLOG=${REPOLOG:-$REPO/log.txt}
ALCFG=${ALCFG:-$REPO/configs/auto_learning/smoke_gbs4_4task_tb5.yaml}
SPLIT_DIR=${SPLIT_DIR:-/data/train/task_splits_50}
THRESH=${THRESH:-/data/eval_results/open_loop/ref50k/pass_thresholds_gmean200_warn.json}
SMOKE_NO_CHECKPOINT=${SMOKE_NO_CHECKPOINT:-1}
RUN_MODEL=${RUN_MODEL:-/data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt}
TRAIN_ID=${TRAIN_ID:-al_smoke_gbs4_nosave}
OFFSET=${OFFSET:-500}; TARGET=${TARGET:-510}; STEPS=$(( TARGET - OFFSET ))
GPU_INDEX=${GPU_INDEX:-0}
cd "$REPO"

# DRY_RUN=1：**完全不启动训练、不碰 GPU**。跳过 nvidia-smi/torch 探测（需 SKIP_GPU_CHECK=1），
# 只做依赖/YAML/命令拼装检查，并用 OUT 里**已有产物**演练 §⑦/§⑧ 与判定逻辑。
DRY_RUN=${DRY_RUN:-0}
SKIP_GPU_CHECK=${SKIP_GPU_CHECK:-$DRY_RUN}
echo "=== 前置校验 $(date '+%F %T') ==="
echo "  模式: SMOKE_NO_CHECKPOINT=$SMOKE_NO_CHECKPOINT（1=不产 DCP/HF，不可 Resume）| DRY_RUN=$DRY_RUN"
if [ "$SMOKE_NO_CHECKPOINT" = "1" ] && [ "${RESUME:-0}" = "1" ]; then
  echo "  ❌ SMOKE_NO_CHECKPOINT=1 不支持 RESUME（启动器也会拒绝）⇒ 中止"; exit 2
fi
if [ "$SKIP_GPU_CHECK" = "1" ]; then
  GPU_NAME=${GPU_NAME:-"<DRY_RUN 未探测>"}; TOTAL_MIB=${TOTAL_MIB:-49140}; DRIVER=${DRIVER:-"<DRY_RUN>"}
  echo "  (DRY_RUN) 跳过 GPU/torch 探测；按 TOTAL_MIB=$TOTAL_MIB 计算阈值与命令"
else
  GPU_NAME=$(nvidia-smi -i $GPU_INDEX --query-gpu=name --format=csv,noheader | head -1)
  TOTAL_MIB=$(nvidia-smi -i $GPU_INDEX --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ' | head -1)
  DRIVER=$(nvidia-smi -i $GPU_INDEX --query-gpu=driver_version --format=csv,noheader | head -1)
fi
echo "  GPU: ${GPU_NAME}  显存: ${TOTAL_MIB} MiB ($(awk -v m="$TOTAL_MIB" 'BEGIN{printf "%.2f", m/1024}') GiB)  驱动: ${DRIVER}"
[ "${TOTAL_MIB:-0}" -gt 0 ] 2>/dev/null || { echo "  ❌ 读不到显存总量（无卡/无 nvidia-smi）⇒ 中止"; exit 5; }
if [ "$SKIP_GPU_CHECK" != "1" ]; then
  $PY -c "
import torch
print(f'  torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} 卡数={torch.cuda.device_count()}')
if torch.cuda.is_available():
    p=torch.cuda.get_device_properties($GPU_INDEX)
    print(f'  device={p.name} total={p.total_memory/2**30:.2f} GiB cc={p.major}.{p.minor}')" 2>&1 | tail -3
fi
VRAM_LIMIT_MIB=$(( TOTAL_MIB * 99 / 100 ))
echo "  显存看门狗阈值: ${VRAM_LIMIT_MIB} MiB = 总显存 99%（采样与阈值同为 MiB）"
echo "  torch.compile: 已禁用（TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-未设}）⇒ 无 dynamo 编译尖峰"
echo "  STEP_OFFSET=$OFFSET MAX_STEPS=$TARGET ⇒ 实际训练 $STEPS 步（期望 $STEPS）"
[ "$STEPS" -ge 1 ] || { echo "  ❌ STEPS=$STEPS 非法（MAX_STEPS 必须 > STEP_OFFSET）⇒ 中止"; exit 2; }
for f in "$ALCFG" "$THRESH" "$SPLIT_DIR/task_baseline.json" "$SPLIT_DIR/manifest.json" "$SPLIT_DIR/combined.train_ids.json"; do
  [ -f "$f" ] && echo "  ✅ $f" || { echo "  ❌ 缺文件 $f"; exit 3; }
done
$PY - "$ALCFG" <<'PY' || exit 4
import sys, yaml
from lingbotvla.auto_learning.config import AutoLearningConfig
c=AutoLearningConfig.from_dict(yaml.safe_load(open(sys.argv[1],encoding="utf-8")))
assert c.eval_interval_steps==5 and c.min_steps_before_defer==5 and c.defer_retry_steps==5, "unit 必须是 5 步"
assert c.batch_size==4 and c.new_slots==3 and c.replay_slots==1, "AL batch 必须 4(3+1)"
assert c.pass_metric=="mse" and c.task_names and len(c.task_names)==4, "4 任务 + GMean-MSE"
print(f"  ✅ YAML: unit={c.eval_interval_steps} 步 | batch={c.batch_size}({c.new_slots} NEW+{c.replay_slots} Replay) | "
      f"metric={c.pass_metric} | 任务={c.task_names} | total_pass_target={c.target_total_passed_tasks} | "
      f"hardness_fraction={c.hardness_probe_fraction} | rescan_every={c.rescan_every_n_task_switches}")
PY
mkdir -p "$OUT"

if [ "$DRY_RUN" = "1" ]; then
  echo "=== [DRY_RUN] 将要执行的启动命令（不执行训练）==="
  echo "  cd $REPO && MICRO=1 GAS=4 N_GPU=1 MIXED=false SMOKE_NO_CHECKPOINT=$SMOKE_NO_CHECKPOINT \\"
  echo "    MAX_STEPS=$TARGET SAVE_EVERY=0 DCP_MODE=final_only PRUNE=0 TB=0 \\"
  echo "    MODEL_PATH=$RUN_MODEL STEP_OFFSET=$OFFSET TRAIN_OUT=$OUT AL_CFG=$ALCFG \\"
  echo "    SPLIT_DIR=$SPLIT_DIR bash experiment/robotwin/al_50task_bf16.sh"
  echo "  --- 启动器 DRY_RUN 预演（打印真实参数，不训练）---"
  DRY_RUN=1 MICRO=1 GAS=4 N_GPU=1 MIXED=false SMOKE_NO_CHECKPOINT="$SMOKE_NO_CHECKPOINT" \
  MAX_STEPS=$TARGET SAVE_EVERY=0 DCP_MODE=final_only PRUNE=0 TB=0 \
  MODEL_PATH="$RUN_MODEL" STEP_OFFSET=$OFFSET TRAIN_OUT="$OUT" AL_CFG="$ALCFG" SPLIT_DIR="$SPLIT_DIR" \
    bash experiment/robotwin/al_50task_bf16.sh 2>&1 | grep -E "smoke_no_checkpoint|save_steps|save_epochs|save_hf_weights|disk_guard|NO_CHECKPOINT|PRUNE=0|收工条件|存档计划|精度|批大小|STEP_OFFSET|max_steps" | sed 's/^/    /'
  RC=0; DUR=0
  cp -f "$REPOLOG" "$OUT/log.txt" 2>/dev/null
  LOG="$OUT/log.txt"; [ -s "$LOG" ] || LOG="$REPOLOG"
  STEPS_DONE=$(grep -ac "StepTime" "$LOG" 2>/dev/null || echo 0)
  EVJSON="$OUT/auto_learning_events.jsonl"
  TB_EV=$(ls "$OUT"/runs/events.out.tfevents.* 2>/dev/null | head -1)
  CKPT_N=$(ls -d "$OUT"/checkpoints/global_step_* 2>/dev/null | wc -l | tr -d ' ')
  echo "  (DRY_RUN) 用已有产物演练判定：log=$(wc -c < "$LOG" 2>/dev/null || echo 0)B steps=$STEPS_DONE events=$(wc -l < "$EVJSON" 2>/dev/null || echo 0)行 ckpt=$CKPT_N"
else

# ---- TensorBoard（仅服务；events 由训练器无条件写）----
setsid nohup $PY -m tensorboard.main --logdir "$OUT/runs" --port 6006 --host 0.0.0.0 > "$OUT/tb.log" 2>&1 < /dev/null &
ln -sfn "$OUT/runs" /root/tf-logs/$TRAIN_ID
echo "  TB: http://0.0.0.0:6006 (symlink /root/tf-logs/$TRAIN_ID)"

# ---- 显存采样（MiB, GPU0）+ 动态阈值看门狗 ----
: > "$OUT/vram.csv"; : > "$OUT/watchdog.log"; rm -f "$OUT/watchdog_killed" "$OUT/RUN_DONE"
( over=0
  while true; do
    line=$(nvidia-smi -i $GPU_INDEX --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    echo "$line" >> "$OUT/vram.csv"
    m=$(echo "$line" | awk -F',' '{gsub(/[^0-9]/,"",$2); print $2+0}')
    if [ "${m:-0}" -gt "$VRAM_LIMIT_MIB" ]; then
      over=$((over+1)); echo "$(date '+%T') 超限 ${m}MiB (> ${VRAM_LIMIT_MIB}MiB = 99%×${TOTAL_MIB}) 连续 $over 次" >> "$OUT/watchdog.log"
      if [ "$over" -ge 2 ]; then
        echo "$(date '+%T') ⇒ 看门狗触发：杀掉训练进程（避免硬 OOM）" >> "$OUT/watchdog.log"
        pkill -f "[t]rain_lingbotvla"; echo "OVERRUN_KILL limit=${VRAM_LIMIT_MIB}MiB (99%)" > "$OUT/watchdog_killed"; break
      fi
    else over=0; fi
    sleep 3
  done ) &
WATCH=$!

START=$(date +%s)
echo "=== 训练启动 $(date '+%F %T') ==="
MICRO=1 GAS=4 N_GPU=1 MIXED=false \
SMOKE_NO_CHECKPOINT="$SMOKE_NO_CHECKPOINT" \
MAX_STEPS=$TARGET SAVE_EVERY=0 DCP_MODE=final_only PRUNE=0 TB=0 \
MODEL_PATH="$RUN_MODEL" STEP_OFFSET=$OFFSET \
TRAIN_OUT="$OUT" AL_CFG="$ALCFG" SPLIT_DIR="$SPLIT_DIR" \
  bash experiment/robotwin/al_50task_bf16.sh
RC=$?
END=$(date +%s); DUR=$((END-START)); kill $WATCH 2>/dev/null
cp -f "$REPOLOG" "$OUT/log.txt" 2>/dev/null
LOG="$OUT/log.txt"; [ -s "$LOG" ] || LOG="$REPOLOG"
STEPS_DONE=$(grep -ac "StepTime" "$LOG" 2>/dev/null || echo 0)
EVJSON="$OUT/auto_learning_events.jsonl"
TB_EV=$(ls "$OUT"/runs/events.out.tfevents.* 2>/dev/null | head -1)
CKPT_N=$(ls -d "$OUT"/checkpoints/global_step_* 2>/dev/null | wc -l | tr -d ' ')

fi   # ← 结束 DRY_RUN / 真实运行 的分支

{
echo "=== SMOKE 结束 $(date '+%F %T') rc=$RC 总耗时 ${DUR}s = $((DUR/60))分$((DUR%60))秒 ==="
echo "  模式: SMOKE_NO_CHECKPOINT=$SMOKE_NO_CHECKPOINT"
echo "  硬件: ${GPU_NAME} | 总显存 ${TOTAL_MIB} MiB | 驱动 ${DRIVER} | 看门狗阈值 ${VRAM_LIMIT_MIB} MiB"
echo "  步数: 期望 $STEPS 步 / 日志实际 $STEPS_DONE 步"
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
mark(r"unit 没跑满|收尾|NO_CHECKPOINT","单元收尾/无存档提示")
mark(r"Saved checkpoint|Distributed checkpoint saved","存档(无存档模式应为空)")
mark(r"VRAM usage","框架显存埋点(max_memory_allocated)")
PY
echo "--- ② 训练 step 明细（Loss/GradNorm/StepTime）---"
grep -a "StepTime" "$LOG" | tail -12
echo "--- ③ 框架显存埋点（PyTorch peak allocated）---"
grep -a "VRAM usage" "$LOG"
echo "--- ④ Hardness 计时（事件）---"
$PY - "$EVJSON" <<'PY'
import json,os,sys
ev=sys.argv[1]
if not os.path.exists(ev): print("  (无 events)"); raise SystemExit
rows=[json.loads(l) for l in open(ev,encoding="utf-8") if l.strip()]
print("  hardness: (task, seconds, is_first, scanned_samples, total_samples, mean_loss, p90_loss) =")
for r in rows:
    if r.get("hardness_scan_seconds") is not None:
        print("   ",r.get("task"),r.get("hardness_scan_seconds"),r.get("hardness_scan_is_first"),
              r.get("n_scanned"),r.get("n_samples"),r.get("hardness_mean_loss"),r.get("hardness_p90_loss"))
PY
echo "--- ⑤ 决策/事件摘要 ---"
$PY - "$EVJSON" <<'PY'
import json,collections,os,sys
ev=sys.argv[1]
if not os.path.exists(ev): print("  (无 events)"); raise SystemExit
rows=[json.loads(l) for l in open(ev,encoding="utf-8") if l.strip()]
print("  action 计数:",dict(collections.Counter(r.get("action") or r.get("kind") for r in rows)))
for r in rows:
    if r.get("action") in ("bootstrap","select","train_unit","decision","rescan","review","finish","round_rollover"):
        print("   ",json.dumps({k:r.get(k) for k in ("action","task","result","steps","pass_source","val_nmse","val_mse",
              "reason","stop_reason","deferred","hardness_scan_seconds") if k in r},ensure_ascii=False))
PY
echo "--- ⑥ 显存峰值（nvidia-smi 采样, MiB）---"
$PY - "$OUT/vram.csv" <<'PY'
import datetime,sys
rows=[]
for line in open(sys.argv[1],encoding="utf-8",errors="ignore"):
    p=[x.strip() for x in line.split(",")]
    if len(p)<3 or not p[0][:2].isdigit(): continue
    try: rows.append((datetime.datetime.strptime(p[0],"%Y/%m/%d %H:%M:%S.%f"),int(p[1].split()[0]),int(p[2].split()[0])))
    except Exception: pass
if rows:
    pk=max(rows,key=lambda r:r[1])
    print(f"  采样 {len(rows)} 条 | 峰值 {pk[1]} MiB = {pk[1]/1024:.2f} GiB @ {pk[0].strftime('%H:%M:%S')} (util {pk[2]}%)")
else: print("  (无采样)")
PY
echo "--- ⑦ TensorBoard 标量/文本 ---"
$PY - "$OUT" <<'PY'
import glob,os,sys
OUT=sys.argv[1]
try:
    from tensorboard.backend.event_processing import event_accumulator as ea
except Exception as e:
    print("  (tensorboard event_accumulator 不可用:",e,")"); raise SystemExit
fs=sorted(glob.glob(os.path.join(OUT,"runs","events.out.tfevents.*")))
if not fs: print("  (无 events 文件)"); raise SystemExit
acc=ea.EventAccumulator(fs[-1]); acc.Reload()
tags=sorted(acc.Tags().get("scalars",[]))
print(f"  events={os.path.basename(fs[-1])} 标量标签数={len(tags)}")
want=["training/loss","auto_learning/unit_loss","curriculum/current_task_name","curriculum/passed_tasks",
      "curriculum/target_passed_tasks","curriculum/bootstrap_pass_count","curriculum/newly_passed_count",
      "sampling/replay_samples_per_unit","sampling/replay_distinct_tasks_per_unit","replay/available_tasks",
      "diagnostics/current_task_val_nmse","auto_learning/hardness_scan_mean_loss","auto_learning/hardness_scan_p90_loss",
      "auto_learning/hardness_scan_seconds","current_skill/val_mse","current_skill/val_to_pass_threshold","steptime"]
for t in want:
    if t in tags:
        pts=[(s.step,round(s.value,6)) for s in acc.Scalars(t)]
        print(f"  {t:<52} n={len(pts):<4} {pts[-4:]}")
    else:
        print(f"  {t:<52} (无该标量标签)")
per_task=[t for t in tags if t.startswith("task/") and t.endswith("/unit_loss")]
print(f"  per-task unit_loss 标签: {per_task if per_task else '(无)'}")
try:
    tt=acc.Tags().get("tensors",[])
    print(f"  Text(tensors) 标签: {[t for t in tt][:5]}")
except Exception: pass
PY
echo "--- ⑧ 存档（无存档模式应为空）---"
CKPT=$(ls -d "$OUT"/checkpoints/global_step_* 2>/dev/null | tail -1)
echo "  ckpt 目录数=$CKPT_N  ckpt_dir=${CKPT:-<无>}"
[ -n "$CKPT" ] && { du -sh "$CKPT"/* 2>/dev/null | head -4; echo "  hf_ckpt 分片: $(ls "$CKPT"/hf_ckpt/*.safetensors 2>/dev/null | wc -l)  index.json: $([ -f "$CKPT"/hf_ckpt/index.json ] && echo 有 || echo 无)"; }
echo "  events.jsonl 行数: $(wc -l < "$EVJSON" 2>/dev/null || echo 0) | runs/: $(ls "$OUT/runs" 2>/dev/null | wc -l) 文件 | tb 进程: $(pgrep -cf '[t]ensorboard.main' 2>/dev/null | head -1 || true)"
} 2>&1 | tee "$SUMMARY"

# ---- 判定（无存档模式：不看存档；看 rc/看门狗/日志与 TB 落盘/步数或 AL 正常收工）----
$PY - "$OUT" "$RC" "$STEPS" "$STEPS_DONE" "$SMOKE_NO_CHECKPOINT" "$CKPT_N" > "$OUT/verdict.txt" <<'PY'
import json,os,sys
OUT,RC,STEPS,STEPS_DONE,NOSAVE,CKPT_N=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),sys.argv[5],int(sys.argv[6])
EV=os.path.join(OUT,"auto_learning_events.jsonl")
reasons=[]; notes=[]
if RC!=0: reasons.append(f"训练进程 rc={RC}（非 0）")
if os.path.exists(os.path.join(OUT,"watchdog_killed")): reasons.append("看门狗触发（显存超限）")
log=os.path.join(OUT,"log.txt")
if not (os.path.exists(log) and os.path.getsize(log)>0): reasons.append("log.txt 缺失或为空")
rows=[]
if os.path.exists(EV):
    for l in open(EV,encoding="utf-8"):
        if l.strip():
            try: rows.append(json.loads(l))
            except Exception: reasons.append("events.jsonl 存在非法 JSON 行")
if not rows: reasons.append("events.jsonl 为空")
tb=[f for f in os.listdir(os.path.join(OUT,"runs")) if f.startswith("events.out.tfevents")] if os.path.isdir(os.path.join(OUT,"runs")) else []
if not tb: reasons.append("TensorBoard events 未落盘")
fin=[r for r in rows if r.get("action")=="finish"]
stop=fin[-1].get("stop_reason") or fin[-1].get("reason") or "" if fin else ""
OK_EARLY=("target_total_passed_reached" in stop or "all_tasks_resolved" in stop
          or "max_new_tasks_passed_reached" in stop or "unit_budget_reached" in stop
          or "max_global_steps" in stop or "max_transitions" in stop)
if STEPS_DONE==STEPS:
    notes.append(f"跑满预期 {STEPS} 步 ⇒ 正常（MAX_STEPS）")
elif OK_EARLY:
    notes.append(f"AL 正常收工（stop_reason={stop}）⇒ 提前结束属预期，非异常")
else:
    reasons.append(f"步数 {STEPS_DONE} != 期望 {STEPS} 且无合法 AL 收工原因（stop_reason={stop!r}）")
if NOSAVE=="1":
    if CKPT_N==0: notes.append("无存档模式生效：未创建任何 checkpoints/ 目录 ✅")
    else: reasons.append(f"无存档模式却产生了 {CKPT_N} 个 checkpoint 目录（模式泄漏）")
else:
    if CKPT_N==0: reasons.append("正式模式但没有任何 checkpoint")
    else: notes.append(f"正式模式：产生 {CKPT_N} 个 checkpoint ✅")
print("VERDICT=" + ("OK" if not reasons else "FAIL"))
print("stop_reason="+ (stop or "<无 finish 事件>"))
for n in notes: print("OK_NOTE: "+n)
for r in reasons: print("FAIL_REASON: "+r)
PY
cat "$OUT/verdict.txt"
VERDICT=$(grep -m1 '^VERDICT=' "$OUT/verdict.txt" | cut -d= -f2)
printf 'rc=%s verdict=%s steps=%s/%s stop=%s nosave=%s ckpt_n=%s\n' "$RC" "$VERDICT" "$STEPS_DONE" "$STEPS" \
       "$(grep -m1 '^stop_reason=' "$OUT/verdict.txt" | cut -d= -f2-)" "$SMOKE_NO_CHECKPOINT" "$CKPT_N" > "$OUT/RUN_DONE"
sync
echo "=== 汇总与判定已落盘：$SUMMARY / $OUT/verdict.txt / $OUT/RUN_DONE ==="
if [ "$DRY_RUN" = "1" ]; then
  echo "=== [DRY_RUN] 判定演练完成，不关机、不退卡 ==="
elif [ "$VERDICT" = "OK" ]; then
  echo "=== ✅ 判定通过 ⇒ 8 秒后自动关机（日志已 sync 落盘）==="; sleep 8; echo "SHUTDOWN_NOW $(date '+%F %T')"; sync; /usr/bin/shutdown
else
  echo "=== ❌ 判定未通过 ⇒ 不关机，等诊断 ==="
fi
echo "=== JOB_END $(date '+%F %T') ==="
