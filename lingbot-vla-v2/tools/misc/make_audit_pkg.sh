#!/usr/bin/env bash
# =============================================================================
# 10-step AL Smoke 审计数据包（不改源码 / 不跑训练 / 不打包权重与凭据）
#   产出：/data/tmp/al_smoke_tb10_audit.tar.gz + 目录 /data/tmp/al_smoke_tb10_audit/
# =============================================================================
set -uo pipefail
PY=/data/miniconda3/envs/lingbotvla/bin/python
OUT=/data/outputs/al_smoke_gbs4_4task_tb10
FAIL1=/data/outputs/al_smoke_gbs4_4task_tb10_botched_run1
P1=/data/outputs/al_smoke_gbs4_4task_p1_botched_run1
REPO=/data/code/lingbot-vla-v2
ROOT=/data/tmp/al_smoke_tb10_audit
rm -rf "$ROOT"; mkdir -p "$ROOT"/{run_ok,run_failed_attempt1,run_failed_p1,config,reconcile,tb}
MISS="$ROOT/MISSING_FILES.md"; : > "$MISS"
note() { echo "- $1" >> "$MISS"; }
copy() { # copy <src> <dstdir> [label]
  local s="$1" d="$2" l="${3:-}"
  if [ -e "$s" ]; then mkdir -p "$d"; cp -a "$s" "$d/"; echo "  ✅ $l$(basename "$s")  ($(du -h "$s" | cut -f1))"
  else echo "  ❌ 不存在: $s"; note "$l\`$s\` — **不存在**"; fi
}
echo "=== ① 目录清单（先看有什么）==="
ls -la "$OUT" 2>/dev/null | sed 's/^/  /'
echo "=== ② 原始日志 / 事件 / 显存 / runner 输出 ==="
copy "$OUT/log.txt"                       "$ROOT/run_ok" "训练日志(OUT 内副本) "
copy "$REPO/log.txt"                      "$ROOT/run_ok" "训练日志(仓库 tee) "
copy "$OUT/auto_learning_events.jsonl"    "$ROOT/run_ok" "AL 事件流 "
copy "$OUT/vram.csv"                      "$ROOT/run_ok" "显存采样 "
copy "$OUT/watchdog.log"                  "$ROOT/run_ok" "看门狗日志 "
copy "$OUT/watchdog_killed"               "$ROOT/run_ok" "看门狗触发标记 "
copy "$OUT/tb.log"                        "$ROOT/run_ok" "TB 启动日志 "
copy /data/tmp/smoke_tb10_v2_job.log      "$ROOT/run_ok" "runner stdout/stderr "
copy /data/tmp/smoke_tb10_v2_summary.txt  "$ROOT/run_ok" "runner 汇总 "
copy "$OUT/lingbotvla_cli.yaml"           "$ROOT/run_ok" "本次实际生效配置转储 "
# history / registry / metrics 类（如存在则收，不存在明确标注）
for f in history.json registry.json metrics.json metrics.csv history.csv registry.csv al_state.json; do
  [ -e "$OUT/$f" ] && copy "$OUT/$f" "$ROOT/run_ok" "AL 状态文件 " || note "\`$OUT/$f\` — **不存在**（该实现把 AL 状态存在 DCP 的 extra_state 中，无独立 history/registry/metrics 文件）"
done
echo "=== ③ TensorBoard events（成功 run 与失败 run 分开）==="
copy "$OUT/runs"   "$ROOT/tb/success_run"  "TB events(成功 run) "
[ -d "$FAIL1/runs" ] && { mkdir -p "$ROOT/tb/failed_attempt1"; cp -a "$FAIL1/runs/." "$ROOT/tb/failed_attempt1/"; echo "  ✅ TB events(失败 attempt1，已分离)"; } || note "失败 attempt1 的 runs/ 不存在"
[ -d "$P1/runs" ]   && { mkdir -p "$ROOT/tb/failed_p1"; cp -a "$P1/runs/." "$ROOT/tb/failed_p1/"; echo "  ✅ TB events(更早 p1 失败轮，已分离)"; }
echo "=== ④ 配置 / 启动脚本 / 实际生效参数 ==="
copy "$REPO/configs/auto_learning/smoke_gbs4_4task_tb5.yaml" "$ROOT/config" "smoke YAML "
copy "$REPO/experiment/robotwin/al_50task_bf16.sh"           "$ROOT/config" "启动器 "
copy "$REPO/train.sh"                                        "$ROOT/config" "train.sh "
copy /data/tmp/run_smoke_tb10_48g_v2.sh                      "$ROOT/config" "本轮 runner "
copy /data/tmp/run_smoke_tb10_48g.sh                         "$ROOT/config" "上一版 runner "
# 实际生效参数：从日志里抽 args 的 JSON 块
LOGSRC="$OUT/log.txt"; [ -s "$LOGSRC" ] || LOGSRC="$REPO/log.txt"
if [ -s "$LOGSRC" ]; then
  $PY - "$LOGSRC" "$ROOT/config/effective_args.json" <<'PY'
import json,re,sys
raw=open(sys.argv[1],encoding="utf-8",errors="ignore").read()
i=raw.find('{\n  "model"'); j=raw.find('\n}\n', i)
if i>=0 and j>i:
    try:
        obj=json.loads(raw[i:j+2]); json.dump(obj,open(sys.argv[2],"w",encoding="utf-8"),ensure_ascii=False,indent=2)
        print("  ✅ effective_args.json 已抽取（trainer 启动时打印的 asdict(args)）")
    except Exception as e: print("  ⚠️ args 抽取失败:",e)
else: print("  ⚠️ 日志里没找到 args JSON 块")
PY
fi
echo "=== ⑤ Git commit / 关键文件 SHA256 ==="
{ echo "# 版本与哈希"; echo; echo '```'; git -C /data/code rev-parse HEAD 2>/dev/null | sed 's/^/git commit: /'
  git -C "$REPO" log --oneline -1 2>/dev/null | sed 's/^/HEAD: /'
  echo; echo "# SHA256"
  sha256sum "$REPO/configs/auto_learning/smoke_gbs4_4task_tb5.yaml" \
            /data/eval_results/open_loop/ref50k/pass_thresholds_gmean200_warn.json \
            /data/train/task_splits_50/task_baseline.json \
            /data/eval_results/open_loop/ref50k/ref_per_traj.jsonl 2>/dev/null
  echo '```'; } > "$ROOT/config/VERSION_AND_HASHES.md"
sed -n '1,40p' "$ROOT/config/VERSION_AND_HASHES.md"
echo "=== ⑥ Hardness 两次扫描 + Checkpoint 清单 ==="
$PY - <<'PY' > "$ROOT/reconcile/C_hardness_and_ckpt.md" 2>&1
import json,os,glob,re,datetime
OUT="/data/outputs/al_smoke_gbs4_4task_tb10"
print("# C. Hardness 扫描对账 & Checkpoint 清单\n")
ev=os.path.join(OUT,"auto_learning_events.jsonl")
rows=[json.loads(l) for l in open(ev,encoding="utf-8") if l.strip()] if os.path.exists(ev) else []
print("## Hardness 扫描（事件流，权威）")
sel=[r for r in rows if r.get("action")=="select"]
for k,r in enumerate(sel,1):
    print(f"- 第 {k} 次: task=`{r.get('task')}` | seconds=**{r.get('hardness_scan_seconds')}** | is_first={r.get('hardness_scan_is_first')} "
          f"| n_scanned={r.get('n_scanned')} | n_total={r.get('n_samples')} | trajs={len(r.get('scanned_traj_ids') or [])} | GPU/CPU/IO 未分别埋点（仅 wall-time）")
print("\n## TensorBoard 是否记录（含第二次）")
try:
    from tensorboard.backend.event_processing import event_accumulator as ea
    fs=sorted(glob.glob(os.path.join(OUT,"runs","events.out.tfevents.*")))
    acc=ea.EventAccumulator(fs[-1]); acc.Reload()
    for tag in ("auto_learning/hardness_scan_seconds","auto_learning/hardness_scan_seconds_warmup",
                "auto_learning/hardness_scan_seconds_steady","auto_learning/hardness_scan_samples","auto_learning/hardness_scan_trajs"):
        pts=[(s.step,round(s.value,4)) for s in acc.Scalars(tag)] if tag in acc.Tags().get("scalars",[]) else None
        print(f"- `{tag}`: {pts if pts else '**无该标签**'}")
except Exception as e:
    print("- (TB 解析失败:",e,")")
print("\n## 日志时间戳（扫描起止）")
rep="/data/outputs/al_smoke_gbs4_4task_tb10/log.txt"
if not os.path.exists(rep): rep="/data/code/lingbot-vla-v2/log.txt"
if os.path.exists(rep):
    for l in open(rep,encoding="utf-8",errors="ignore"):
        if re.search(r"eval .*n_ids=4|TrainRequest|Step 50\d/|unit 没跑满|收尾|DiskCheck|async_hf|Distributed checkpoint",l):
            print("- "+l.rstrip()[:190])
print("\n## global_step_510 文件清单（不含权重内容）")
ck=os.path.join(OUT,"checkpoints","global_step_510")
if os.path.isdir(ck):
    tot=0
    for root,dirs,files in os.walk(ck):
        for f in sorted(files):
            p=os.path.join(root,f); s=os.path.getsize(p); tot+=s
            if f.endswith(".safetensors") or f.endswith(".distcp") or f.endswith(".pt"):
                print(f"- `{os.path.relpath(p,ck)}` — {s/2**20:.1f} MiB **(仅清单，未打包)**")
            else:
                print(f"- `{os.path.relpath(p,ck)}` — {s/1024:.1f} KiB")
    print(f"- 合计 {tot/2**30:.2f} GiB")
    hf=os.path.join(ck,"hf_ckpt")
    print(f"- `hf_ckpt/` 存在: {os.path.isdir(hf)} | index.json: {os.path.isfile(os.path.join(hf,'index.json'))} | 分片数: {len(glob.glob(os.path.join(hf,'*.safetensors')))}")
    tmp=glob.glob(os.path.join(ck,".hf_ckpt.tmp.*"))
    print(f"- ⚠️ 未完成的 HF 临时目录: {tmp if tmp else '无'}")
else:
    print("- **不存在** `checkpoints/global_step_510`")
PY
echo "  → $ROOT/reconcile/C_hardness_and_ckpt.md"
echo "=== ⑦ A. Unit Loss 对账（501-510 原始值 + 两个五步均值）==="
$PY - <<'PY' > "$ROOT/reconcile/A_unit_loss.md" 2>&1
import glob,os,statistics,json
OUT="/data/outputs/al_smoke_gbs4_4task_tb10"
print("# A. Unit Loss 对账（独立计算，不依赖曲线外观）\n")
loss={}
try:
    from tensorboard.backend.event_processing import event_accumulator as ea
    fs=sorted(glob.glob(os.path.join(OUT,"runs","events.out.tfevents.*")))
    acc=ea.EventAccumulator(fs[-1]); acc.Reload()
    loss={s.step: s.value for s in acc.Scalars("training/loss")}
    print("## `training/loss` 原始值（TB events）")
    for k in sorted(loss): print(f"- step {k}: {loss[k]:.6f}")
    u={s.step: s.value for s in acc.Scalars("auto_learning/unit_loss")} if "auto_learning/unit_loss" in acc.Tags().get("scalars",[]) else {}
    print("\n## `auto_learning/unit_loss`（TB events）")
    for k in sorted(u): print(f"- step {k}: {u[k]:.6f}")
    def mean(rng):
        v=[loss[k] for k in rng if k in loss]
        return (statistics.mean(v), len(v)) if v else (float('nan'),0)
    m1,n1=mean(range(501,506)); m2,n2=mean(range(506,511))
    print(f"\n## 独立计算的两个五步均值")
    print(f"- 501–505: mean = **{m1:.6f}**  (n={n1})")
    print(f"- 506–510: mean = **{m2:.6f}**  (n={n2})")
    print(f"\n## 与 TB 的 unit_loss 差异")
    for step,m in ((505,m1),(510,m2)):
        if step in u:
            d=u[step]-m; rel=d/u[step]*100 if u[step] else float('nan')
            print(f"- @{step}: unit_loss={u[step]:.6f} vs 独立均值={m:.6f} ⇒ 差 {d:+.6f} ({rel:+.3f}%)  {'✅ 一致' if abs(rel)<0.5 else '⚠️ 不一致，需查'}")
        else: print(f"- @{step}: **TB 无 unit_loss 点**")
    print("\n> 说明：`unit_loss` 由 `TrainResult.loss = mean(per_step_losses)`（`real/hook.py`）给出；"
          "若与上面的独立均值不一致，通常意味着 unit 内的 step 与 `training/loss` 的 step 口径不同（如首步/边界步）。")
except Exception as e:
    print("TB 解析失败:",e)
PY
echo "  → $ROOT/reconcile/A_unit_loss.md"
echo "=== ⑧ B. Replay 对账 ==="
$PY - <<'PY' > "$ROOT/reconcile/B_replay.md" 2>&1
import json,os,collections
OUT="/data/outputs/al_smoke_gbs4_4task_tb10"
print("# B. Replay 对账（Bootstrap 两个 PASS 有没有进 Replay 池）\n")
ev=os.path.join(OUT,"auto_learning_events.jsonl")
rows=[json.loads(l) for l in open(ev,encoding="utf-8") if l.strip()] if os.path.exists(ev) else []
print(f"事件行数: {len(rows)}\n")
print("## bootstrap（谁 PASS 了）")
for r in rows:
    if r.get("action")=="bootstrap":
        print(f"- {r.get('task')}: result={r.get('result')} pass_source={r.get('pass_source')} scout_nmse={r.get('scout_nmse')}")
print("\n## 每个 unit / select 的全部相关字段")
KEYS=("action","task","step","tb_step","steps","deferred","pass_source","n_new","n_old","old_tasks","replay_slots",
      "replay_plan","samples_seen","batches_built","unique_batches","new_slot_counts","old_slot_counts","probs")
for r in rows:
    if r.get("action") in ("select","train_unit","rescan","review","finish","round_rollover"):
        print("\n```json"); print(json.dumps({k:r.get(k) for k in KEYS if k in r},ensure_ascii=False,indent=1)); print("```")
print("\n## `memory/pass_pool_size` 等 TB 指标")
try:
    import glob
    from tensorboard.backend.event_processing import event_accumulator as ea
    fs=sorted(glob.glob(os.path.join(OUT,"runs","events.out.tfevents.*")))
    acc=ea.EventAccumulator(fs[-1]); acc.Reload()
    for tag in ("memory/pass_pool_size","memory/reopen_count","memory/forgotten_count"):
        if tag in acc.Tags().get("scalars",[]):
            print(f"- `{tag}`: {[(s.step,round(s.value,3)) for s in acc.Scalars(tag)]}")
        else: print(f"- `{tag}`: 无")
except Exception as e: print("- TB 解析失败:",e)
print("\n## 结论（自动判定，人工复核）")
tt=[r for r in rows if r.get("action")=="train_unit"]
if tt:
    for r in tt:
        n_old=r.get("n_old"); tasks=r.get("old_tasks") or r.get("replay_plan")
        print(f"- unit task={r.get('task')} step={r.get('tb_step') or r.get('step')}: n_old={n_old} old_tasks/replay_plan={tasks}")
    print("\n⇒ 若所有 unit 的 `n_old` 都为 0 或 `old_tasks` 为空，则**Bootstrap PASS 的 click_bell/click_alarmclock 未被采样**；"
          "下一步按 design 追查：`Scheduler.replay_plan()` 是否要求 PASS 时已存在 snapshot（`replay_sample_policy=pass_snapshot`），"
          "以及 `build.py` 里 replay 池的填充时机（初始化 vs PASS 时）。**发现 Bug 只报告，不改 Scheduler。**")
else:
    print("- 事件流里没有 train_unit 记录")
PY
echo "  → $ROOT/reconcile/B_replay.md"
echo "=== ⑨ README / 清单 / 打包 ==="
{
echo "# al_smoke_tb10 审计数据包"
echo
echo "来源：\`/data/outputs/al_smoke_gbs4_4task_tb10/\`（48G 单卡 BF16，MICRO=1/GAS=4/GBS=4，5-step unit，STEP_OFFSET=500，MAX_STEPS=510，torch.compile **已禁用**）"
echo
echo "## 目录"
echo '```'
echo "run_ok/            成功 run 的原始日志 / 事件 / 显存 / runner 输出"
echo "tb/success_run/    成功 run 的 TensorBoard events（原始）"
echo "tb/failed_attempt1/ 之前的失败 run（被看门狗杀掉）events，**已分离标记**"
echo "tb/failed_p1/      更早一轮失败的 events，**已分离标记**"
echo "config/            YAML / 启动器 / runner / VERSION_AND_HASHES.md / effective_args.json"
echo "reconcile/         A_unit_loss.md / B_replay.md / C_hardness_and_ckpt.md"
echo "MISSING_FILES.md   本次请求但不存在的文件（明确标注）"
echo "MANIFEST.sha256    所有打包文件的 SHA256"
echo '```'
echo
echo "## 未打包（按规则排除）"
echo "- DCP（\`*.distcp\`）、HF 权重（\`hf_ckpt/*.safetensors\`）、数据集、大型缓存、任何凭据"
echo "- Checkpoint 只提供**文件清单与大小**（见 \`reconcile/C_hardness_and_ckpt.md\`）"
} > "$ROOT/README_AUDIT.md"
( cd "$ROOT" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )
( cd "$ROOT" && find . -type f | sort | sed 's/^/  /' )
TAR=/data/tmp/al_smoke_tb10_audit.tar.gz
tar czf "$TAR" -C /data/tmp al_smoke_tb10_audit
echo "=== ✅ 打包完成 ==="; ls -la "$TAR"; sha256sum "$TAR"
echo "=== 缺失清单 ==="; cat "$MISS"
