#!/usr/bin/env bash
# =============================================================================
# tools/al_b1_regression.sh
#   B1 收口修复（commit 265b350）之后的 **48GB BF16 短回归 A–G**
# -----------------------------------------------------------------------------
# 目的：把 `stage_b1_repo_review_v0_2.md` §14 冻结 Gate 的**最后一项**跑完。
#       只做**机制回归**，不重跑长训练；除特别说明外都是 bf16 短跑（省显存省时）。
#
# 用法（在 GPU 机上，任意目录）:
#   bash tools/al_b1_regression.sh                 # 跑 A–G 全部
#   ONLY=F,G bash tools/al_b1_regression.sh        # 只跑指定项（逗号分隔）
#   DRY_RUN=1 bash tools/al_b1_regression.sh       # 只打印计划与命令
#   KEEP=1 bash tools/al_b1_regression.sh          # 保留上次产物（默认每次新建时间戳目录）
#
# 产物: /data/tmp/al_b1_regression/<RUN_ID>/{logs,reports}/  +  summary.txt
#
# 前置（脚本会自检）:
#   ① export QWEN3VL_PATH=/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct
#   ② /data/train/task_splits{,2}/manifest.json 存在
#   ③ 磁盘 ≥ 60G（每份 DCP 约 31G，本脚本 PRUNE=0 且各 run 独立 output_dir）
#
# ⚠️ 本脚本**不自动关机** —— 全部跑完会打印一行提示，由你决定 shutdown。
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-/data/miniconda3/envs/lingbotvla/bin/python}
# 🔴 torchrun 在 conda env 的 bin 里；脚本被 `bash tools/...` 直接调用时 PATH 常没有它
#    ⇒ 显式补上（首次跑出 rc=127 `torchrun: command not found`，2026-10-07）
export PATH="$(dirname "$PY"):$PATH"
export PY
PHASES=${PHASES:-/data/train/phases}
SPLIT_1=${SPLIT_1:-/data/train/task_splits}
SPLIT_2=${SPLIT_2:-/data/train/task_splits_2task}
CFG_YAML=${CFG_YAML:-/data/train/configs/robotwin_official_paths.yaml}
# baseline 要用**完整 cli yaml**：`robotwin_official_paths.yaml` 只是「路径文件」，
# 缺 `data.prompt_type` / `data.img_size` 等字段 ⇒ compute_task_baseline 会 AttributeError。
BASELINE_CFG=${BASELINE_CFG:-/data/outputs/single/click_bell/lingbotvla_cli.yaml}
MODEL_BASE=${MODEL_BASE:-/data/models/lingbot-vla-v2-6b-base/lingbot-vla-v2-6b}
# 评测/训练都要求 QWEN3VL_PATH 真的 export（否则 4 个推理 server 全 DEAD）——
# 与 experiment/robotwin/single_task_train.sh 一致：默认给上，可用 QWEN3VL=... 覆盖。
export QWEN3VL_PATH=${QWEN3VL:-/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct}

ONLY=${ONLY:-A,B,C,D,E,F,G}
DRY_RUN=${DRY_RUN:-0}
KEEP=${KEEP:-0}

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-/data/tmp/al_b1_regression}
OUT="$OUT_ROOT/$RUN_ID"
LOGS="$OUT/logs"
REPORTS="$OUT/reports"
SUMMARY="$OUT/summary.txt"

# 基线（P1-5 现在**默认必填**）。脚本会在 step 0 尝试算出它；算不出来才退回 override。
BASELINE_1="$SPLIT_1/task_baseline.json"
BASELINE_2="$SPLIT_2/task_baseline.json"

mkdir -p "$LOGS" "$REPORTS"
: > "$SUMMARY"

# ---- 小工具 ------------------------------------------------------------------
_c() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
_record() {  # _record <item> <PASS|FAIL|SKIP> <note>
    printf '[%s] %s — %s\n' "$2" "$1" "$3" | tee -a "$SUMMARY"
}
_want() { [[ ",$ONLY," == *",$1,"* ]]; }

_have_qwen() { [ -n "${QWEN3VL_PATH:-}" ] && [ -d "${QWEN3VL_PATH}" ]; }

# 单次 AL 训练（bf16 / micro=1 / gas=10 / gbs=10 ⇒ 48G 够）
#   al_train <tag> <out_dir> <manifest> <cfg_yaml> <episode_ids> <max_steps> <save_steps> <resume:0|1> [extra args...]
al_train() {
    local tag="$1" out="$2" manifest="$3" cfg="$4" ids="$5"
    local max_steps="$6" save_steps="$7" resume="$8"; shift 8
    local rb; [ "$resume" = "1" ] && rb=true || rb=false
    local log="$LOGS/${tag}.log"
    # 可选提速开关：NUM_WORKERS=<n> ⇒ 覆盖 config 的 data.num_workers（默认 8）。
    # B1 收口后 sampler 走 `stats_upto(已消费步数)`，**允许 num_workers>0（含 prefetch）**，
    # 所以可以安全调大以并行化视频解码（num_workers=0 时单步会慢一个数量级）。
    local extra_workers=()
    [ -n "${NUM_WORKERS:-}" ] && extra_workers=(--data.num_workers "$NUM_WORKERS")
    local cmd=(
      bash "$REPO/train.sh" tasks/vla/train_lingbotvla.py "$CFG_YAML"
      --model.model_path "$MODEL_BASE"
      --data.train_path "$PHASES/datasets.txt"
      --data.episode_ids_file "$ids"
      --train.output_dir "$out"
      --train.micro_batch_size 1
      --train.gradient_accumulation_steps 10
      --train.global_batch_size 10
      --train.num_train_epochs 2
      --train.max_steps "$max_steps"
      --train.save_steps "$save_steps"
      --train.save_epochs 0
      --train.save_hf_weights true
      --train.async_save_hf_weights false
      --train.enable_resume "$rb"
      --train.train_expert_only false
      --train.freeze_vision_encoder true
      --train.enable_mixed_precision false
      --data.image_augment false
      --train.disk_guard false
      --train.auto_learning "$cfg"
      --train.auto_learning_manifest "$manifest"
      --train.auto_learning_baseline "$BASELINE_FOR_RUN"
      "${extra_workers[@]+"${extra_workers[@]}"}"
      "${@}"
    )
    echo "[al_train] $tag ⇒ $out  (max_steps=$max_steps save_steps=$save_steps resume=$rb)" | tee -a "$log"
    if [ "$DRY_RUN" = "1" ]; then printf '  %q' "${cmd[@]}"; echo; return 0; fi
    ( cd "$REPO" && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        QWEN3VL_PATH="$QWEN3VL_PATH" \
        PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} \
        "${cmd[@]}" ) >> "$log" 2>&1
    local rc=$?
    echo "[al_train] $tag rc=$rc" | tee -a "$log"
    return $rc
}

_al_trained_steps() {  # 从 log 里取最后一个 "Step N/" 的 N
    grep -oE 'Step [0-9]+/' "$1" 2>/dev/null | tail -1 | tr -dc '0-9'
}
# =============================================================================
_c "前置自检"
PRECHECK_OK=1
[ -x "$PY" ] || { echo "❌ 找不到 $PY"; PRECHECK_OK=0; }
[ -f "$CFG_YAML" ] || { echo "❌ 找不到 $CFG_YAML"; PRECHECK_OK=0; }
[ -f "$SPLIT_1/manifest.json" ] || { echo "❌ 缺 $SPLIT_1/manifest.json（先跑 tools/task_split.py）"; PRECHECK_OK=0; }
[ -f "$SPLIT_2/manifest.json" ] || echo "⚠️  缺 $SPLIT_2/manifest.json ⇒ E 项会跳过（先跑 tools/task_split.py --task click_bell,click_alarmclock）"
_have_qwen || echo "⚠️  QWEN3VL_PATH 未 export 或不存在 ⇒ 训练/评测起不来。先: export QWEN3VL_PATH=/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct"
AV_GB=$(df -BG --output=avail /data 2>/dev/null | tail -1 | tr -dc '0-9'); AV_GB=${AV_GB:-0}
echo "磁盘可用 ${AV_GB}G（建议 ≥60G；每份 DCP≈31G）"
git -C "$REPO" rev-parse --short HEAD | sed 's/^/仓库 HEAD = /'
[ "$PRECHECK_OK" = "1" ] || { echo "❌ 前置未通过，退出"; exit 2; }

# ---- step 0：baseline（无 GPU 也能算；P1-5 现在默认必填）---------------------
_c "step 0 — Fixed Baseline（P1-5 必填；有缓存则秒过）"
BASELINE_FOR_RUN="__NONE__"
if [ -f "$BASELINE_1" ]; then
    BASELINE_FOR_RUN="$BASELINE_1"; echo "✅ 已有 baseline 缓存，直接用: $BASELINE_1"
elif [ "$DRY_RUN" = "1" ]; then
    echo "(dry-run) 将尝试: compute_task_baseline --manifest $SPLIT_1/manifest.json --out $BASELINE_1"
else
    "$PY" -u -m lingbotvla.auto_learning.tools.compute_task_baseline \
        --manifest "$SPLIT_1/manifest.json" --config "$BASELINE_CFG" \
        --out "$BASELINE_1" --tasks click_bell \
        > "$LOGS/baseline_1task.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$BASELINE_1" ]; then
        BASELINE_FOR_RUN="$BASELINE_1"; echo "✅ baseline 就绪: $BASELINE_1"
    else
        echo "⚠️  baseline 算失败（rc=$rc，见 $LOGS/baseline_1task.log）⇒ 退回 allow_missing_baseline"
    fi
fi

# 没有真 baseline 时，生成一个 override 配置（显式声明「这是 smoke，允许无基线」）
AL_CFG_G10="$REPO/configs/auto_learning/g10_click_bell.yaml"
AL_CFG_2T="$REPO/configs/auto_learning/smoke_2task.yaml"
if [ "$BASELINE_FOR_RUN" = "__NONE__" ]; then
    AL_CFG_G10="$REPORTS/g10_click_bell.nobaseline.yaml"
    AL_CFG_2T="$REPORTS/smoke_2task.nobaseline.yaml"
    if [ "$DRY_RUN" != "1" ]; then
        cp "$REPO/configs/auto_learning/g10_click_bell.yaml" "$AL_CFG_G10"
        echo 'allow_missing_baseline: true' >> "$AL_CFG_G10"
        cp "$REPO/configs/auto_learning/smoke_2task.yaml" "$AL_CFG_2T"
        echo 'allow_missing_baseline: true' >> "$AL_CFG_2T"
        echo "（已生成 override 配置：$AL_CFG_G10 / $AL_CFG_2T）"
    else
        echo "(dry-run) 真跑时会生成 override 配置（追加 allow_missing_baseline: true）"
    fi
fi

# =============================================================================
# A — Legacy 3-step disabled parity（§28：数据/标签/mask 逐值一致，不允许容差）
if _want A; then
    _c "A — Legacy vs Integration 对拍"
    LG=${LEGACY_REPO:-/data/tmp/legacy/lingbot-vla-v2}
    if [ ! -d "$LG" ]; then
        _record A SKIP "找不到 LEGACY checkout $LG（用 LEGACY_REPO=... 指定）"
    elif [ "$DRY_RUN" = "1" ]; then
        echo "(dry-run) 将对拍 $LG 与 $REPO"
    else
        "$PY" -u "$REPO/tools/al_legacy_parity.py" --repo "$REPO" --config "$CFG_YAML" \
            --episode-ids "$SPLIT_1/click_bell.val_ids.json" \
            --out "$REPORTS/parity_integration.json" > "$LOGS/parity_integration.log" 2>&1
        r1=$?
        "$PY" -u "$REPO/tools/al_legacy_parity.py" --repo "$LG" --config "$CFG_YAML" \
            --episode-ids "$SPLIT_1/click_bell.val_ids.json" \
            --out "$REPORTS/parity_legacy.json" > "$LOGS/parity_legacy.log" 2>&1
        r2=$?
        if [ $r1 -eq 0 ] && [ $r2 -eq 0 ]; then
            "$PY" -u "$REPO/tools/al_legacy_parity.py" --diff \
                "$REPORTS/parity_legacy.json" "$REPORTS/parity_integration.json" \
                > "$LOGS/parity_diff.log" 2>&1
            [ $? -eq 0 ] && _record A PASS "LEGACY == INTEG（见 logs/parity_diff.log）" \
                         || _record A FAIL "diff 不为空（见 logs/parity_diff.log）"
        else
            _record A FAIL "dump 失败 rc=$r1/$r2"
        fi
    fi
fi

# =============================================================================
# B / C — 真实模型侧的确定性 & 2→4（复用 R 系列测试，需要一份 hf_ckpt）
if _want B || _want C; then
    _c "B/C — R 系列（hardness 确定性 / 2→4 缓存）"
    CKPT=${AL_TEST_MODEL_PATH:-}
    if [ -z "$CKPT" ]; then
        # 自动找一份：优先 al_g10 / al_2task，其次 base 目录
        for cand in /data/outputs/al_g10/checkpoints/global_step_*/hf_ckpt \
                    /data/outputs/al_2task/checkpoints/global_step_*/hf_ckpt \
                    "$MODEL_BASE"; do
            [ -d "$cand" ] && CKPT="$cand" && break
        done
    fi
    if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
        _want B && _record B SKIP "找不到 hf_ckpt（用 AL_TEST_MODEL_PATH=... 指定）"
        _want C && _record C SKIP "同上"
    else
        export AL_TEST_MODEL_PATH="$CKPT"
        export AL_TEST_CONFIG=${AL_TEST_CONFIG:-/data/outputs/single/click_bell/lingbotvla_cli.yaml}
        export AL_TEST_MANIFEST=${AL_TEST_MANIFEST:-$SPLIT_1/manifest.json}
        export AL_TEST_TASK=${AL_TEST_TASK:-click_bell}
        echo "用 ckpt: $CKPT"
        if [ "$DRY_RUN" = "1" ]; then
            echo "(dry-run) pytest tests/test_auto_learning_real_model.py -k R2/R5b/R6"
        else
            if _want B; then
                ( cd "$REPO" && "$PY" -m pytest \
                    tests/test_auto_learning_real_model.py -q \
                    -k "R2_hardness_is_deterministic or R5b_hardness_bf16" ) \
                    > "$LOGS/B_hardness.log" 2>&1
                [ $? -eq 0 ] && _record B PASS "R2 + R5b 全绿" || _record B FAIL "见 logs/B_hardness.log"
            fi
            if _want C; then
                ( cd "$REPO" && "$PY" -m pytest \
                    tests/test_auto_learning_real_model.py -q -k "R6_two_then_four" ) \
                    > "$LOGS/C_2to4.log" 2>&1
                [ $? -eq 0 ] && _record C PASS "R6 全绿" || _record C FAIL "见 logs/C_2to4.log"
            fi
        fi
    fi
fi

# =============================================================================
# F — 50-step unit 当场回填（本次 P0-2 的核心）
#   判据：MAX_STEPS=50 + SAVE_EVERY=50 ⇒ 存档**正好落在 unit 边界**；
#         重启 RESUME=1 必须**干净接上**（既不 fail-fast，也不出现「从头重跑」）。
if _want F; then
    _c "F — 50-step unit 当场回填 + 边界存档可恢复"
    F_OUT=/data/outputs/al_reg_F
    if [ "$DRY_RUN" = "1" ]; then
        al_train F1 "$F_OUT" "$SPLIT_1/manifest.json" "$AL_CFG_G10" \
            "$SPLIT_1/click_bell.train_ids.json" 50 50 0 --train.skip_final_save_on_max_steps true
        al_train F2 "$F_OUT" "$SPLIT_1/manifest.json" "$AL_CFG_G10" \
            "$SPLIT_1/click_bell.train_ids.json" 60 0 1 --train.skip_final_save_on_max_steps true
        echo "(dry-run) 另会断言 log 里不出现「从头重跑」/「unit 中途」"
    else
        rm -rf "$F_OUT"
        al_train F1 "$F_OUT" "$SPLIT_1/manifest.json" "$AL_CFG_G10" \
            "$SPLIT_1/click_bell.train_ids.json" 50 50 0 --train.skip_final_save_on_max_steps true
        rc1=$?
        al_train F2 "$F_OUT" "$SPLIT_1/manifest.json" "$AL_CFG_G10" \
            "$SPLIT_1/click_bell.train_ids.json" 60 0 1 --train.skip_final_save_on_max_steps true
        rc2=$?
        bad=$(grep -cE '从头重跑|unit \*\*中途\*\*|落在 learning unit' "$LOGS/F2.log" 2>/dev/null); bad=${bad:-1}
        [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] && [ "$bad" = "0" ] \
            && _record F PASS "step50 对齐存档 + RESUME 干净接上（无「从头重跑」）" \
            || _record F FAIL "rc1=$rc1 rc2=$rc2 可疑日志行=$bad（见 logs/F*.log）"
    fi
fi

# =============================================================================
# E — 2 任务 7+3 smoke（真实 PASS 池 ⇒ Replay）
if _want E; then
    _c "E — 2 任务 7+3 smoke"
    if [ ! -f "$SPLIT_2/manifest.json" ]; then
        _record E SKIP "缺 $SPLIT_2/manifest.json"
    else
        E_OUT=/data/outputs/al_reg_E
        if [ "$DRY_RUN" = "1" ]; then
            al_train E1 "$E_OUT" "$SPLIT_2/manifest.json" "$AL_CFG_2T" \
                "$SPLIT_2/combined.train_ids.json" 12 0 0 --train.skip_final_save_on_max_steps true
        else
            rm -rf "$E_OUT"
            al_train E1 "$E_OUT" "$SPLIT_2/manifest.json" "$AL_CFG_2T" \
                "$SPLIT_2/combined.train_ids.json" 12 0 0 --train.skip_final_save_on_max_steps true
            rc=$?
            ev="$E_OUT/auto_learning_events.jsonl"
            rs=$(grep -o '"system/replay_slots[^,]*' "$LOGS/E1.log" 2>/dev/null | tail -1)
            ru=$(grep -o '"system/replay_unique_tasks[^,]*' "$LOGS/E1.log" 2>/dev/null | tail -1)
            [ $rc -eq 0 ] && [ -f "$ev" ] \
                && _record E PASS "rc=0；$rs；$ru" \
                || _record E FAIL "rc=$rc（见 logs/E1.log）"
        fi
    fi
fi

# =============================================================================
# D — train → eval → train（同一个 unit 序列里相邻 unit 之间发生了评测）
if _want D; then
    _c "D — train → eval → train"
    EV=${D_EVENTS:-/data/outputs/al_reg_F/auto_learning_events.jsonl}
    if [ "$DRY_RUN" = "1" ]; then
        echo "(dry-run) 断言 $EV 里 train_unit 行 ≥2 且相邻两行的 task 切换后仍有 eval 记录"
    elif [ ! -f "$EV" ]; then
        _record D SKIP "没有 $EV（先跑 F）"
    else
        n=$("$PY" - "$EV" <<'PYEOF'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1],encoding='utf-8') if l.strip()]
tu=[r for r in rows if r.get('kind','event')=='event' and (r.get('action')=='train_unit')]
print(len(tu))
PYEOF
)
        [ "${n:-0}" -ge 2 ] && _record D PASS "events 里有 $n 个 train_unit（评测在 unit 之间发生）" \
                            || _record D FAIL "train_unit 行只有 ${n:-0} 个"
    fi
fi

# =============================================================================
# G — boundary checkpoint → restart → resume **语义等价**
#   (i)  连续 100 步
#   (ii) 50 步 → 存档 → 重启 RESUME=1 → 到 100
#   比较：最终 scheduler 记账（events 末行）+ resume 后前 5 步的 loss
if _want G; then
    _c "G — boundary resume 语义等价（continuous 100 vs 50+resume→100）"
    G1=/data/outputs/al_reg_G_cont
    G2=/data/outputs/al_reg_G_resume
    if [ "$DRY_RUN" = "1" ]; then
        al_train G1 "$G1" "$SPLIT_1/manifest.json" "$AL_CFG_G10" \
            "$SPLIT_1/click_bell.train_ids.json" 100 50 0 --train.skip_final_save_on_max_steps true
        al_train G2a "$G2" "$SPLIT_1/manifest.json" "$AL_CFG_G10" \
            "$SPLIT_1/click_bell.train_ids.json" 50 50 0 --train.skip_final_save_on_max_steps true
        al_train G2b "$G2" "$SPLIT_1/manifest.json" "$AL_CFG_G10" \
            "$SPLIT_1/click_bell.train_ids.json" 100 0 1 --train.skip_final_save_on_max_steps true
    else
        rm -rf "$G1" "$G2"
        al_train G1 "$G1" "$SPLIT_1/manifest.json" "$AL_CFG_G10" \
            "$SPLIT_1/click_bell.train_ids.json" 100 50 0 --train.skip_final_save_on_max_steps true
        rca=$?
        al_train G2a "$G2" "$SPLIT_1/manifest.json" "$AL_CFG_G10" \
            "$SPLIT_1/click_bell.train_ids.json" 50 50 0 --train.skip_final_save_on_max_steps true
        rcb=$?
        al_train G2b "$G2" "$SPLIT_1/manifest.json" "$AL_CFG_G10" \
            "$SPLIT_1/click_bell.train_ids.json" 100 0 1 --train.skip_final_save_on_max_steps true
        rcc=$?
        "$PY" - "$G1/auto_learning_events.jsonl" "$G2/auto_learning_events.jsonl" \
                "$LOGS/G1.log" "$LOGS/G2b.log" "$REPORTS/G_compare.txt" <<'PYEOF'
import json, re, sys
evA, evB, logA, logB, out = sys.argv[1:6]
def rows(p):
    try:
        return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
    except Exception:
        return []
a = [r for r in rows(evA) if r.get('action') == 'train_unit']
b = [r for r in rows(evB) if r.get('action') == 'train_unit']
keys = ('task', 'step', 'attempt', 'attempt_step', 'train_nmse', 'val_nmse', 'decision', 'samples_seen')
def sig(rs):
    return [tuple(r.get(k) for k in keys) for r in rs]
def last_steps(p, n=5):
    try:
        txt = open(p, encoding='utf-8', errors='ignore').read()
    except Exception:
        return []
    return re.findall(r'Step \d+/\d+.*?Loss ([0-9.]+).*?GradNorm ([0-9.]+)', txt)[-n:]
la, lb = last_steps(logA), last_steps(logB)
lines = []
lines.append(f'A(continuous) units={len(a)}  B(50+resume) units={len(b)}')
lines.append('A sig: ' + repr(sig(a)))
lines.append('B sig: ' + repr(sig(b)))
lines.append('scheduler 记账一致: ' + str(sig(a) == sig(b)))
lines.append(f'A 末 {len(la)} 步 (loss,grad): {la}')
lines.append(f'B 末 {len(lb)} 步 (loss,grad): {lb}')
same = sig(a) == sig(b)
if la and lb:
    same = same and all(abs(float(x[0]) - float(y[0])) < 1e-4 and
                        abs(float(x[1]) - float(y[1])) < 1e-3
                        for x, y in zip(la, lb))
lines.append('PASS(记账一致 且 末步 loss/grad 一致): ' + str(same))
open(out, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
sys.exit(0 if same else 1)
PYEOF
        rcg=$?
        [ $rca -eq 0 ] && [ $rcb -eq 0 ] && [ $rcc -eq 0 ] && [ $rcg -eq 0 ] \
            && _record G PASS "continuous ≡ boundary-resume（见 reports/G_compare.txt）" \
            || _record G FAIL "rc=$rca/$rcb/$rcc compare=$rcg（见 reports/G_compare.txt）"
    fi
fi

# =============================================================================
_c "汇总"
cat "$SUMMARY"
echo
echo "产物: $OUT"
echo "⚠️ 本脚本不自动关机。全部 PASS 后可: sudo shutdown -h now"
