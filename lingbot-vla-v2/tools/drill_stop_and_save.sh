#!/usr/bin/env bash
# =============================================================================
# 演练：STOP_AND_SAVE —— 外部 touch 文件 ⇒ 收尾存档并正常退出
#
# 机制（train_lingbotvla.py:1275-1293）：
#   每个 step 检查 `<output_dir>/STOP_AND_SAVE` 是否存在 → 读内容（**只记日志，不校验**）
#   → `os.replace` 改名为 `.done` → `stop_requested_by_file=True` → `reached_max_steps=True` → break
#   ⇒ 之后走与「max_steps 到顶」相同的收尾存档路径。
#   `skip_final_save_on_max_steps` 被 `not stop_requested_by_file` 守卫 ⇒ 不会误跳过本次存档。
#
# 断言（9 条，全过才算演练通过）：
#   1. 日志有 `[STOP_AND_SAVE] 检测到`
#   2. 日志有 `[STOP_AND_SAVE] 已改名为`
#   3. `<OUT>/STOP_AND_SAVE.done` 存在（原文件已被改名，防重复触发）
#   4. 训练**远早于** MAX_STEPS 就停了（证明是文件触发的，不是跑到头）
#   5. `checkpoints/` 下**只有一份** ckpt（SAVE_EVERY=0 ⇒ 无步存档，收尾只存一次）
#   6. 该 ckpt 的 `hf_ckpt/model.safetensors.index.json` 存在（存档完整）
#   7. `du -sh` 连续两次相同（写完盘、不再增长）
#   8. 退出码 = 0
#   9. 无 `OutOfMemoryError`
#
# 用法：
#   MIXED=false MICRO=1 GAS=1 STEPS=20 bash tools/drill_stop_and_save.sh
#   默认 MIXED=false MICRO=1（适配 48G 卡）
# =============================================================================
set -uo pipefail

REPO=/data/code/lingbot-vla-v2
TASK=${TASK:-click_bell}
STEPS=${STEPS:-20}
MIXED=${MIXED:-false}
MICRO=${MICRO:-1}
GAS=${GAS:-1}
OUT=${OUT:-/data/outputs/drill/stop_and_save}
TOUCH_AT=${TOUCH_AT:-2}          # 看到 `Step <TOUCH_AT>/` 就 touch
LOG=${LOG:-/data/outputs/drill/stop_and_save.log}

PASS=0; FAIL=0
ok()  { echo "  [OK]   $1"; PASS=$((PASS + 1)); }
bad() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }
hr()  { printf '%.0s─' {1..78}; echo; }

cd "$REPO" || { echo "❌ 仓库目录不存在: $REPO"; exit 1; }

rm -rf "$OUT" "$LOG"; mkdir -p "$(dirname "$LOG")" "$OUT"

hr; echo "① 启动训练（MAX_STEPS=$STEPS SAVE_EVERY=0 PRUNE=0 MIXED=$MIXED MICRO=$MICRO）"; hr
setsid nohup env TASK="$TASK" MICRO="$MICRO" GAS="$GAS" MAX_STEPS="$STEPS" \
    SAVE_EVERY=0 SKIP_FINAL_SAVE=0 PRUNE=0 MIXED="$MIXED" TRAIN_OUT="$OUT" \
    bash experiment/robotwin/single_task_train.sh > "$LOG" 2>&1 < /dev/null &
LAUNCH_PID=$!
echo "  已启动（pid $LAUNCH_PID），日志 $LOG"

# ---- 等模型加载 + 跑到 TOUCH_AT 步，然后 touch -------------------------------
hr; echo "② 等训练到 Step $TOUCH_AT 后 touch STOP_AND_SAVE"; hr
t0=$(date +%s)
early_exit=0
for _ in $(seq 1 240); do
    # ⚠️ 日志里**有两种**步数文本，别搞混：
    #   tqdm 行：  `Step: 2/20 [00:21<02:55, 9.73s/it]`   ← 分母 = MAX_STEPS，**"Step:" 带冒号**
    #   INFO 行：  `... - Step 2/3085, Epoch 1, Loss ...` ← 分母 = 每轮步数，无冒号
    #   ⇒ 只能匹配 tqdm 的 `Step: N/`（冒号是关键，漏了永远匹配不上）
    if grep -qE "Step: ${TOUCH_AT}/" "$LOG" 2>/dev/null; then break; fi
    if ! ps -p "$LAUNCH_PID" >/dev/null 2>&1 && ! pgrep -f '[t]rain_lingbotvla' >/dev/null 2>&1; then
        early_exit=1; echo "  ⚠️ 训练进程提前退出（touch 来不及），本次演练无效"; break
    fi
    sleep 5
done
if [ "$early_exit" = "1" ]; then
    echo "  ❌ 未能在训练结束前 touch ⇒ 无法验证 STOP_AND_SAVE（检查上面的匹配模式）"
    exit 1
fi
echo "  等待用时 $(( $(date +%s) - t0 ))s；touch $OUT/STOP_AND_SAVE"
date > "$OUT/STOP_AND_SAVE"
cat "$OUT/STOP_AND_SAVE"

# ---- 等训练结束 -------------------------------------------------------------
hr; echo "③ 等训练收尾（含存档，可能数分钟）"; hr
for _ in $(seq 1 360); do
    pgrep -f '[t]rain_lingbotvla' >/dev/null 2>&1 || break
    sleep 5
done
sleep 5
wait "$LAUNCH_PID" 2>/dev/null
RC=$?
echo "  训练已结束，rc=$RC"

# ---- 断言 -------------------------------------------------------------------
hr; echo "④ 断言"; hr
grep -qF '[STOP_AND_SAVE] 检测到' "$LOG" && ok "日志有「检测到」" || bad "日志缺「检测到」"
grep -qF '[STOP_AND_SAVE] 已改名为' "$LOG" && ok "日志有「已改名为」" || bad "日志缺「已改名为」"
[ -f "$OUT/STOP_AND_SAVE.done" ] && ok "STOP_AND_SAVE.done 存在" || bad "STOP_AND_SAVE.done 不存在"
[ ! -f "$OUT/STOP_AND_SAVE" ] && ok "原 STOP_AND_SAVE 已被改名（防重复触发）" || bad "原文件还在"

LAST=$(grep -oE 'Step [0-9]+/[0-9]+' "$LOG" | tail -1 | sed 's|Step ||; s|/.*||')
if [ -n "$LAST" ] && [ "$LAST" -lt "$STEPS" ]; then
    ok "训练在 Step $LAST 停止（< MAX_STEPS=$STEPS）⇒ 是文件触发的"
else
    bad "训练步数 = ${LAST:-未知}，未能在 MAX_STEPS 前停（可能是跑到头了）"
fi

N_CKPT=$(ls -d "$OUT"/checkpoints/global_step_* 2>/dev/null | wc -l | tr -d ' ')
[ "$N_CKPT" = "1" ] && ok "只存了 1 份 ckpt（无重复存档）" || bad "ckpt 份数 = $N_CKPT（应为 1）"

CK=$(ls -d "$OUT"/checkpoints/global_step_* 2>/dev/null | head -1)
[ -n "$CK" ] && [ -f "$CK/hf_ckpt/model.safetensors.index.json" ] \
    && ok "hf_ckpt 完整（index.json 在）" || bad "hf_ckpt 不完整"

S1=$(du -sb "$OUT" 2>/dev/null | cut -f1); sleep 20
S2=$(du -sb "$OUT" 2>/dev/null | cut -f1)
[ "$S1" = "$S2" ] && ok "存档体积稳定（$(numfmt --to=iec "$S1" 2>/dev/null || echo "$S1")B）" \
    || bad "体积仍在增长：$S1 → $S2（HF 异步存档可能没写完）"

[ "$RC" = "0" ] && ok "退出码 0" || bad "退出码 $RC"
grep -qE 'OutOfMemoryError|CUDA out of memory' "$LOG" && bad "出现 OOM" || ok "无 OOM"

hr
echo "  $PASS 通过 / $FAIL 失败"
if [ "$FAIL" -eq 0 ]; then
    echo "  ✅ STOP_AND_SAVE 演练通过"
else
    echo "  ❌ 有失败项；日志：$LOG"
fi
exit "$([ "$FAIL" -eq 0 ] && echo 0 || echo 1)"
