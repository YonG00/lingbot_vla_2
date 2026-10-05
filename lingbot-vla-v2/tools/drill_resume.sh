#!/usr/bin/env bash
# =============================================================================
# 演练②：save·resume 完整性 —— 从已有 ckpt 续训，验证真的「接上」而不是「从零重跑」
#
# 为什么需要：resume 失败时训练**不报错**，而是打一行日志后**静默从零开始**，
#   而且会往同一 output_dir 再存一遍同名 ckpt。日志里「训练正常启动」与成功时长得一样，
#   所以**唯一能区分两者的信号是步数**（以及 `Load distributed checkpoint ... successfully!`）。
#
# 断言（6 条）：
#   1. 日志有 `Load distributed checkpoint from <...>/global_step_N successfully!`
#   2. **不出现 `Step 1/`** ⇒ 不是从零重跑（核心）
#   3. 不出现 `Starting training from scratch`
#   4. 无 `FileNotFoundError`
#   5. 续训后原 ckpt 仍在（守卫/无看门狗时不该被剪）
#   6. 退出码 0
#
# 用法：
#   CKPT_ROOT=/data/outputs/drill/stop_and_save bash tools/drill_resume.sh
#   默认 CKPT_ROOT=/data/outputs/drill/stop_and_save、MIXED=false MICRO=1（适配 48G 卡）
#   ⚠️ 默认 SKIP_FINAL_SAVE=1 ⇒ **不会**再存一份 48G，纯验证 resume
# =============================================================================
set -uo pipefail

REPO=/data/code/lingbot-vla-v2
TASK=${TASK:-click_bell}
CKPT_ROOT=${CKPT_ROOT:-/data/outputs/drill/stop_and_save}
EXTRA_STEPS=${EXTRA_STEPS:-3}          # 从 ckpt 那步再往前跑几步
MIXED=${MIXED:-false}
MICRO=${MICRO:-1}
GAS=${GAS:-1}
SKIP_FINAL_SAVE=${SKIP_FINAL_SAVE:-1}  # 1 = 不再存一份（省 48G）
LOG=${LOG:-/data/outputs/drill/drill_resume.log}

PASS=0; FAIL=0
ok()  { echo "  [OK]   $1"; PASS=$((PASS + 1)); }
bad() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }
hr()  { printf '%.0s─' {1..78}; echo; }

cd "$REPO" || { echo "❌ 仓库目录不存在: $REPO"; exit 1; }

# ---- 找已有的最大 ckpt ------------------------------------------------------
CK=$(ls -d "$CKPT_ROOT"/checkpoints/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
[ -n "$CK" ] || { echo "❌ $CKPT_ROOT/checkpoints 下没有 ckpt"; exit 1; }
SAVED=$(basename "$CK" | sed 's/global_step_//')
MAX_STEPS=$((SAVED + EXTRA_STEPS))
[ -f "$CK/hf_ckpt/model.safetensors.index.json" ] || { echo "❌ $CK hf_ckpt 不完整"; exit 1; }

rm -f "$LOG"
hr; echo "① 从 global_step_$SAVED 续训到 step $MAX_STEPS"; hr
echo "  ckpt-root : $CKPT_ROOT"
echo "  ckpt      : $CK"
echo "  配置      : RESUME=1 MAX_STEPS=$MAX_STEPS SKIP_FINAL_SAVE=$SKIP_FINAL_SAVE MIXED=$MIXED MICRO=$MICRO"
echo "  日志      : $LOG"

setsid nohup env TASK="$TASK" MICRO="$MICRO" GAS="$GAS" MAX_STEPS="$MAX_STEPS" \
    SAVE_EVERY=0 SKIP_FINAL_SAVE="$SKIP_FINAL_SAVE" PRUNE=0 RESUME=1 MIXED="$MIXED" \
    TRAIN_OUT="$CKPT_ROOT" \
    bash experiment/robotwin/single_task_train.sh > "$LOG" 2>&1 < /dev/null &
LAUNCH_PID=$!

hr; echo "② 等训练结束"; hr
t0=$(date +%s)
for _ in $(seq 1 300); do
    ps -p "$LAUNCH_PID" >/dev/null 2>&1 || break
    pgrep -f '[t]rain_lingbotvla' >/dev/null 2>&1 || break
    sleep 5
done
sleep 5
wait "$LAUNCH_PID" 2>/dev/null
RC=$?
echo "  用时 $(( $(date +%s) - t0 ))s，rc=$RC"

hr; echo "③ 断言"; hr
grep -qF "Load distributed checkpoint from" "$LOG" \
    && ok "日志有「Load distributed checkpoint from ... successfully!」" \
    || bad "日志缺「Load distributed checkpoint from」⇒ 根本没恢复"

# 🔴 核心：resume 失败时会静默从零跑，唯一信号就是「出现 Step 1/」
if grep -qE 'Step: 1/' "$LOG"; then
    bad "出现「Step: 1/」⇒ 从零重跑了，resume 没生效"
else
    ok "未出现「Step: 1/」⇒ 不是从零重跑"
fi
grep -qF 'Starting training from scratch' "$LOG" \
    && bad "出现「Starting training from scratch」⇒ 静默降级了" \
    || ok "未出现「Starting training from scratch」"
grep -qF 'FileNotFoundError' "$LOG" && bad "出现 FileNotFoundError（ckpt 被剪？）" || ok "无 FileNotFoundError"
[ -d "$CK" ] && ok "续训后原 ckpt 仍在" || bad "原 ckpt 不见了"
[ "$RC" = "0" ] && ok "退出码 0" || bad "退出码 $RC"

hr
echo "  实际跑到的步数："
grep -aoE 'Step: [0-9]+/[0-9]+' "$LOG" | tail -3 | sed 's/^/    /'
hr
echo "  $PASS 通过 / $FAIL 失败"
if [ "$FAIL" -eq 0 ]; then echo "  ✅ save·resume 完整性演练通过"; else echo "  ❌ 有失败项；日志：$LOG"; fi
exit "$([ "$FAIL" -eq 0 ] && echo 0 || echo 1)"
