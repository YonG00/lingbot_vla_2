#!/usr/bin/env bash
# =============================================================================
# tools/resume_e2e_test.sh —— **resume 端到端测试**（需要 GPU，约 15 分钟）
# -----------------------------------------------------------------------------
# `resume` 这条路径在本项目**从未被验证过**（待办⑥）。本测试真跑两次训练来钉住它：
#
#   R1  从零跑 2 步、每 2 步存档            ⇒ 得到 checkpoints/global_step_2（含 DCP）
#   R2  RESUME=1 跑到 4 步（不存档、不评测） ⇒ 必须**从 step 2 接上**，而不是从 0 重来
#
# 断言（全过才算 resume 有效）：
#   a) R2 日志里有 `Load distributed checkpoint from <OUT>/checkpoints/global_step_2 successfully!`
#   b) R2 里**不出现** `Step 1/`（出现即说明从 0 重来了）
#   c) R2 里没有 `FileNotFoundError`（就是 2026-10-05 那个坑）
#   d) R2 结束后 global_step_2 的 DCP 仍在（守卫没剪掉要恢复的那份）
#   e) R1 / R2 退出码都是 0
#
# ⚠️ 需要 GPU（要真的加载 6B 模型）。
# 用法：bash tools/resume_e2e_test.sh
# =============================================================================
set -uo pipefail

REPO=/data/code/lingbot-vla-v2
TASK=${TASK:-click_bell}
OUT=${OUT:-/data/outputs/smoke/resume_e2e}
LOG1="$OUT.r1.log"
LOG2="$OUT.r2.log"
IDS=/data/outputs/smoke/ids

cd "$REPO"

mkdir -p "$IDS"
[ -f "$IDS/train50.json" ] || printf '[50]' > "$IDS/train50.json"
[ -f "$IDS/val51.json" ]   || printf '[51]' > "$IDS/val51.json"

pass=0; fail=0
ok() { echo "  [OK]   $1"; pass=$((pass + 1)); }
no() { echo "  [FAIL] $1"; fail=$((fail + 1)); }

echo "========================================================================"
echo "  resume 端到端测试   TASK=$TASK   OUT=$OUT"
echo "========================================================================"

# ---------------- R1：从零跑 2 步、每 2 步存档 ----------------
rm -rf "$OUT"
echo
echo "[R1] 从零跑 2 步、每 2 步存档（MAX_STEPS=2 SAVE_EVERY=2，不跑开环评测）"
echo "     日志 $LOG1"
TASK="$TASK" MICRO=10 GAS=1 MAX_STEPS=2 SAVE_EVERY=2 OPEN_LOOP_EVAL_STEPS=0 \
PRUNE=1 PRUNE_MIN_AGE=60 TRAIN_OUT="$OUT" \
bash experiment/robotwin/single_task_train.sh > "$LOG1" 2>&1
RC1=$?
echo "[R1] 退出码 rc=$RC1"

if [ -f "$OUT/checkpoints/global_step_2/model/.metadata" ]; then
    ok "R1 产出 global_step_2 的 DCP（model/.metadata 在）"
else
    no "R1 没产出可续训的 DCP —— 后面没法测"
fi
[ "$RC1" -eq 0 ] && ok "R1 退出码 0" || no "R1 退出码 $RC1"

# ---------------- R2：RESUME=1 跑到 4 步 ----------------
echo
echo "[R2] RESUME=1 跑到 4 步（MAX_STEPS=4 SAVE_EVERY=0 SKIP_FINAL_SAVE=1，不存档不评测）"
echo "     日志 $LOG2"
TASK="$TASK" MICRO=10 GAS=1 MAX_STEPS=4 SAVE_EVERY=0 SKIP_FINAL_SAVE=1 \
OPEN_LOOP_EVAL_STEPS=0 PRUNE=1 PRUNE_MIN_AGE=60 RESUME=1 TRAIN_OUT="$OUT" \
bash experiment/robotwin/single_task_train.sh > "$LOG2" 2>&1
RC2=$?
echo "[R2] 退出码 rc=$RC2"

# a) 恢复了正确的那份
if grep -qF "Load distributed checkpoint from $OUT/checkpoints/global_step_2 successfully" "$LOG2"; then
    ok "a) 从 global_step_2 恢复成功（日志有 Load distributed checkpoint … successfully）"
else
    no "a) 没看到「从 global_step_2 恢复成功」"
fi

# b) 没有从 0 重来
if grep -qE 'Step 1/' "$LOG2"; then
    no "b) 出现了 Step 1/ ⇒ 从 0 重来了，resume 没生效"
else
    ok "b) 没有 Step 1/ ⇒ 不是从 0 重来"
fi

# c) 没有那个 FileNotFoundError
if grep -qF 'FileNotFoundError' "$LOG2"; then
    no "c) 出现 FileNotFoundError（ckpt 被剪或路径不对）"
else
    ok "c) 无 FileNotFoundError"
fi

# d) 守卫保住了要恢复的那份 DCP
if [ -f "$OUT/checkpoints/global_step_2/model/.metadata" ]; then
    ok "d) R2 结束后 global_step_2 的 DCP 仍在（守卫生效）"
else
    no "d) global_step_2 的 DCP 被剪了 ⇒ 守卫失效"
fi

[ "$RC2" -eq 0 ] && ok "e) R2 退出码 0" || no "e) R2 退出码 $RC2"

echo
echo "========================================================================"
echo "  $pass 通过 / $fail 失败"
if [ "$fail" -eq 0 ]; then
    echo "  ✅ resume 有效：能正确从已有 ckpt 接上，且不会被看门狗剪掉"
else
    echo "  ❌ resume 有问题 —— 不要依赖 RESUME=1"
    echo "     R2 日志尾部："
    tail -15 "$LOG2" | sed 's/^/       /'
fi
echo "========================================================================"
[ "$fail" -eq 0 ]
