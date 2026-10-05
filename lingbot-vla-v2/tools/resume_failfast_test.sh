#!/usr/bin/env bash
# =============================================================================
# tools/resume_failfast_test.sh —— resume **失败必须报错退出**（需要 GPU，约 4 分钟）
# -----------------------------------------------------------------------------
# 为什么需要它
# ------------
# `resume_e2e_test.sh` 只验了**成功**路径（ckpt 完好 ⇒ 能接上）。
# 但真正危险的是**失败**路径：原代码在「找到候选却全加载失败」时只打一行 info
# 然后**静默地从零重跑** —— 白烧算力、覆盖存档，而且日志里跟成功长得一模一样。
# 已改成 `raise RuntimeError`（tasks/vla/train_lingbotvla.py:764）。
# 本脚本就是来钉住这个行为的。
#
# 做法（便宜：不训练、不存档，模型加载完就报错）
#   R1' 把已有的一份完好 ckpt **mv**（同盘瞬时）到独立 OUT
#   故意把 `model/.metadata` 改名 ⇒ DCP 必加载失败
#   R3  RESUME=1 再跑 ⇒ 必须**报错退出**
#
# 断言：
#   a) rc != 0（必须失败）
#   b) 日志含 `resume 失败：找到`（新的 RuntimeError 文案）
#   c) 日志**不含** `Starting training from scratch`（没有静默重跑）
#   d) 日志**不含** `Step 1/`（没有偷偷开始训练）
#
# 用法：bash tools/resume_failfast_test.sh [源 ckpt 的 output_dir]
#       默认源 = /data/outputs/smoke/resume_e2e
# =============================================================================
set -uo pipefail

REPO=/data/code/lingbot-vla-v2
SRC_OUT=${1:-/data/outputs/smoke/resume_e2e}
OUT=/data/outputs/smoke/resume_fail
LOG="$OUT.log"
IDS=/data/outputs/smoke/ids
TASK=${TASK:-click_bell}

cd "$REPO"

pass=0; fail=0
ok() { echo "  [OK]   $1"; pass=$((pass + 1)); }
no() { echo "  [FAIL] $1"; fail=$((fail + 1)); }

echo "========================================================================"
echo "  resume 失败必须报错（不静默重跑）   SRC=$SRC_OUT"
echo "========================================================================"

# 找一份带 DCP 的 ckpt
STEP_DIR=$(ls -d "$SRC_OUT"/checkpoints/global_step_* 2>/dev/null | sort -V | tail -1)
if [ -z "$STEP_DIR" ] || [ ! -f "$STEP_DIR/model/.metadata" ]; then
    echo "❌ $SRC_OUT 下没有带 DCP 的 checkpoint（先跑 tools/resume_e2e_test.sh）" >&2
    exit 2
fi
echo "[setup] 用 $STEP_DIR"

rm -rf "$OUT"
mkdir -p "$OUT"
mv "$SRC_OUT/checkpoints" "$OUT/checkpoints"          # 同盘 mv，瞬时
STEP_NAME=$(basename "$STEP_DIR")

# 故意破坏：把 DCP 的元数据改名 ⇒ Checkpointer.load 必失败
mv "$OUT/checkpoints/$STEP_NAME/model/.metadata" \
   "$OUT/checkpoints/$STEP_NAME/model/.metadata.CORRUPTED"
echo "[setup] 已把 $STEP_NAME/model/.metadata 改名 ⇒ 这份 ckpt 必然加载失败"

mkdir -p "$IDS"
[ -f "$IDS/train50.json" ] || printf '[50]' > "$IDS/train50.json"

echo
echo "[R3] RESUME=1（ckpt 已损坏）⇒ 期望**报错退出**，日志 $LOG"
TASK="$TASK" MICRO=10 GAS=1 MAX_STEPS=4 SAVE_EVERY=0 SKIP_FINAL_SAVE=1 \
OPEN_LOOP_EVAL_STEPS=0 PRUNE=1 PRUNE_MIN_AGE=60 RESUME=1 TRAIN_OUT="$OUT" \
bash experiment/robotwin/single_task_train.sh > "$LOG" 2>&1
RC=$?
echo "[R3] 退出码 rc=$RC"

# a) 必须失败
if [ "$RC" -ne 0 ]; then
    ok "a) rc=$RC ≠ 0 ⇒ 确实报错退出了"
else
    no "a) rc=0 ⇒ 竟然"成功"了，说明没报错（静默重跑？）"
fi

# b) 有新的错误文案
if grep -qF 'resume 失败：找到' "$LOG"; then
    ok "b) 日志含「resume 失败：找到 …」⇒ 新的 fail-fast 生效"
else
    no "b) 没看到 fail-fast 的错误文案"
fi

# c) 没有静默重跑
if grep -qF 'Starting training from scratch' "$LOG"; then
    no "c) 出现了「Starting training from scratch」⇒ 仍在静默重跑"
else
    ok "c) 没有「Starting training from scratch」"
fi

# d) 没有偷偷开始训练
if grep -qE 'Step 1/' "$LOG"; then
    no "d) 出现 Step 1/ ⇒ 偷偷开始训练了"
else
    ok "d) 没有 Step 1/ ⇒ 没有偷偷开训"
fi

# 复原（让源 ckpt 可继续复用）
mv "$OUT/checkpoints/$STEP_NAME/model/.metadata.CORRUPTED" \
   "$OUT/checkpoints/$STEP_NAME/model/.metadata"
mv "$OUT/checkpoints" "$SRC_OUT/checkpoints"
echo "[cleanup] 已复原 $SRC_OUT/checkpoints"

echo
echo "========================================================================"
echo "  $pass 通过 / $fail 失败"
if [ "$fail" -eq 0 ]; then
    echo "  ✅ resume 失败会**报错退出**，不会静默从零重跑"
else
    echo "  ❌ 仍存在静默重跑风险 —— 不要依赖 RESUME=1"
    echo "     日志尾部："
    tail -20 "$LOG" | sed 's/^/       /'
fi
echo "========================================================================"
[ "$fail" -eq 0 ]
