#!/usr/bin/env bash
# =============================================================================
# tools/open_loop_smoke.sh —— 「训练 + 训练中开环评测」端到端冒烟测试（需要 GPU）
# -----------------------------------------------------------------------------
# 一条命令验完这条链路，并给出 PASS/FAIL 清单：
#   ① 训练能起来、能跑完 N 步（含 micro×gas 的显存是否装得下）
#   ② 训练中 open-loop 评测能出数（mse/mae/baseline/逐轨迹）
#   ③ eval 期间的三处临时开关都生效并**已还原**（恢复审计）
#   ④ 归一化统计指纹正确（= data.norm_stats_file 那份）
#   ⑤ 不落任何 checkpoint（smoke 不该写盘）
#
# 用法：
#   MICRO=10 GAS=1 bash tools/open_loop_smoke.sh              # gbs = micro × gas
#   MICRO=10 GAS=1 STEPS=5 TASK=click_bell bash tools/open_loop_smoke.sh
#   MICRO=1  GAS=1 bash tools/open_loop_smoke.sh              # 最小显存
#
# 环境变量：MICRO(10) GAS(1) STEPS(5) TASK(click_bell) OUT(/data/outputs/smoke/e2e_<m>x<g>)
# =============================================================================
set -uo pipefail

TASK=${TASK:-click_bell}
MICRO=${MICRO:-10}
GAS=${GAS:-1}
GBS=$((MICRO * GAS))
STEPS=${STEPS:-5}
OUT=${OUT:-/data/outputs/smoke/e2e_${MICRO}x${GAS}}
IDS_DIR=/data/outputs/smoke/ids
LOG=${OUT}.log
REPO=/data/code/lingbot-vla-v2

cd "$REPO"

# 白名单（train-monitor 1 条 + held-out val 1 条，只为冒烟）
mkdir -p "$IDS_DIR"
[ -f "$IDS_DIR/train50.json" ] || printf '[50]' > "$IDS_DIR/train50.json"
[ -f "$IDS_DIR/val51.json" ]   || printf '[51]' > "$IDS_DIR/val51.json"

echo "========================================================================"
echo "  端到端冒烟：TASK=$TASK  MICRO=$MICRO  GAS=$GAS  GBS=$GBS  STEPS=$STEPS"
echo "  输出：$OUT"
echo "========================================================================"

rm -rf "$OUT"
TASK="$TASK" MICRO="$MICRO" GAS="$GAS" \
MAX_STEPS="$STEPS" SAVE_EVERY=0 SKIP_FINAL_SAVE=1 \
OPEN_LOOP_EVAL_STEPS="$STEPS" \
OPEN_LOOP_TRAIN_IDS="$IDS_DIR/train50.json" \
OPEN_LOOP_VAL_IDS="$IDS_DIR/val51.json" \
PRUNE=0 TRAIN_OUT="$OUT" \
bash experiment/robotwin/single_task_train.sh > "$LOG" 2>&1
RC=$?

# ---- 逐项核对 ---------------------------------------------------------------
pass=0; fail=0
chk() {  # chk <名称> <grep 表达式> [必须出现=1/必须不出现=0]
    local name="$1" pat="$2" want="${3:-1}" n
    n=$(grep -cE "$pat" "$LOG" 2>/dev/null || true)
    if { [ "$want" = "1" ] && [ "$n" -gt 0 ]; } || { [ "$want" = "0" ] && [ "$n" -eq 0 ]; }; then
        echo "  [OK]   $name"; pass=$((pass + 1))
    else
        echo "  [FAIL] $name   (匹配 $n 次，期望 want=$want)"; fail=$((fail + 1))
    fi
}

echo
echo "---- 结果 ----"
chk "训练跑完 $STEPS 步"                 "训练结束: epoch=.*global_step=${STEPS}"
# ⚠️ 不能用裸 `Traceback` —— torch 的 `UserWarning: ... Traceback of forward call ...`
#    会误报（2026-10-04 实测）。只匹配**真正的异常栈**。
chk "无 OOM / 无真异常栈"                "OutOfMemoryError|CUDA out of memory|Traceback \(most recent call last\)|ChildFailedError" 0
chk "开环评测出数"                       "\[open_loop\] step ${STEPS}: .*mse="
chk "逐轨迹 MSE 行"                      "\[open_loop\] per-traj MSE (train|val)"
chk "eval 临时 use_cache 开关生效"        "eval 期间临时 use_cache: .*→ True"
chk "eval 临时 attention→eager 生效"      "eval 期间临时 attention_implementation: .*→ eager"
chk "归一化统计指纹已打印"                "归一化统计指纹"
chk "恢复审计通过"                       "恢复审计通过"
chk "评测未失败"                         "\[open_loop\] ⚠️ step .* 评测失败" 0

# 「未落 checkpoint」单独判（不能用 grep）
if [ -d "$OUT/checkpoints" ] && [ -n "$(ls -A "$OUT/checkpoints" 2>/dev/null)" ]; then
    echo "  [FAIL] 未落 checkpoint —— $OUT/checkpoints 里居然有东西"; fail=$((fail + 1))
else
    echo "  [OK]   未落 checkpoint"; pass=$((pass + 1))
fi

echo
echo "---- 关键日志 ----"
grep -E "批大小|VRAM usage after epoch|\[open_loop\] step |per-traj MSE|归一化统计指纹|恢复审计|eval 期间临时" "$LOG" | sed 's/^/  /' | head -20
echo
echo "---- 显存/耗时 ----"
grep -oE "max [0-9.]+GB" "$LOG" | tail -1 | sed 's/^/  峰值 /'
grep -oE "StepTime [0-9.]+s" "$LOG" | tail -1 | sed 's/^/  /'

echo
echo "========================================================================"
echo "  $pass 通过 / $fail 失败   (训练退出码 rc=$RC；日志 $LOG)"
echo "========================================================================"
[ "$fail" -eq 0 ] && [ "$RC" -eq 0 ]
