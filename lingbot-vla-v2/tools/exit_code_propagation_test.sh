#!/usr/bin/env bash
# =============================================================================
# 退出码传递回归测试（纯本地、无 GPU、秒级）
#
# 背景（2026-10-05 实测踩坑）：
#   train.sh 里是 `torchrun ... | tee log.txt` —— 管道之后 `$?` 是 **tee** 的（永远 0）
#   ⇒ 训练崩溃被上报成「成功」，调用方和自动化全部误判。
#   single_task_train.sh 同理：`bash train.sh` 之后还有 echo/banner，脚本会返回最后一个
#   echo 的 0 ⇒ 即使 train.sh 已经正确返回非 0，也会被吞掉。
#
# 本测试用**桩程序**（假 torchrun / 假 nvidia-smi）钉住这两条，不需要模型、不需要 GPU。
#
# 用法：bash tools/exit_code_propagation_test.sh
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SINGLE="$REPO/experiment/robotwin/single_task_train.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PASS=0
FAIL=0
ok()  { echo "  [OK]   $1"; PASS=$((PASS + 1)); }
bad() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }

echo "== 退出码传递回归测试 =="
echo "   repo = $REPO"

# ---------------------------------------------------------------------------
# 桩：torchrun 按 $FAKE_RC 退出；nvidia-smi 报 1 张卡（让 NPROC_PER_NODE=1）
# ---------------------------------------------------------------------------
mkdir -p "$WORK/bin"
cat > "$WORK/bin/torchrun" <<'STUB'
#!/usr/bin/env bash
echo "[stub torchrun] argv: $*"
echo "[stub torchrun] 模拟训练日志"
exit "${FAKE_RC:-0}"
STUB
cat > "$WORK/bin/nvidia-smi" <<'STUB'
#!/usr/bin/env bash
echo "GPU 0: Fake GPU (stub)"
STUB
chmod +x "$WORK/bin/torchrun" "$WORK/bin/nvidia-smi"

# ---------------------------------------------------------------------------
# T1/T2 语法检查（能抓到「引号不配对」这类会直接改坏脚本的编辑事故）
# ---------------------------------------------------------------------------
for f in "$REPO/train.sh" "$SINGLE"; do
    if bash -n "$f" 2>"$WORK/syn.err"; then
        ok "语法检查 $(basename "$f")"
    else
        bad "语法检查 $(basename "$f")：$(head -2 "$WORK/syn.err" | tr '\n' ' ')"
    fi
done

# ---------------------------------------------------------------------------
# T3/T4 train.sh 的退出码传递（核心）
# ---------------------------------------------------------------------------
for rc in 0 7; do
    (cd "$WORK" && PATH="$WORK/bin:$PATH" FAKE_RC="$rc" bash "$REPO/train.sh" fake.py \
        >/dev/null 2>&1)
    got=$?
    if [ "$got" = "$rc" ]; then
        ok "train.sh 退出码传递：torchrun rc=$rc ⇒ 脚本 rc=$got"
    else
        bad "train.sh 退出码传递：torchrun rc=$rc ⇒ 脚本 rc=${got}（应为 ${rc}，说明管道吞了）"
    fi
done

# ---------------------------------------------------------------------------
# T5/T6 single_task_train.sh 的结构性检查（跑全流程太重，改做静态断言）
# ---------------------------------------------------------------------------
if grep -qE '^TRAIN_RC=\$\?' "$SINGLE"; then
    # 必须紧跟 `bash train.sh ...` 那条命令之后（中间只允许续行/注释/空行）
    line_invoke=$(grep -n '^bash train\.sh' "$SINGLE" | head -1 | cut -d: -f1)
    line_catch=$(grep -n '^TRAIN_RC=\$\?' "$SINGLE" | head -1 | cut -d: -f1)
    if [ -n "$line_invoke" ] && [ "$line_catch" -gt "$line_invoke" ] && \
       [ $((line_catch - line_invoke)) -lt 40 ]; then
        ok "single_task_train.sh 在 train.sh 之后立刻接住 rc（第 $line_invoke → $line_catch 行）"
    else
        bad "single_task_train.sh 的 TRAIN_RC 位置不对（invoke=$line_invoke catch=${line_catch}）"
    fi
else
    bad "single_task_train.sh 没有 TRAIN_RC=\$?（训练失败会被后面的 echo 吞掉）"
fi

if tail -3 "$SINGLE" | grep -qE '^exit "\$\{TRAIN_RC:-0\}"$'; then
    ok "single_task_train.sh 以 exit \"\${TRAIN_RC:-0}\" 收尾"
else
    bad "single_task_train.sh 结尾没有把 TRAIN_RC 传出去（$(tail -1 "$SINGLE")）"
fi

# ---------------------------------------------------------------------------
# T7 端到端：single_task_train.sh → train.sh → torchrun 的 rc 必须一路传出
#
# 技巧：`single_task_train.sh` 会 `export PATH=<conda>/bin:$PATH`，所以把桩放 PATH 前面
# 会被它盖掉。改用 **BASH_ENV** —— 非交互 bash 启动时会 source 它，在里面定义
# `torchrun()` **函数**；函数优先于 PATH 查找 ⇒ 不用污染 conda bin，也不用改脚本。
# 需要远端环境（conda python + 划分文件）；本地/无卡缺件时自动 SKIP。
# ---------------------------------------------------------------------------
PY_BIN=/data/miniconda3/envs/lingbotvla/bin/python
SPLIT_DEFAULT=/data/train/task_splits
if [ -x "$PY_BIN" ] && [ -f "$SPLIT_DEFAULT/manifest.json" ]; then
    cat > "$WORK/stub_env.sh" <<'STUB'
torchrun() { echo "[stub torchrun] argv: $*"; return "${FAKE_RC:-0}"; }
STUB
    had_log=0
    if [ -f "$REPO/log.txt" ]; then had_log=1; cp -f "$REPO/log.txt" "$WORK/log.txt.bak"; fi
    for rc in 0 7; do
        (cd "$WORK" && BASH_ENV="$WORK/stub_env.sh" FAKE_RC="$rc" \
            TASK=click_bell PRUNE=0 MAX_STEPS=4 SAVE_EVERY=0 \
            TRAIN_OUT="$WORK/out_rc$rc" \
            bash "$SINGLE" >"$WORK/e2e_rc$rc.log" 2>&1)
        got=$?
        if [ "$got" = "$rc" ]; then
            ok "端到端 single_task_train.sh：torchrun rc=$rc ⇒ 脚本 rc=${got}"
        else
            bad "端到端 single_task_train.sh：torchrun rc=$rc ⇒ 脚本 rc=${got}（应为 ${rc}；尾部：$(tail -1 "$WORK/e2e_rc$rc.log" | cut -c1-70)）"
        fi
    done
    if [ "$had_log" = "1" ]; then cp -f "$WORK/log.txt.bak" "$REPO/log.txt"; else rm -f "$REPO/log.txt"; fi
else
    echo "  [SKIP] 端到端 T7（需远端环境：$PY_BIN + $SPLIT_DEFAULT/manifest.json）"
fi

# ---------------------------------------------------------------------------
echo
echo "  $PASS 通过 / $FAIL 失败"
if [ "$FAIL" -eq 0 ]; then
    echo "  ✅ 退出码能正确传出（训练崩溃不会再被当成成功）"
else
    echo "  ❌ 有失败项：退出码会被吞，自动化会误判"
fi
exit "$([ "$FAIL" -eq 0 ] && echo 0 || echo 1)"
