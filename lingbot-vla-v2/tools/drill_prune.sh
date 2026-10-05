#!/usr/bin/env bash
# =============================================================================
# 演练③：prune 看门狗 —— 剪掉真 ckpt 的 DCP、保住 hf_ckpt、回收磁盘
#
# 背景（10-05 真实踩坑）：训练中存档后 DCP 占 24G，若看门狗**剪得太慢**，
#   disk_guard 会先判定空间不足而**提前停训**（实测差 11 秒）。
#   修法：把静默期从 300s 调到 60s，并在 RESUME=1 时强制 `--keep-last 1`。
#
# 本演练做什么：
#   A) `--dry-run --once`  → 只打印计划，**不动任何文件**（先确认它会删什么）
#   B) `--once`（keep-last=0, min-age=0）→ 真剪，验证 DCP 消失、hf_ckpt 留下、磁盘回收
#
# 断言（7 条）：
#   1. dry-run 的计划里包含要删的 DCP（model / optimizer / extra_state）
#   2. dry-run 后 DCP **仍在**（dry-run 真的没动文件）
#   3. 真剪后 `model` / `optimizer` / `extra_state` 都不在了
#   4. `hf_ckpt/model.safetensors.index.json` 仍在（**评测/续训都要它**）
#   5. 磁盘回收 ≥ 20G
#   6. 剪后 hf_ckpt 体积不变（没被误伤）
#   7. prune 进程退出码 0
#
# 用法：
#   CKPT_ROOT=/data/outputs/drill/stop_and_save bash tools/drill_prune.sh
#
# ⚠️ 覆盖范围说明：本演练验证「看门狗能正确剪一个**真实** 48G ckpt」。
#    完整的「prune 与 disk_guard 的竞态」需要 20+ 分钟的真跑（中间存档 + 训练继续），
#    不在本脚本范围内；那部分的真实证据来自 10-05 探针（step 550 被 disk_guard 停）。
# =============================================================================
set -uo pipefail

REPO=/data/code/lingbot-vla-v2
CKPT_ROOT=${CKPT_ROOT:-/data/outputs/drill/stop_and_save}
PY=/data/miniconda3/envs/lingbotvla/bin/python

PASS=0; FAIL=0
ok()  { echo "  [OK]   $1"; PASS=$((PASS + 1)); }
bad() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }
hr()  { printf '%.0s─' {1..78}; echo; }
free_gb() { df -BG /data | tail -1 | awk '{gsub(/G/,"",$4); print $4}'; }

cd "$REPO" || { echo "❌ 仓库目录不存在: $REPO"; exit 1; }

CK=$(ls -d "$CKPT_ROOT"/checkpoints/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
[ -n "$CK" ] || { echo "❌ $CKPT_ROOT/checkpoints 下没有 ckpt"; exit 1; }
[ -d "$CK/model" ] || { echo "❌ $CK 没有 DCP（model/）—— 本演练需要一份**完整**存档"; exit 1; }

D0=$(free_gb); H0=$(du -sm "$CK/hf_ckpt" | cut -f1)
hr; echo "① 初始状态"; hr
echo "  ckpt      : $CK"
echo "  各部件    : $(du -sh "$CK"/* 2>/dev/null | tr '\n' ' ')"
echo "  可用磁盘  : ${D0}G"

# ---- A) dry-run（只跑一次，结果落文件再判读）--------------------------------
hr; echo "② dry-run（只打印计划）"; hr
DRYOUT=/tmp/drill_prune_dry.txt
"$PY" -u tools/prune_dcp.py --ckpt-root "$CKPT_ROOT" --keep-last 0 \
    --min-age-seconds 0 --once --dry-run > "$DRYOUT" 2>&1
DRY=$?
tail -12 "$DRYOUT" | sed 's/^/    /'
grep -qE 'model|optimizer|extra_state' "$DRYOUT" \
    && ok "dry-run 计划里包含要删的 DCP" || bad "dry-run 计划里没看到 DCP（可能被判为不完整）"
[ -d "$CK/model" ] && ok "dry-run 后 DCP 仍在（确实没动文件）" || bad "dry-run 竟然删了文件！"

# ---- B) 真剪 ----------------------------------------------------------------
hr; echo "③ 真剪（keep-last=0, min-age=0）"; hr
"$PY" -u tools/prune_dcp.py --ckpt-root "$CKPT_ROOT" --keep-last 0 \
    --min-age-seconds 0 --once 2>&1 | tail -10 | sed 's/^/    /'
PRC=${PIPESTATUS[0]}
echo "  prune rc=$PRC"

hr; echo "④ 断言"; hr
[ ! -d "$CK/model" ] && ok "model/ 已删" || bad "model/ 还在"
[ ! -d "$CK/optimizer" ] && ok "optimizer/ 已删" || bad "optimizer/ 还在"
[ ! -d "$CK/extra_state" ] && ok "extra_state/ 已删" || bad "extra_state/ 还在"
[ -f "$CK/hf_ckpt/model.safetensors.index.json" ] \
    && ok "hf_ckpt/index.json 仍在（评测/续训要用）" || bad "hf_ckpt 被误删！"

D1=$(free_gb); FREED=$((D1 - D0))
[ "$FREED" -ge 20 ] && ok "磁盘回收 ${FREED}G（≥20G）" || bad "只回收 ${FREED}G（应 ≥20G）"

H1=$(du -sm "$CK/hf_ckpt" | cut -f1)
[ "$H0" = "$H1" ] && ok "hf_ckpt 体积不变（${H1}MB）" || bad "hf_ckpt 体积变了：${H0} → ${H1} MB"
[ "$PRC" = "0" ] && ok "prune 退出码 0" || bad "prune 退出码 $PRC"

hr
echo "  剪后：$(du -sh "$CK"/* 2>/dev/null | tr '\n' ' ')"
echo "  磁盘：${D0}G → ${D1}G"
hr
echo "  $PASS 通过 / $FAIL 失败"
if [ "$FAIL" -eq 0 ]; then echo "  ✅ prune 看门狗演练通过"; else echo "  ❌ 有失败项"; fi
exit "$([ "$FAIL" -eq 0 ] && echo 0 || echo 1)"
