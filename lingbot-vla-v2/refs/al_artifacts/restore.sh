#!/usr/bin/env bash
# =============================================================================
# AL 派生产物恢复脚本（把 refs/al_artifacts 的内容放回训练机）
#
# 用法：
#   bash refs/al_artifacts/restore.sh --local                  # 已在本机（训练机上）
#   bash refs/al_artifacts/restore.sh <ssh目标> <端口> [密钥]    # 从开发机推过去
# 例：
#   bash refs/al_artifacts/restore.sh root@36.150.116.206 32763 .ssh/cpu1_ed25519
#
# 做三件事：① 落位 ② sha256 校验 ③ 报告还缺什么（如 hardness 未覆盖的任务）
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:---local}"

# 训练机上的目标根（可用环境变量覆盖）
WS_ROOT="${WS_ROOT:-/workspace}"
PHASES_DST="${PHASES_DST:-$WS_ROOT/al/phases_al}"
if [ "$MODE" = "--local" ]; then
    SPLIT_DST="${SPLIT_DST:-$WS_ROOT/al/task_splits_50}"
    SCOUT_DST="${SCOUT_DST:-$WS_ROOT/al/scout_cache}"
    HARD_DST="${HARD_DST:-$WS_ROOT/al/hardness_cache}"
    REF_DST="${REF_DST:-$WS_ROOT/eval_results/open_loop/ref50k}"
else
    TARGET="$2"; PORT="${3:-22}"; KEY="${4:-}"
    SSHOPT="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT"
    [ -n "$KEY" ] && SSHOPT="$SSHOPT -i $KEY"
fi

remote() { if [ "$MODE" = "--local" ]; then bash -c "$1"; else ssh $SSHOPT "$TARGET" "$1"; fi; }

echo "==================== 恢复 AL 派生产物 ===================="
remote "mkdir -p '$SPLIT_DST' '$SCOUT_DST' '$HARD_DST' '$PHASES_DST' '$REF_DST'"

if [ "$MODE" = "--local" ]; then
    cp -a "$HERE/al/task_splits_50/."   "$SPLIT_DST/"   2>/dev/null
    cp -a "$HERE/al/scout_cache/."      "$SCOUT_DST/"   2>/dev/null
    cp -a "$HERE/al/hardness_cache/."   "$HARD_DST/"    2>/dev/null
    cp -a "$HERE/al/phases_al/."        "$PHASES_DST/"  2>/dev/null
    cp -a "$HERE/eval_results/open_loop/ref50k/." "$REF_DST/" 2>/dev/null
else
    # 本地 rsync 2.6.9 不建父目录 ⇒ 先 mkdir（远端已建），再逐项同步
    rsync -a -e "ssh $SSHOPT" "$HERE/al/task_splits_50/"   "$TARGET:$SPLIT_DST/"
    rsync -a -e "ssh $SSHOPT" "$HERE/al/scout_cache/"      "$TARGET:$SCOUT_DST/"
    rsync -a -e "ssh $SSHOPT" "$HERE/al/hardness_cache/"   "$TARGET:$HARD_DST/"
    rsync -a -e "ssh $SSHOPT" "$HERE/al/phases_al/"        "$TARGET:$PHASES_DST/"
    rsync -a -e "ssh $SSHOPT" "$HERE/eval_results/open_loop/ref50k/" "$TARGET:$REF_DST/"
fi

echo "---- ① 落位结果 ----"
remote "for d in '$SPLIT_DST' '$SCOUT_DST' '$HARD_DST' '$PHASES_DST' '$REF_DST'; do echo -n \"  \$d: \"; ls \$d 2>/dev/null | wc -l; done"

echo "---- ② SHA256 校验（本包自校验）----"
if [ -f "$HERE/SHA256SUMS" ]; then
    ( cd "$HERE" && shasum -a 256 -c SHA256SUMS 2>/dev/null | grep -c ": OK" | awk '{print "  通过文件数: "$1}' )
else
    echo "  ⚠️ 缺 SHA256SUMS"
fi

echo "---- ③ 关键不变量核对 ----"
remote "PY=/opt/robotwin-env/bin/python; [ -x \$PY ] || PY=python3; \$PY - <<'EOF'
import json, pathlib
def j(p):
    try: return json.load(open(p))
    except Exception: return None
b = j('$SPLIT_DST/task_baseline.json') or {}
t = j('$REF_DST/pass_thresholds_gmean100_warn.json') or {}
s = j('$SCOUT_DST/scout.json') or {}
h = j('$HARD_DST/hardness.json') or {}
print('  baseline 指纹 :', b.get('config_fingerprint'))
print('  阈值表 指纹   :', t.get('config_fingerprint'), '(必须与上行相同)')
print('  baseline 任务数:', len(b.get('tasks') or {}))
print('  阈值表 任务数 :', len(t.get('tasks') or {}))
print('  scout   model :', s.get('model'), '| 记录', len(s.get('records') or {}))
hd = h.get('tasks') or {}
print('  hardness model:', h.get('model'), '| 已覆盖任务', len(hd), '（未覆盖的会在训练中补扫）')
ds = pathlib.Path('$PHASES_DST/datasets.txt')
if ds.is_file():
    lines = [x for x in ds.read_text().splitlines() if x.strip()]
    print('  datasets.txt  : %d 行（必须为 1 行）=>' % len(lines), lines[0][:80] if lines else '')
EOF"
echo "========================================================"
echo "恢复完成。启动训练请用：bash experiment/robotwin/start_al_8gpu.sh"
