#!/usr/bin/env bash
# AL 仪表盘一键启动（2026-10-12）—— 免记 3 个环境变量、默认自动选最新 run 并持续刷新。
#
# 用法（在机器上，任意目录）：
#     bash tools/auto_learning_pipeline/al_watch.sh                 # 自动选最新 run，每 3s 刷新
#     bash tools/auto_learning_pipeline/al_watch.sh al_v65_formal   # 指定 run
#     bash tools/auto_learning_pipeline/al_watch.sh al_v65_formal --pool   # 透传面板参数
#     ONCE=1 bash tools/auto_learning_pipeline/al_watch.sh          # 只打印一次
#
# 环境变量（都可省）：
#     AL_ROOT       run 根目录（默认自动探测）
#     AL_WATCH_SEC  刷新间隔秒（默认 3）
#     PY            python 解释器（默认 /opt/robotwin-env/bin/python）
#     ONCE=1        只打印一次（脚本/日志用）
#
# 为什么需要它（2026-10-12 用户踩坑）：
#   · 面板默认 root 是 `/models/robotwin-persistent/al_runs`，而 v46+ 的 run 在
#     `/workspace/al/al_launch_runs` ⇒ 不设 AL_ROOT 会报"未找到 run"；
#   · 面板要在**仓库根**下运行（PYTHONPATH=<repo>），否则 import 失败；
#   · run 名必须写全（`al_v65` ≠ `al_v65_formal`）—— 本脚本会在写错时列出候选。
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/opt/robotwin-env/bin/python}"
SEC="${AL_WATCH_SEC:-3}"
PANEL="$REPO/tools/auto_learning_pipeline/al_panel.py"

# ---- 根目录自动探测 ----
# 🔴 为什么不能"存在就用"（2026-10-12 实测）：本机**两个**根目录都在
#   · /models/robotwin-persistent/al_runs ← 老生产目录（只有 al_v44 及更早）
#   · /workspace/al/al_launch_runs       ← v46+ 的新 run 都在这
#   固定优先生产目录 ⇒ 会选到**旧的 al_v44**、且找不到新 run。
#   ⇒ 按"**训练日志最新修改时间**"判定：正在跑的 run 日志一定在实时更新。
if [[ -z "${AL_ROOT:-}" ]]; then
  _best_root=""; _best_mtime=-1
  for _d in /models/robotwin-persistent/al_runs /workspace/al/al_launch_runs; do
    [[ -d "$_d/logs" ]] || continue
    _m=$(ls -t "$_d"/logs/train_al_*.log 2>/dev/null | head -1)
    [[ -n "$_m" ]] || continue
    _t=$(stat -c %Y "$_m" 2>/dev/null || echo 0)
    if (( _t > _best_mtime )); then _best_mtime=$_t; _best_root=$_d; fi
  done
  AL_ROOT="$_best_root"
  if [[ -z "$AL_ROOT" ]]; then
    echo "  ✗ 找不到 al_runs 根目录；请显式设 AL_ROOT=<目录>" >&2
    exit 2
  fi
fi
export AL_ROOT PYTHONPATH="$REPO"

# ---- 第一个非选项参数 = run 名（可省）----
RUN=""
ARGS=()
for a in "$@"; do
  if [[ -z "$RUN" && "$a" != -* ]]; then RUN="$a"; else ARGS+=("$a"); fi
done

# ---- run 名写错 ⇒ 给出候选（省一轮来回）----
# 🔴 两种目录布局都要认（2026-10-12 用户实测踩坑）：
#   · 旧：`<root>/train_<run>/`   —— 启动器**不传 `--train-out`** 时（v65 及更早）
#   · 新：`<root>/<run>/`         —— 启动器**显式传 `--train-out <root>/<run>`** 时（v69 起）
#     两者内部结构相同（`lingbotvla_cli.yaml` / `auto_learning_events.jsonl` / `runs` / `checkpoints`）。
#   只认旧布局 ⇒ 对新布局误报"找不到 run 目录"（v69 实测：run 在 /work_ov/al_runs/al_v69）。
_rundir=""
for _cand in "$AL_ROOT/train_$RUN" "$AL_ROOT/$RUN"; do
  if [[ -n "$RUN" && -f "$_cand/lingbotvla_cli.yaml" ]]; then _rundir="$_cand"; break; fi
done
if [[ -n "$RUN" && -z "$_rundir" ]]; then
  echo "  ✗ 找不到 run 目录：$AL_ROOT/train_$RUN（也不存在 $AL_ROOT/$RUN，或其中缺 lingbotvla_cli.yaml）" >&2
  echo "    现有 run（最近 8 个）：" >&2
  ls -1dt "$AL_ROOT"/train_* "$AL_ROOT"/al_v* 2>/dev/null | head -8 \
    | sed -E 's|.*/(train_)?|      |' >&2
  exit 3
fi

if [[ "${ONCE:-0}" == "1" ]]; then
  exec "$PY" "$PANEL" ${RUN:+"$RUN"} ${ARGS[@]+"${ARGS[@]}"}
fi
exec "$PY" "$PANEL" ${RUN:+"$RUN"} ${ARGS[@]+"${ARGS[@]}"} --watch "$SEC"
