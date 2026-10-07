#!/usr/bin/env bash
# =============================================================================
# 只跑 sim 侧 —— 连到**已在运行的**常驻推理服务（先跑 tools/policy_server.sh start）。
#
# 好处：加任务 / 换任务 / 补测新任务**都不重载模型**（服务一直在 9330 端口等活）。
# 这是把 launcher 的「Phase 2（sim 侧排队）」单独拆出来，命令与 launcher 逐字一致
# （见 start_robotwin_infer_and_eval.sh:504-519），只是不再自己起服务。
#
# 用法（在 /data/code/lingbot-vla-v2 下）：
#   TASKS="stamp_seal,move_pillbottle_pad" EPISODES=5 bash tools/eval_tasks.sh
#   TASKS="click_bell" EPISODES=5 CONFIG=demo_randomized bash tools/eval_tasks.sh
#   DRY_RUN=1 TASKS="stamp_seal" bash tools/eval_tasks.sh          # 只看计划
#
# 环境变量：
#   TASKS        必填，逗号/空格分隔的任务名（必须是 RoboTwin 50 个官方任务）
#   EPISODES     每个任务的回合数（默认 5）
#   CONFIG       demo_clean / demo_randomized（默认 demo_clean）
#   PORT         推理服务端口（默认 9330）
#   SEED         基础种子（默认 0；客户端内部按 100000+ 派生）
#   STEP         仅用于默认 TAG / 目录命名（默认 500）
#   TAG          输出子目录名（默认 step${STEP}_${CONFIG#demo_}，与既有布局一致）
#   ROBOTWIN_DIR / CONDA_SH / SIM_ENV / REPO
#   DRY_RUN      1 = 只打印计划
#
# 与 launcher 的差别（有意为之，别无脑对比）：
#   * 不做 slot 看门狗 / 重试；某个任务失败只记录，然后继续下一个。
#   * 运行目录名带 `server_` 前缀，以区分"走常驻服务"的跑法。
# =============================================================================
set -euo pipefail

REPO=${REPO:-/data/code/lingbot-vla-v2}
TASKS=${TASKS:-}
EPISODES=${EPISODES:-5}
CONFIG=${CONFIG:-demo_clean}
PORT=${PORT:-9330}
SEED=${SEED:-0}
STEP=${STEP:-500}
TAG=${TAG:-step${STEP}_${CONFIG#demo_}}
ROBOTWIN_DIR=${ROBOTWIN_DIR:-/data/code/RoboTwin-lingbot}
CONDA_SH=${CONDA_SH:-/data/miniconda3/etc/profile.d/conda.sh}
SIM_ENV=${SIM_ENV:-RoboTwin}
DRY_RUN=${DRY_RUN:-0}
POLICY_NAME=${POLICY_NAME:-ACT}
ROBO_NAME=${ROBO_NAME:-robotwin}
VIDEO_FPS=${VIDEO_FPS:-10}

hr() { printf '%.0s─' {1..78}; echo; }
die() { echo "❌ $*" >&2; exit 1; }

[[ -n "$TASKS" ]] || die "必须给 TASKS（例：TASKS=\"stamp_seal,move_pillbottle_pad\"）"
[[ -f "$CONDA_SH" ]] || die "找不到 $CONDA_SH"
[[ -d "$ROBOTWIN_DIR" ]] || die "RoboTwin 仓库不存在：$ROBOTWIN_DIR"
cd "$REPO" || die "仓库目录不存在：$REPO"

read -r -a TASK_ARR <<< "$(printf '%s' "$TASKS" | tr ',' ' ')"
N_TASKS=${#TASK_ARR[@]}

# ---- ① 服务必须在跑（否则连不上，白跑）----------------------------------------
# ⚠️ 本容器没有 `ss` / `netstat`（实测）⇒ 读 /proc/net/tcp{,6} 的 LISTEN(0A) 表项。
port_listening() {
    local hex; hex=$(printf '%04X' "$PORT")
    awk -v h=":${hex}" '$4=="0A" && substr($2, length($2)-4) == h { f=1 } END { exit !f }' \
        /proc/net/tcp /proc/net/tcp6 2>/dev/null
}
port_listening || die "端口 $PORT 没有服务在监听。
   先起常驻服务：  CKPT_ROOT=<训练输出目录> STEP=$STEP bash tools/policy_server.sh start
   或看状态：      bash tools/policy_server.sh status"

# ---- ② 幂等同步（launcher Phase 2 开头做的事，缺了会跑到旧代码）-----------------
CLIENT_SRC="$REPO/experiment/robotwin/eval_policy_client_lingbotvla.py"
CLIENT_DST="$ROBOTWIN_DIR/script/eval_policy_client_lingbotvla.py"
[[ -f "$CLIENT_SRC" ]] || die "找不到评测客户端源文件：$CLIENT_SRC"
if [[ ! -f "$CLIENT_DST" ]] || ! cmp -s "$CLIENT_SRC" "$CLIENT_DST"; then
    cp "$CLIENT_SRC" "${CLIENT_DST}.tmp.$$" && mv -f "${CLIENT_DST}.tmp.$$" "$CLIENT_DST"
    echo "🔄 已同步 eval client -> $CLIENT_DST"
fi
DEPLOY_DST="$ROBOTWIN_DIR/script/deploy"
mkdir -p "$DEPLOY_DST"
for f in __init__.py websocket_client_policy.py msgpack_numpy.py; do
    if [[ ! -f "$DEPLOY_DST/$f" ]] || ! cmp -s "$REPO/deploy/$f" "$DEPLOY_DST/$f"; then
        cp "$REPO/deploy/$f" "$DEPLOY_DST/$f" && echo "🔄 已同步 deploy/$f"
    fi
done

# ---- ③ curobo 路径自愈（幂等；只在与当前路径不符时动手）------------------------
_eval_resolved="$(cd "$ROBOTWIN_DIR" && pwd -P)"
_rep_yml="$ROBOTWIN_DIR/assets/embodiments/aloha-agilex/curobo_left.yml"
if [[ -f "$_rep_yml" ]] && ! grep -qF "${_eval_resolved}/assets" "$_rep_yml" 2>/dev/null; then
    echo "🔄 curobo yml 路径过期 ⇒ 重新生成"
    ( cd "$ROBOTWIN_DIR" && source "$CONDA_SH" && conda activate "$SIM_ENV" \
      && python script/update_embodiment_config_path.py ) || die "curobo 路径重生成失败"
fi
_env_site="$( cd "$ROBOTWIN_DIR" && source "$CONDA_SH" && conda activate "$SIM_ENV" \
    && python -c 'import site;print(site.getsitepackages()[0])' )"
_curobo_pth="$(echo "${_env_site}"/__editable__.nvidia_curobo-*.pth)"
_expected_curobo_src="${_eval_resolved}/envs/curobo/src"
if [[ -f "$_curobo_pth" ]]; then
    _cur="$(grep -v '^[[:space:]]*#' "$_curobo_pth" | head -1)"
    if [[ "$_cur" != "$_expected_curobo_src" ]]; then
        echo "$_expected_curobo_src" > "$_curobo_pth"
        echo "🔄 已修正 curobo editable .pth"
    fi
fi

# ---- ④ 运行目录 ---------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="/data/eval_results/closed_loop/${TAG}/server_step${STEP}_${CONFIG#demo_}_${TS}"
mkdir -p "$RUN_DIR/eval_logs" "$RUN_DIR/eval_results"
STATS="$RUN_DIR/stats.txt"

hr
cat <<EOF
  模式        : 只跑 sim 侧（连常驻服务，模型不重载）
  任务        : ${TASK_ARR[*]}
  任务个数    : $N_TASKS
  条件        : $CONFIG   （每任务回合数 $EPISODES）
  推理服务    : 127.0.0.1:$PORT   （已在监听 ✅）
  输出        : $RUN_DIR
EOF
hr
if [[ "$DRY_RUN" == "1" ]]; then echo "[dry-run] 只打印计划"; exit 0; fi

# 客户端命令模板（与 launcher 逐字一致）
launch_one() {
    local task="$1" log="$2"
    cd "$ROBOTWIN_DIR"
    setsid bash -c "source ${CONDA_SH} && conda activate ${SIM_ENV} \
        && export PYTHONPATH=\"\$(python -c 'import site;print(site.getsitepackages()[0])')\${PYTHONPATH:+:\$PYTHONPATH}\" \
        && PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
           SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 python -u ${CLIENT_DST} \
        --config policy/${POLICY_NAME}/deploy_policy.yml \
        --overrides \
        --task_name ${task} \
        --task_config ${CONFIG} \
        --train_config_name 0 \
        --seed ${SEED} \
        --policy_name ${POLICY_NAME} \
        --port ${PORT} \
        --robo_name ${ROBO_NAME} \
        --video_fps ${VIDEO_FPS} \
        --eval_video_log False \
        --output_dir ${RUN_DIR}/eval_results \
        --test_num ${EPISODES}" > "$log" 2>&1
}

declare -a SUM_LINES=()
OK=0; BAD=0

for task in "${TASK_ARR[@]}"; do
    log="$RUN_DIR/eval_logs/${task}.log"
    echo "▶ [$(date +%H:%M:%S)] $task 开始（$EPISODES 回合）"
    t0=$(date +%s)
    rc=0
    launch_one "$task" "$log" || rc=$?      # set -e 下必须用 `|| rc=$?` 接住
    t1=$(date +%s); dur=$((t1 - t0))

    res="$RUN_DIR/eval_results/${task}/_result.txt"
    rate="—"
    if [[ -f "$res" ]]; then
        rate=$(grep -vE '^[[:space:]]*$' "$res" 2>/dev/null | tail -1 | tr -d '[:space:]' || true)
        [[ -n "$rate" ]] || rate="—"
    fi
    if [[ -f "$res" ]]; then
        OK=$((OK + 1))
        # 由 0–1 的 rate 反推成功回合数（用于汇总表）
        succ=$(awk -v r="$rate" -v n="$EPISODES" 'BEGIN{printf "%d", r*n+0.5}')
        printf '  ✅ %s 完成：%ds ｜ 成功 %s/%s（%s）\n' "$task" "$dur" "$succ" "$EPISODES" "$rate"
        SUM_LINES+=("$(printf '%-30s %10d %10s/%s %12s' "$task" "$dur" "$succ" "$EPISODES" "$rate")")
    else
        BAD=$((BAD + 1))
        printf '  ❌ %s 未产出结果（rc=%s，%ds）⇒ 看日志：%s\n' "$task" "$rc" "$dur" "$log"
        SUM_LINES+=("$(printf '%-30s %10d %10s %12s' "$task" "$dur" "FAILED" "-")")
    fi
done

# ---- ⑤ 汇总（格式与既有 stats.txt 对齐）--------------------------------------
{
    echo "============================================"
    echo "  Eval Result Stats  (常驻服务模式)"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Model: server_step${STEP}"
    echo "  Tasks: $N_TASKS"
    echo "  Task Config: $CONFIG"
    echo "  Inference: 常驻 service on port ${PORT}（模型只加载一次）"
    echo "  Result: $OK done, $BAD failed"
    echo "============================================"
    echo
    printf '%-30s %10s %14s %12s\n' "Task" "Time(s)" "Success/Total" "Rate"
    printf '%.0s-' {1..78}; echo
    for l in "${SUM_LINES[@]}"; do echo "$l"; done
} | tee "$STATS"

hr
echo "汇总已写入：$STATS"
[[ "$BAD" -gt 0 ]] && exit 1 || exit 0
