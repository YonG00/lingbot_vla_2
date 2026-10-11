#!/usr/bin/env bash
# =============================================================================
# 只跑 sim 侧 —— 连到**已在运行的**常驻推理服务（先跑 tools/misc/policy_server.sh start）。
#
# 好处：加任务 / 换任务 / 补测新任务**都不重载模型**（服务一直在 9330 端口等活）。
# 这是把 launcher 的「Phase 2（sim 侧排队）」单独拆出来，命令与 launcher 逐字一致
# （见 start_robotwin_infer_and_eval.sh:504-519），只是不再自己起服务。
#
# 用法（在 /data/code/lingbot-vla-v2 下）：
#   TASKS="stamp_seal,move_pillbottle_pad" EPISODES=5 bash tools/misc/eval_tasks.sh
#   TASKS="click_bell" EPISODES=5 CONFIG=demo_randomized bash tools/misc/eval_tasks.sh
#   DRY_RUN=1 TASKS="stamp_seal" bash tools/misc/eval_tasks.sh          # 只看计划
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

# === 本机适配层（ROCm / cpu1）—— 2026-10-11 新增；整体删除即回到 AutoDL 默认 ===
# 原脚本写死 AutoDL 路径（/data/code、/data/models、miniconda、/data/eval_results），
# 且一律 `conda activate`；本机**没有 conda**，只有 venv（/opt/robotwin-env）。
# 规则：AutoDL 路径存在 ⇒ 完全按原行为；不存在 ⇒ 用下面的本机默认值。
#       任何**已 export 的环境变量仍然优先**（`: "${VAR:=...}"` 语义）。
# =============================================================================
_ROCm_VENV=${ROCM_VENV:-/opt/robotwin-env}
_ROCm_OVERLAY=${ROCM_OVERLAY:-/models/robotwin-persistent}
if [ ! -d /data/miniconda3 ]; then
    : "${REPO:=$_REPO_DERIVED}"
    : "${ROBOTWIN_DIR:=/RoboTwin}"
    : "${QWEN3VL:=/workspace/models/Qwen3-VL-4B-Instruct-config-tokenizer}"
    : "${EVAL_RESULTS_ROOT:=$_ROCm_OVERLAY/eval_results}"
    : "${PRECISION:=bf16}"                # bf16 训练的 HF 导出是 bf16；fp32 服务不了它
    : "${AITER_USE_SYSTEM_TRITON:=1}"      # aiter 要求 triton>=3.6（镜像是 3.5.1）⇒ 不设则模型 import 直接 RuntimeError
    export AITER_USE_SYSTEM_TRITON
fi
#: conda 兼容层：有 conda.sh 就照旧 activate；没有就直接用 venv（设 PATH/PYTHONPATH/VIRTUAL_ENV）
run_env() {
    if [ -f "${CONDA_SH:-/nonexistent}" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_SH" && conda activate "$1"
    else
        export VIRTUAL_ENV="$_ROCm_VENV"
        export PATH="$_ROCm_VENV/bin:$PATH"
        local _sp
        _sp=$("$_ROCm_VENV/bin/python" -c 'import site;print(site.getsitepackages()[0])' 2>/dev/null || true)
        [ -n "$_sp" ] && export PYTHONPATH="${_sp}${PYTHONPATH:+:$PYTHONPATH}"
    fi
    return 0
}
# ========================= 适配层结束 =========================

# 参数区用的是 `${VAR:-<AutoDL 默认>}` ⇒ 未 export 时会填成 AutoDL 路径；
# 这里在**参数区之前**把它们改成本机默认（显式 export 的值不动）。
# 🔴 2026-10-11：仓库路径改为**按本脚本位置推导**（`tools/misc/` 往上两级 = 仓库根）。
_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_REPO_DERIVED=$(cd "$_SCRIPT_DIR/../.." && pwd)
if [ ! -d /data/miniconda3 ]; then
    [ "${REPO:-}" = "/data/code/lingbot-vla-v2" ]              && REPO=$_REPO_DERIVED
    [ "${ROBOTWIN_DIR:-}" = "/data/code/RoboTwin-lingbot" ]    && ROBOTWIN_DIR=/RoboTwin
    [ "${QWEN3VL:-}" = "/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct" ] \
        && QWEN3VL=/workspace/models/Qwen3-VL-4B-Instruct-config-tokenizer
    [ -z "${PRECISION:-}" ]  && PRECISION=bf16
    export REPO ROBOTWIN_DIR QWEN3VL PRECISION
fi
export _ROCm_VENV _ROCm_OVERLAY

REPO=${REPO:-$_REPO_DERIVED}
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
POLICY_NAME=${POLICY_NAME:-LingBot-VLA-v2}
#: 客户端分叉：official = /RoboTwin 新分叉（官方 run_clean_benchmark.py 同款）；legacy = 老分叉
CLIENT_FLAVOR=${CLIENT_FLAVOR:-official}
#: 官方客户端的脚本路径（仅 official 用）
CLIENT_PY=${CLIENT_PY:-/RoboTwin/scripts/eval_policy_xpolicylab.py}
#: 官方客户端的仿真设备号（单卡测试恒为 0；多卡时每个 slot 各给一个）
DEVICE_ID=${DEVICE_ID:-0}
#: 指令口径：seen | unseen。⚠️ 上一轮"闭环 0/5"就是因为跑成了 unseen，这里显式默认 seen
INSTRUCTION_TYPE=${INSTRUCTION_TYPE:-seen}
EXPERT_CHECK=${EXPERT_CHECK:-true}
ACCEPT_EXPERT_INFO_ON_FAILURE=${ACCEPT_EXPERT_INFO_ON_FAILURE:-true}
ROBO_NAME=${ROBO_NAME:-robotwin}
VIDEO_FPS=${VIDEO_FPS:-10}

# =============================================================================


hr() { printf '%.0s─' {1..78}; echo; }
die() { echo "❌ $*" >&2; exit 1; }

[[ -n "$TASKS" ]] || die "必须给 TASKS（例：TASKS=\"stamp_seal,move_pillbottle_pad\"）"
{ [ -f "$CONDA_SH" ] || [ -x "$_ROCm_VENV/bin/python" ]; } || die "既找不到 $CONDA_SH，也没有 venv：$_ROCm_VENV"
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
   先起常驻服务：  CKPT_ROOT=<训练输出目录> STEP=$STEP bash tools/misc/policy_server.sh start
   或看状态：      bash tools/misc/policy_server.sh status"

# ---- ② 幂等同步（launcher Phase 2 开头做的事，缺了会跑到旧代码）-----------------
if [[ "${CLIENT_FLAVOR}" == "official" ]]; then
    [[ -f "$CLIENT_PY" ]] || die "official 客户端不存在：$CLIENT_PY"
    CLIENT_DST="$CLIENT_PY"      # 由 RoboTwin 仓库自带，无需同步
else
    CLIENT_SRC="$REPO/experiment/robotwin/eval_policy_client_lingbotvla.py"
    mkdir -p "$ROBOTWIN_DIR/script"
    CLIENT_DST="$ROBOTWIN_DIR/script/eval_policy_client_lingbotvla.py"
fi
if [[ "${CLIENT_FLAVOR}" == "legacy" ]]; then
    [[ -f "$CLIENT_SRC" ]] || die "找不到评测客户端源文件：$CLIENT_SRC"
    if [[ ! -f "$CLIENT_DST" ]] || ! cmp -s "$CLIENT_SRC" "$CLIENT_DST"; then
        cp "$CLIENT_SRC" "${CLIENT_DST}.tmp.$$" && mv -f "${CLIENT_DST}.tmp.$$" "$CLIENT_DST"
        echo "🔄 已同步 eval client -> $CLIENT_DST"
    fi
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
    if [ -f "$CONDA_SH" ]; then
        ( cd "$ROBOTWIN_DIR" && run_env "$SIM_ENV" \
          && python script/update_embodiment_config_path.py ) || die "curobo 路径重生成失败"
    else
        echo "ℹ️  无 conda（ROCm/venv）⇒ 跳过 curobo yml 自愈；仿真侧由"
        echo "    ROBOTWIN_DISABLE_CUROBO=1 + ROBOTWIN_EE_PLANNER=mplib 走 mplib 规划（官方脚本同款）"
    fi
fi
_env_site="$( if [ -f "$CONDA_SH" ]; then cd "$ROBOTWIN_DIR" && run_env "$SIM_ENV" \
    && python -c 'import site;print(site.getsitepackages()[0])'; else \
    "$_ROCm_VENV/bin/python" -c 'import site;print(site.getsitepackages()[0])'; fi )"
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
RUN_DIR="${EVAL_RESULTS_ROOT:-/data/eval_results}/closed_loop/${TAG}/server_step${STEP}_${CONFIG#demo_}_${TS}"
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
export -f run_env 2>/dev/null || true

launch_one() {
    local task="$1" log="$2"
    cd "$ROBOTWIN_DIR"
    # 仿真侧公共环境（与官方 run_clean_benchmark.py 同款）
    local sim_env_exports="export PYTHONPATH=\"\$(python -c 'import site;print(site.getsitepackages()[0])')\${PYTHONPATH:+:\$PYTHONPATH}\" \
        && export ROBOTWIN_DISABLE_CUROBO=1 ROBOTWIN_EE_PLANNER=mplib PYOPENGL_PLATFORM=egl \
        && export AITER_USE_SYSTEM_TRITON=1"

    if [[ "${CLIENT_FLAVOR}" == "official" ]]; then
        # ── 官方分叉：/RoboTwin/scripts/eval_policy_xpolicylab.py（不需要 --config）─────
        mkdir -p "${RUN_DIR}/episode_info"
        setsid bash -c "run_env ${SIM_ENV} \
            && ${sim_env_exports} \
            && PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
               SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 python -u ${CLIENT_PY} \
            --bench_name RoboTwin \
            --task_name ${task} \
            --task_config ${CONFIG} \
            --policy_name ${POLICY_NAME} \
            --protocol lingbot_vla_v2 \
            --host 127.0.0.1 \
            --port ${PORT} \
            --device_id ${DEVICE_ID} \
            --seed ${SEED} \
            --test_num ${EPISODES} \
            --expert_check ${EXPERT_CHECK} \
            --accept_expert_info_on_failure ${ACCEPT_EXPERT_INFO_ON_FAILURE} \
            --instruction_type ${INSTRUCTION_TYPE} \
            --eval_batch false \
            --episode_info_output ${RUN_DIR}/episode_info/${task}.jsonl \
            --additional_info eval_video_log=False" > "$log" 2>&1

        # 官方客户端把成功率打到 stdout（`Final success rate: N/M = x%`）
        # ⇒ 反写 _result.txt，让本脚本既有的「末行=成功率」汇总逻辑继续可用。
        local res="${RUN_DIR}/eval_results/${task}/_result.txt"
        mkdir -p "$(dirname "$res")"
        local rate
        rate=$(grep -aoE 'Final success rate: [0-9]+/[0-9]+' "$log" 2>/dev/null | tail -1 \
               | grep -oE '[0-9]+/[0-9]+' | awk -F/ '{printf "%.4f", ($2>0? $1/$2 : 0)}')
        if [[ -n "$rate" ]]; then
            printf '%s\n' "$rate" > "$res"
            echo "  （已从日志反写成功率 $rate -> $res）"
        fi
    else
        # ── 老分叉（原 AutoDL 行为，保留以便回滚）──────────────────────────────────
        setsid bash -c "run_env ${SIM_ENV} \
            && ${sim_env_exports} \
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
    fi
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
