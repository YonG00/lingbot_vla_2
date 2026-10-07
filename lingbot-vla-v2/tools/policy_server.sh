#!/usr/bin/env bash
# =============================================================================
# 常驻推理服务（websocket policy server）—— 只负责「把模型加载好并一直在线」。
#
# 为什么需要它
# ------------
# `start_robotwin_infer_and_eval.sh`（launcher）把「起服务 + 排队跑任务」焊在一个
# 脚本里 ⇒ **每加一批任务都要整套重来 = 重新加载模型**（本机 CPU 初始化阶段约
# 2.5–3 分钟，其间显存不涨、看着像卡死）。而且它的 `task_queue` 是启动时**一次性**
# 读入的（`mapfile < task_list_file`）⇒ 无法热追加任务。
#
# 本脚本把「起服务」单独拆出来常驻；之后用 `tools/eval_tasks.sh` 跑任务即可：
# **想加就加、模型不重载**。
#
# ⚠️ 代价（必须知道）
#   服务常驻 = 显存一直被占（约 28–35 GB）⇒ **单卡上不能同时训练**。
#   要开训练前，先 `bash tools/policy_server.sh stop`。
#
# 用法（在 /data/code/lingbot-vla-v2 下）：
#   CKPT_ROOT=/data/outputs/single/click_bell STEP=500 bash tools/policy_server.sh start
#   bash tools/policy_server.sh status
#   bash tools/policy_server.sh stop
#
# 环境变量：
#   CKPT_ROOT      训练输出目录（默认 /data/outputs/single/${TASK}）
#   TASK           仅用于推导默认 CKPT_ROOT
#   STEP           评哪一步 ckpt（默认 500）
#   PORT           端口（默认 9330）
#   USE_LENGTH     action chunk 长度（默认 50）
#   QWEN3VL        Qwen3-VL backbone（默认 /data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct）
#   REPO / CONDA_SH / INFERENCE_ENV
#   WAIT           start 后最多等多少秒就绪（默认 420；0 = 不等，立刻返回）
#   LOG / PIDFILE  默认 /data/tmp/policy_server_${PORT}.{log,pid}
# =============================================================================
set -euo pipefail

REPO=${REPO:-/data/code/lingbot-vla-v2}
TASK=${TASK:-click_bell}
CKPT_ROOT=${CKPT_ROOT:-/data/outputs/single/$TASK}
STEP=${STEP:-500}
PORT=${PORT:-9330}
USE_LENGTH=${USE_LENGTH:-50}
QWEN3VL=${QWEN3VL:-/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct}
CONDA_SH=${CONDA_SH:-/data/miniconda3/etc/profile.d/conda.sh}
INFERENCE_ENV=${INFERENCE_ENV:-lingbotvla}
WAIT=${WAIT:-420}
LOG=${LOG:-/data/tmp/policy_server_${PORT}.log}
PIDFILE=${PIDFILE:-/data/tmp/policy_server_${PORT}.pid}

hr() { printf '%.0s─' {1..78}; echo; }
die() { echo "❌ $*" >&2; exit 1; }

CKPT="$CKPT_ROOT/checkpoints/global_step_${STEP}/hf_ckpt"

gpu_mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1; }

# 端口是否在监听。
# ⚠️ 这个容器里**没有 `ss` 也没有 `netstat`**（实测 `ss: command not found`）⇒ 直接读
#    /proc/net/tcp{,6} 的 LISTEN(0A) 表项。零依赖、且**不会向 server 发起连接**。
port_listening() {
    local hex; hex=$(printf '%04X' "$PORT")
    awk -v h=":${hex}" '$4=="0A" && substr($2, length($2)-4) == h { f=1 } END { exit !f }' \
        /proc/net/tcp /proc/net/tcp6 2>/dev/null
}

pid_alive() {
    [[ -f "$PIDFILE" ]] || return 1
    local p; p=$(cat "$PIDFILE" 2>/dev/null || echo "")
    [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null
}

# ---- start ------------------------------------------------------------------
do_start() {
    if pid_alive; then
        echo "ℹ️  服务已在运行（PID=$(cat "$PIDFILE")，端口 $PORT）"
        port_listening && echo "   端口已在监听 ✅" || echo "   端口尚未监听（可能仍在 CPU 初始化阶段）"
        exit 0
    fi
    if port_listening; then
        die "端口 $PORT 已被别的进程占用（可能是另一个 launcher / 服务）。
   先确认：grep -i 2472 /proc/net/tcp        # 9330 = 0x2472，状态 0A = LISTEN
   或者换端口：PORT=9331 ... bash tools/policy_server.sh start"
    fi

    [[ -f "$CKPT/model.safetensors.index.json" ]] \
        || die "ckpt 不完整或缺 index.json：$CKPT"
    [[ -d "$QWEN3VL" ]] || die "QWEN3VL 路径不存在：$QWEN3VL"
    [[ -f "$CONDA_SH" ]] || die "找不到 $CONDA_SH"
    cd "$REPO" || die "仓库目录不存在：$REPO"

    hr
    cat <<EOF
  模式        : 常驻推理服务（只起服务，不跑任务）
  ckpt        : $CKPT
  端口        : $PORT
  chunk       : $USE_LENGTH
  精度        : fp32（use_bf16=false, use_fp32=true, use_compile=false）
  QWEN3VL_PATH: $QWEN3VL
  日志        : $LOG
  PID 文件    : $PIDFILE
EOF
    hr

    # 与 launcher 启动推理服务的命令**逐字一致**（见 start_robotwin_infer_and_eval.sh:343）
    export QWEN3VL_PATH="$QWEN3VL"
    setsid bash -c "source ${CONDA_SH} && conda activate ${INFERENCE_ENV} && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 python -m deploy.lingbot_vla_v2_policy \
        --model_path '${CKPT}' \
        --use_length '${USE_LENGTH}' \
        --use_bf16 false \
        --use_fp32 true \
        --use_compile false \
        --port '${PORT}'" > "$LOG" 2>&1 < /dev/null &

    local pid=$!
    echo "$pid" > "$PIDFILE"
    echo "🚀 已启动（PID=$pid，进程组同号 ⇒ stop 会整组结束）"

    if [[ "${WAIT}" -le 0 ]]; then
        echo "（WAIT=0 ⇒ 不等待就绪，用 status 查看）"
        exit 0
    fi

    # 等端口监听就绪；期间每 15 秒报一次，好看清进度
    local t0=$SECONDS
    while (( SECONDS - t0 < WAIT )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "❌ 服务进程已退出，日志尾部："
            tail -n 20 "$LOG" || true
            rm -f "$PIDFILE"
            exit 1
        fi
        if port_listening; then
            hr
            echo "✅ 服务就绪：端口 $PORT 已监听（耗时 $((SECONDS - t0))s）"
            echo "   显存占用 : $(gpu_mem)"
            echo "   下一步   : TASKS=\"<任务>\" EPISODES=5 bash tools/eval_tasks.sh"
            hr
            exit 0
        fi
        printf '   …等待就绪 %3ds / %ds（显存 %s，仍可能在 CPU 初始化）\n' \
               $((SECONDS - t0)) "$WAIT" "$(gpu_mem)"
        sleep 15
    done
    echo "⚠️  等满 ${WAIT}s 端口仍未监听。用 status 继续观察（本机 CPU 阶段可能较慢）。"
}

# ---- status -----------------------------------------------------------------
do_status() {
    hr
    echo "  端口        : $PORT"
    if pid_alive; then
        echo "  进程        : 运行中（PID=$(cat "$PIDFILE")）"
    else
        echo "  进程        : 未运行（无有效 PID 文件）"
    fi
    if port_listening; then
        echo "  监听        : ✅ 已在监听"
    else
        echo "  监听        : ❌ 未监听"
    fi
    echo "  显存占用    : $(gpu_mem)"
    echo "  实际进程    : $(pgrep -fc 'lingbot_vla_v2_polic[y]' || echo 0) 个"
    echo "  日志        : $LOG"
    echo "  ---- 日志尾部 ----"
    tail -n 6 "$LOG" 2>/dev/null || echo "  （无日志）"
    hr
}

# ---- stop -------------------------------------------------------------------
do_stop() {
    if ! pid_alive; then
        # 兜底：按进程名清理
        local n; n=$(pgrep -fc 'lingbot_vla_v2_polic[y]' || true)
        if [[ "${n:-0}" -gt 0 ]]; then
            echo "⚠️  PID 文件无效，但发现 $n 个推理进程 ⇒ 按进程名结束"
            pkill -f 'lingbot_vla_v2_polic[y]' || true
            sleep 2
            port_listening && echo "  端口仍在监听（可能已被别的服务接管）" || echo "  端口已释放 ✅"
        else
            echo "ℹ️  没有在运行的服务"
        fi
        rm -f "$PIDFILE"
        exit 0
    fi
    local pid; pid=$(cat "$PIDFILE")
    echo "🛑 结束服务（PID 组 $pid）…"
    kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 15); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "  仍在 ⇒ SIGKILL"
        kill -9 "-$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
        sleep 2
    fi
    rm -f "$PIDFILE"
    echo "  显存占用 : $(gpu_mem)"
    port_listening && echo "  ⚠️  端口仍在监听" || echo "  ✅ 端口已释放"
}

case "${1:-}" in
    start)  do_start  ;;
    status) do_status ;;
    stop)   do_stop   ;;
    *) echo "用法：bash tools/policy_server.sh {start|status|stop}"; exit 2 ;;
esac
