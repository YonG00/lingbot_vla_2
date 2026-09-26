#!/bin/bash
# ============================================================
# Inference + simulation launcher (queue-scheduled).
# num_per_gpu inference servers stay resident per GPU; sim tasks match
# inference slots from a queue. Finishing a task frees its slot and starts the next.
#
# Usage: bash start_robotwin_infer_and_eval.sh [options]
#   --model_path        inference model path (default: /path/to/your/checkpoint, or $MODEL_PATH)
#   --inference_script  inference-side module path (default: deploy/lingbot_vla_v2_policy.py)
#   --inference_workdir inference-side working dir (default: current working dir)
#   --eval_workdir      sim-side (RoboTwin repo) working dir (REQUIRED; or $EVAL_WORKDIR; default placeholder /path/to/RoboTwin)
#   --inference_env     inference-side conda env (default: lingbotvla, or $INFERENCE_ENV)
#   --sim_env           sim-side conda env (default: RoboTwin, or $SIM_ENV)
#   --conda_sh          conda.sh path to source (default: /path/to/miniconda3/etc/profile.d/conda.sh, or $CONDA_SH)
#   --output_base       result output path (default: /path/to/VLABenchmarkResult, or $OUTPUT_BASE)
#   --start_port        starting port (default: 9330)
#   --pid_name          PID file prefix (default: test_pid)
#   --num_tasks         number of sim tasks, taken in order from the task list (default: 50, max: 50)
#   --num_gpus          total GPUs (default: 1)
#   --num_per_gpu       inference servers per GPU (default: 1)
#   --use_length        chunk length (default: 50)
#   --robo_name         robot config name (default: robotwin)
#   --video_fps         video recording fps (default: 10)
#   --no_video          disable video recording to speed up simulation
#   --use_bf16          use bfloat16 inference (default: False)
#   --use_fp32          use float32 inference (default: True; release reproduction setting)
#   --use_compile       enable model compile in policy inference (default: True)
#   --task_config       RoboTwin task config (default: demo_clean, or $TASK_CONFIG)
#   --keep_inference    keep inference servers resident after simulation
#
# Examples:
#   bash start_robotwin_infer_and_eval.sh --eval_workdir /path/to/RoboTwin --num_tasks 50 --num_gpus 4 --num_per_gpu 1
#   bash start_robotwin_infer_and_eval.sh --eval_workdir /path/to/RoboTwin --sim_env RoboTwin --num_tasks 20 --num_per_gpu 2
# ============================================================

# ===== Parse keyword arguments =====
export QWEN3VL_PATH="${QWEN3VL_PATH:-/path/to/your/checkpoints/Qwen3-VL-4B-Instruct}"

project_root="$(pwd)"
inference_workdir="${project_root}/"
inference_script="deploy/lingbot_vla_v2_policy.py"
model_path="${MODEL_PATH:-/path/to/your/checkpoint}"
eval_workdir="${EVAL_WORKDIR:-/path/to/RoboTwin}"
output_base="${OUTPUT_BASE:-/path/to/VLABenchmarkResult}"
# Conda env names + conda.sh path for both sides (overridable via flags or env).
inference_env="${INFERENCE_ENV:-lingbotvla}"
sim_env="${SIM_ENV:-RoboTwin}"
conda_sh="${CONDA_SH:-/path/to/miniconda3/etc/profile.d/conda.sh}"
start_port=9330
pid_name="test_pid"
num_tasks=50
num_gpus=1
num_per_gpu=1
use_length=50
use_bf16=False
use_fp32=True
use_compile=True
robo_name="robotwin"
video_fps=10
enable_video=True
keep_inference=false
enable_video=False
task_config="${TASK_CONFIG:-demo_clean}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --inference_workdir) inference_workdir="$2"; shift 2 ;;
        --inference_script)  inference_script="$2";  shift 2 ;;
        --model_path)        model_path="$2";        shift 2 ;;
        --eval_workdir)      eval_workdir="$2";      shift 2 ;;
        --output_base)       output_base="$2";       shift 2 ;;
        --start_port)        start_port="$2";        shift 2 ;;
        --pid_name)          pid_name="$2";          shift 2 ;;
        --num_tasks)         num_tasks="$2";         shift 2 ;;
        --num_gpus)          num_gpus="$2";          shift 2 ;;
        --num_per_gpu)       num_per_gpu="$2";       shift 2 ;;
        --use_length)        use_length="$2";        shift 2 ;;
        --use_bf16)          use_bf16="$2";        shift 2 ;;
        --use_fp32)          use_fp32="$2";        shift 2 ;;
        --use_compile)       use_compile="$2";     shift 2 ;;
        --robo_name)         robo_name="$2";         shift 2 ;;
        --task_config)       task_config="$2";       shift 2 ;;
        --video_fps)         video_fps="$2";         shift 2 ;;
        --no_video)          enable_video=False;     shift ;;
        --keep_inference)    keep_inference=true;    shift ;;
        --inference_env)     inference_env="$2";     shift 2 ;;
        --sim_env)           sim_env="$2";           shift 2 ;;
        --conda_sh)          conda_sh="$2";          shift 2 ;;
        -h|--help)
            echo "Usage: bash $0 [options]"
            echo "  --model_path        inference model path"
            echo "  --inference_script  inference-side module path"
            echo "  --inference_workdir inference-side working dir"
            echo "  --eval_workdir      sim-side (RoboTwin repo) working dir (REQUIRED; or \$EVAL_WORKDIR)"
            echo "  --inference_env     inference-side conda env (default: lingbotvla, or \$INFERENCE_ENV)"
            echo "  --sim_env           sim-side conda env (default: RoboTwin, or \$SIM_ENV)"
            echo "  --conda_sh          conda.sh path to source (default: /path/to/miniconda3/etc/profile.d/conda.sh, or \$CONDA_SH)"
            echo "  --output_base       result output path"
            echo "  --start_port        starting port (default: 9330)"
            echo "  --pid_name          PID file prefix (default: test_pid)"
            echo "  --num_tasks         number of sim tasks (default: 50, max: 50)"
            echo "  --num_gpus          total GPUs (default: 1)"
            echo "  --num_per_gpu       inference servers per GPU (default: 1)"
            echo "  --use_length        chunk length (default: 50)"
            echo "  --use_bf16          use bfloat16 inference (default: False)"
            echo "  --use_fp32          use float32 inference (default: True; release reproduction setting)"
            echo "  --use_compile       enable model compile (default: True)"
            echo "  --robo_name         robot config name (default: robotwin)"
            echo "  --task_config       RoboTwin task config (default: demo_clean)"
            echo "  --keep_inference    keep inference servers resident after simulation"
            echo "  --video_fps         video recording fps (default: 10)"
            echo "  --no_video          disable video recording to speed up simulation"
            exit 0 ;;
        *)
            echo -e "\033[31mUnknown argument: $1\033[0m"; exit 1 ;;
    esac
done


# ===== Common environment =====
# Cleanup: kill all child processes on exit / Ctrl-C / kill
cleanup() {
    echo ""
    echo -e "\033[33m=== Cleaning up all child processes ===\033[0m"
    # Kill inference servers
    for slot in $(seq 0 $((num_slots-1))); do
        local_pid=${inference_pids[$slot]:-0}
        if [ "$local_pid" != "0" ] && kill -0 "$local_pid" 2>/dev/null; then
            kill -TERM -- -"$(ps -o pgid= -p "$local_pid" 2>/dev/null | tr -d ' ')" 2>/dev/null \
                || kill -TERM "$local_pid" 2>/dev/null || true
        fi
    done
    # Kill eval workers
    for slot in $(seq 0 $((num_slots-1))); do
        local_pid=${slot_pid[$slot]:-0}
        if [ "$local_pid" != "0" ] && kill -0 "$local_pid" 2>/dev/null; then
            kill -TERM -- -"$(ps -o pgid= -p "$local_pid" 2>/dev/null | tr -d ' ')" 2>/dev/null \
                || kill -TERM "$local_pid" 2>/dev/null || true
        fi
    done
    sleep 1
    # Force kill survivors
    for slot in $(seq 0 $((num_slots-1))); do
        for local_pid in ${inference_pids[$slot]:-0} ${slot_pid[$slot]:-0}; do
            if [ "$local_pid" != "0" ] && kill -0 "$local_pid" 2>/dev/null; then
                kill -KILL -- -"$(ps -o pgid= -p "$local_pid" 2>/dev/null | tr -d ' ')" 2>/dev/null \
                    || kill -KILL "$local_pid" 2>/dev/null || true
            fi
        done
    done
    echo -e "\033[33m=== Cleanup done ===\033[0m"
}
trap cleanup EXIT INT TERM



# ===== Working dir (RoboTwin sim side) -- REQUIRED =====
# Must be supplied via --eval_workdir flag or $EVAL_WORKDIR env. No default.
if [ -z "$eval_workdir" ]; then
    eval_workdir="${EVAL_WORKDIR:-}"
fi
if [ -z "$eval_workdir" ]; then
    echo -e "\033[31mError: --eval_workdir (or \$EVAL_WORKDIR) is required (the RoboTwin repo root).\033[0m"
    exit 1
fi
if [ ! -d "$eval_workdir" ]; then
    echo -e "\033[31mError: sim workdir '${eval_workdir}' not found.\033[0m"
    echo "  Pass --eval_workdir <path> or set EVAL_WORKDIR=<path> (the RoboTwin repo root)."
    exit 1
fi
echo -e "\033[36mSim workdir: ${eval_workdir}\033[0m"

# ===== Sim-side python (conda env: ${sim_env}) =====
# conda_sh / sim_env / inference_env are resolved above (flag > env > builtin default).
if [ ! -f "$conda_sh" ]; then
    echo -e "\033[31mError: conda profile not found at ${conda_sh}\033[0m"
    exit 1
fi
# shellcheck disable=SC1090
source "$conda_sh"
if ! conda activate "$sim_env" 2>/dev/null; then
    echo -e "\033[31mError: conda env '${sim_env}' not found; create it first\033[0m"
    exit 1
fi
if ! command -v python >/dev/null 2>&1; then
    echo -e "\033[31mError: python command not found in conda env '${sim_env}'\033[0m"
    exit 1
fi
sim_python="$(command -v python)"
echo -e "\033[36mSim-side python (${sim_env}): ${sim_python} ($(${sim_python} --version 2>&1))\033[0m"
# Deactivate so later phases don't inherit sim env; the actual sim launch
# activates ${sim_env} itself via `setsid bash -c`.
conda deactivate 2>/dev/null || true

# ===== Task count validation =====
if [ "$num_tasks" -gt 50 ]; then
    echo -e "\033[31mError: num_tasks ${num_tasks} exceeds max 50; use 1~50\033[0m"
    exit 1
fi

# ===== Sim-side args =====
policy_name=ACT
train_config_name=0
seed=0

# ===== Compute inference slot count =====
# actual slots = min(num_tasks, num_gpus * num_per_gpu)
# When tasks < max concurrency, spin up only as many servers as tasks to avoid
# the waste of starting then shutting down extras.
num_slots=$(( num_gpus * num_per_gpu ))
if [ "$num_tasks" -lt "$num_slots" ]; then
    echo -e "\033[36mnum_tasks ${num_tasks} < max slots ${num_slots}; starting servers by task count\033[0m"
    num_slots=$num_tasks
fi

# ===== Full task list (50) =====
task_list_all=("lift_pot" "hanging_mug" "stack_bowls_three" "scan_object" "handover_block" "click_bell" "put_object_cabinet" "open_microwave" "stack_blocks_three" "place_shoe" "adjust_bottle" "beat_block_hammer" "blocks_ranking_rgb" "blocks_ranking_size" "click_alarmclock" "dump_bin_bigbin" "grab_roller" "handover_mic" "move_can_pot" "move_pillbottle_pad" "move_playingcard_away" "place_cans_plasticbox" "place_container_plate" "place_dual_shoes" "place_empty_cup" "place_fan" "place_mouse_pad" "place_object_basket" "place_object_scale" "place_object_stand" "place_phone_stand" "move_stapler_pad" "open_laptop" "pick_diverse_bottles" "pick_dual_bottles" "place_a2b_left" "place_a2b_right" "place_bread_basket" "place_bread_skillet" "place_burger_fries" "place_can_basket" "press_stapler" "rotate_qrcode" "shake_bottle_horizontally" "shake_bottle" "stack_blocks_two" "stack_bowls_two" "stamp_seal" "turn_switch" "put_bottles_dustbin")

# Build the sim task queue
task_queue=()
for i in $(seq 0 $((num_tasks-1))); do
    task_queue+=("${task_list_all[$i]}")
done
echo -e "\033[36mTasks this run (${num_tasks}): ${task_queue[*]}\033[0m"
echo -e "\033[36mInference config: ${num_gpus} GPU x ${num_per_gpu} servers/GPU = ${num_slots} slots\033[0m"
echo -e "\033[36mInference precision: use_bf16=${use_bf16}, use_fp32=${use_fp32}, use_compile=${use_compile}\033[0m"

# ===== Common variables =====
batch_time=$(date +%Y%m%d_%H%M%S)
# Extract experiment name and step from model_path (e.g. qwen35_robotwin_forward_no_mask_30k)
_exp_name=$(echo "$model_path" | grep -oP '[^/]+(?=/checkpoints)')
_step_num=$(echo "$model_path" | grep -oP 'global_step_\K\d+')
_step_k=$(( _step_num / 1000 ))k
run_dir="${output_base}/${_exp_name}_${_step_k}_${task_config}_${batch_time}"
mkdir -p "${run_dir}/inference_logs" "${run_dir}/eval_logs"
log_dir="${run_dir}"
inference_pid_file="${run_dir}/inference_pids.txt"
eval_pid_file="${run_dir}/eval_pids.txt"
> "$inference_pid_file"
> "$eval_pid_file"
stats_file="${run_dir}/stats.txt"
echo -e "\033[36mRun directory: ${run_dir}\033[0m"

# Result collection arrays
declare -a result_task_names=()
declare -a result_durations=()
declare -a result_logs=()
declare -a result_status=()

# ============================================================
# Phase 1: start inference-side QwenPi servers (resident)
# ============================================================
echo -e "\033[32m========== Starting inference side: ${num_slots} QwenPi servers, ports ${start_port}~$((start_port + num_slots - 1)) ==========\033[0m"

cd "$inference_workdir" || { echo -e "\033[31mError: inference workdir ${inference_workdir} missing\033[0m"; exit 1; }

for slot in $(seq 0 $((num_slots-1))); do
    gpu_id=$(( slot % num_gpus ))
    port=$(( start_port + slot ))

    export CUDA_VISIBLE_DEVICES=${gpu_id}

    log_file="${run_dir}/inference_logs/qwenpi_slot${slot}_gpu${gpu_id}_port${port}.log"

    echo -e "\033[33m[inf slot $slot] GPU: ${gpu_id}, PORT: ${port}, Log: ${log_file}\033[0m"

    # deploy/qwen3_5vl_dit_policy.py -> deploy.qwen3_5vl_dit_policy
    inference_script_for_module="${inference_script}"
    if [[ "${inference_script_for_module}" = /* ]]; then
        inference_script_for_module="${inference_script_for_module#${inference_workdir%/}/}"
    fi
    inference_module=$(echo "${inference_script_for_module}" | sed 's|/|.|g; s|\.py$||')
    setsid bash -c "source ${conda_sh} && conda activate ${inference_env} && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 python -m ${inference_module} \
        --model_path '${model_path}' \
        --use_length '${use_length}' \
        --use_bf16 "${use_bf16}" \
        --use_fp32 "${use_fp32}" \
        --use_compile "${use_compile}" \
        --port '${port}'" > "$log_file" 2>&1 &

    pid=$!
    echo "${pid}" >> "$inference_pid_file"
done

echo -e "\033[32m${num_slots} inference servers started, PIDs saved to ${inference_pid_file}\033[0m"

# Record each slot's inference server PID for on-demand shutdown
declare -a inference_pids=()
for slot in $(seq 0 $((num_slots-1))); do
    inference_pids[$slot]=0
done
# Re-read PIDs (in launch order, mapping 1:1 to slots)
mapfile -t pid_list < "$inference_pid_file"
for slot in $(seq 0 $((num_slots-1))); do
    if [ $slot -lt ${#pid_list[@]} ]; then
        inference_pids[$slot]=${pid_list[$slot]}
    fi
done
active_inference=$num_slots

# ============================================================
# Phase 2: queue-scheduled sim tasks
# ============================================================
echo -e "\033[32m========== Starting sim side (queue-scheduled, ${num_slots} concurrent slots) ==========\033[0m"

cd "$eval_workdir" || { echo -e "\033[31mError: sim workdir ${eval_workdir} missing\033[0m"; exit 1; }

# ===== Ensure the lingbot eval client is present under RoboTwin/script =====
# The eval client resolves its ../task_config relative to its own location, so it
# must live at <RoboTwin>/script/ for _camera_config.yml to be found.
eval_client_src="${inference_workdir}experiment/robotwin/eval_policy_client_lingbotvla.py"
eval_client_dst="${eval_workdir}/script/eval_policy_client_lingbotvla.py"
if [ ! -f "$eval_client_dst" ]; then
    if [ ! -f "$eval_client_src" ]; then
        echo -e "\033[31mError: eval client source not found: ${eval_client_src}\033[0m"
        exit 1
    fi
    echo -e "\033[36mCopying eval client -> ${eval_client_dst}\033[0m"
    cp "$eval_client_src" "$eval_client_dst"
fi

# ===== Ensure the deploy client helpers are present under RoboTwin/script/deploy =====
# The eval client does `from script.deploy.websocket_client_policy import WebsocketClientPolicy`,
# which pulls in a sibling msgpack_numpy. Copy any that are missing from the inference repo.
deploy_pkg_src="${inference_workdir}deploy"
deploy_pkg_dst="${eval_workdir}/script/deploy"
mkdir -p "$deploy_pkg_dst"
for f in __init__.py websocket_client_policy.py msgpack_numpy.py; do
    if [ ! -f "$deploy_pkg_dst/$f" ]; then
        if [ ! -f "$deploy_pkg_src/$f" ]; then
            echo -e "\033[31mError: deploy helper source not found: ${deploy_pkg_src}/${f}\033[0m"
            exit 1
        fi
        echo -e "\033[36mCopying deploy helper -> ${deploy_pkg_dst}/${f}\033[0m"
        cp "$deploy_pkg_src/$f" "$deploy_pkg_dst/$f"
    fi
done

# ===== Ensure curobo absolute paths point at this RoboTwin checkout =====
# Two artifacts embed absolute paths and go stale when RoboTwin is moved to a
# new directory (both would break the curobo planner at sim start):
#   (1) assets/embodiments/*/curobo*.yml -- urdf_path / collision_spheres
#       (regenerated from *_tmp.yml templates via script/update_embodiment_config_path.py)
#   (2) the editable-install .pth in the RoboTwin conda env pointing at envs/curobo/src
# Both checks are idempotent: only act when the embedded path != current location.
_eval_resolved="$(pwd -P)"   # physical path of this RoboTwin checkout (after cd)
_rep_yml="assets/embodiments/aloha-agilex/curobo_left.yml"
if [ -f "$_rep_yml" ] && ! grep -qF "${_eval_resolved}/assets" "$_rep_yml" 2>/dev/null; then
    echo -e "\033[36mCurobo yml paths stale -> regenerating for ${_eval_resolved}\033[0m"
    ( source ${conda_sh} && conda activate ${sim_env} \
      && python script/update_embodiment_config_path.py ) \
      || { echo -e "\033[31mError: curobo path regeneration failed\033[0m"; exit 1; }
fi
_env_site="$( source ${conda_sh} && conda activate ${sim_env} \
    && python -c 'import site;print(site.getsitepackages()[0])' )"
_curobo_pth="$(echo "${_env_site}"/__editable__.nvidia_curobo-*.pth)"
_expected_curobo_src="${_eval_resolved}/envs/curobo/src"
if [ -f "$_curobo_pth" ]; then
    _cur_curobo_src="$(grep -v '^[[:space:]]*#' "$_curobo_pth" | head -1)"
    if [ "$_cur_curobo_src" != "$_expected_curobo_src" ]; then
        echo -e "\033[36mCurobo editable .pth stale (${_cur_curobo_src}) -> ${_expected_curobo_src}\033[0m"
        echo "$_expected_curobo_src" > "$_curobo_pth"
    fi
fi

total_start_time=$(date +%s)

# Slot state: free or busy
# slot_pid[slot]   = sim process PID (0 = free)
# slot_task[slot]  = current task name
# slot_start[slot] = task start time
# slot_log[slot]   = log file path
declare -a slot_pid=()
declare -a slot_task=()
declare -a slot_start=()
declare -a slot_log=()
for slot in $(seq 0 $((num_slots-1))); do
    slot_pid[$slot]=0
    slot_task[$slot]=""
    slot_start[$slot]=0
    slot_log[$slot]=""
done

# Queue pointer and retry counters
queue_idx=0
max_retries=3
declare -A task_retries=()
for t in "${task_queue[@]}"; do
    task_retries[$t]=0
done

# Launch a sim task onto a given slot
launch_task() {
    local slot=$1
    local task_name=$2
    local gpu_id=$(( slot % num_gpus ))
    local port=$(( start_port + slot ))

    export CUDA_VISIBLE_DEVICES=${gpu_id}

    local log_file="${run_dir}/eval_logs/${task_name}.log"

    local attempt=$((task_retries[$task_name]+1))
    echo -e "\033[33m  [launch] ${task_name} -> slot $slot (GPU: ${gpu_id}, PORT: ${port}, Log: ${log_file}) [attempt ${attempt}/${max_retries}]\033[0m"

    # On retry, append instead of overwriting the failed log; mark this attempt with a banner
    {
        echo ""
        echo "================================================================"
        echo "  [attempt ${attempt}/${max_retries}] $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  task=${task_name}  slot=${slot}  gpu=${gpu_id}  port=${port}"
        echo "================================================================"
    } >> "$log_file"

    # Prepend the RoboTwin env's site-packages to PYTHONPATH (set inside the
    # bash -c after `conda activate`) so the env's numpy 1.26.4 -- which mplib /
    # sapien / open3d were compiled against -- is used instead of ~/.local's
    # numpy 2.x. The user-site numpy 2.x otherwise shadows it and segfaults
    # mplib's convert_physx_component during planner init.
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    setsid bash -c "source ${conda_sh} && conda activate ${sim_env} && export PYTHONPATH=\"\$(python -c 'import site;print(site.getsitepackages()[0])')\${PYTHONPATH:+:\$PYTHONPATH}\" && PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 python -u ${eval_client_dst} --config policy/${policy_name}/deploy_policy.yml \
        --overrides \
        --task_name ${task_name} \
        --task_config ${task_config} \
        --train_config_name ${train_config_name} \
        --seed ${seed} \
        --policy_name ${policy_name} \
        --port ${port} \
        --robo_name ${robo_name} \
        --video_fps ${video_fps} \
        --eval_video_log ${enable_video} \
        --output_dir '${run_dir}/eval_results'" >> "$log_file" 2>&1 &

    local pid=$!
    echo "${pid}" >> "$eval_pid_file"
    slot_pid[$slot]=$pid
    slot_task[$slot]="$task_name"
    slot_start[$slot]=$(date +%s)
    slot_log[$slot]="$log_file"
}

# Initial fill: assign a task to each free slot
for slot in $(seq 0 $((num_slots-1))); do
    if [ $queue_idx -lt ${#task_queue[@]} ]; then
        launch_task $slot "${task_queue[$queue_idx]}"
        queue_idx=$((queue_idx + 1))
    fi
done

# Shut down the inference server on a given slot
shutdown_inference_slot() {
    local slot=$1
    local inf_pid=${inference_pids[$slot]}
    if [ "$inf_pid" != "0" ] && kill -0 "$inf_pid" 2>/dev/null; then
        kill "$inf_pid" 2>/dev/null
        echo -e "\033[36m  [release] slot $slot inference server (PID: ${inf_pid}) stopped, remaining: $((active_inference - 1))\033[0m"
    fi
    inference_pids[$slot]=0
    active_inference=$((active_inference - 1))
}

# ============================================================
# Phase 3: scheduling loop - detect finished slots, reclaim and dispatch
# ============================================================
completed=0
skipped=0
total_tasks=${#task_queue[@]}

# Exit when all tasks done/skipped and nothing running
has_running() {
    for slot in $(seq 0 $((num_slots-1))); do
        [ "${slot_pid[$slot]}" != "0" ] && return 0
    done
    return 1
}

# Count remaining tasks (running + queued)
remaining_tasks() {
    local running=0
    for slot in $(seq 0 $((num_slots-1))); do
        [ "${slot_pid[$slot]}" != "0" ] && running=$((running + 1))
    done
    local queued=$(( ${#task_queue[@]} - queue_idx ))
    echo $((running + queued))
}

while [ $((completed + skipped)) -lt $total_tasks ] && has_running; do
    for slot in $(seq 0 $((num_slots-1))); do
        pid=${slot_pid[$slot]}

        # Skip free slots
        [ "$pid" = "0" ] && continue

        # Check whether the process has exited
        if ! kill -0 "$pid" 2>/dev/null; then
            task_end_time=$(date +%s)
            task_duration=$(( task_end_time - slot_start[$slot] ))
            wait "$pid"
            exit_code=$?

            task_name="${slot_task[$slot]}"
            log_file="${slot_log[$slot]}"

            if [ $exit_code -eq 0 ]; then
                completed=$((completed + 1))
                echo -e "\033[32m  [done] ${task_name} slot $slot took ${task_duration}s (progress ${completed}/${total_tasks})\033[0m"
                result_status+=("done")
                result_task_names+=("$task_name")
                result_durations+=("$task_duration")
                result_logs+=("$log_file")

                # Free the slot
                slot_pid[$slot]=0
                slot_task[$slot]=""
                slot_log[$slot]=""

                # Dispatch next task from the queue
                if [ $queue_idx -lt ${#task_queue[@]} ]; then
                    launch_task $slot "${task_queue[$queue_idx]}"
                    queue_idx=$((queue_idx + 1))
                fi
            else
                task_retries[$task_name]=$((task_retries[$task_name] + 1))
                if [ ${task_retries[$task_name]} -lt $max_retries ]; then
                    echo -e "\033[31m  [fail] ${task_name} slot $slot took ${task_duration}s exit=${exit_code}, retrying on this slot (${task_retries[$task_name]}/${max_retries})\033[0m"
                    # Retry immediately on the same slot
                    launch_task $slot "$task_name"
                else
                    skipped=$((skipped + 1))
                    echo -e "\033[31m  [skip] ${task_name} failed ${max_retries} times, no more retries\033[0m"
                    result_status+=("skip:${exit_code}")
                    result_task_names+=("$task_name")
                    result_durations+=("$task_duration")
                    result_logs+=("$log_file")

                    # Free the slot
                    slot_pid[$slot]=0
                    slot_task[$slot]=""
                    slot_log[$slot]=""

                    # Dispatch next task from the queue
                    if [ $queue_idx -lt ${#task_queue[@]} ]; then
                        launch_task $slot "${task_queue[$queue_idx]}"
                        queue_idx=$((queue_idx + 1))
                    fi
                fi
            fi
        fi
    done

    # Adaptive shutdown: when remaining tasks < active inference servers, stop idle-slot servers
    remaining=$(remaining_tasks)
    for slot in $(seq 0 $((num_slots-1))); do
        [ $remaining -lt $active_inference ] || break
        if [ "${slot_pid[$slot]}" = "0" ] && [ "${inference_pids[$slot]}" != "0" ]; then
            shutdown_inference_slot $slot
        fi
    done

    sleep 5
done

total_end_time=$(date +%s)
total_duration=$(( total_end_time - total_start_time ))

if [ $skipped -gt 0 ]; then
    echo -e "\033[31m========== Sim finished: ${completed} done, ${skipped} skipped, total ${total_duration}s ==========\033[0m"
else
    echo -e "\033[32m========== All sim tasks done, total ${total_duration}s ==========\033[0m"
fi

# ============================================================
# Phase 4: generate the stats file
# ============================================================
echo -e "\033[36mGenerating stats file: ${stats_file}\033[0m"

{
    echo "============================================"
    echo "  Eval Result Stats"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Model: ${_exp_name}_${_step_k}"
    echo "  Model path: ${model_path}"
    echo "  Tasks: ${num_tasks}"
    echo "  Task Config: ${task_config}"
    echo "  Inference: ${num_gpus} GPU x ${num_per_gpu}/GPU = ${num_slots} slots"
    echo "  Precision: use_bf16=${use_bf16}, use_fp32=${use_fp32}, use_compile=${use_compile}"
    echo "  Result: ${completed} done, ${skipped} skipped"
    echo "============================================"
    echo ""
    printf "%-30s %-10s %-12s %-10s %-10s\n" "Task" "Time(s)" "Done(100)" "Success/Total" "Rate"
    echo "--------------------------------------------------------------------------------"

    total_success=0
    total_episodes=0
    all_complete=true

    for i in "${!result_task_names[@]}"; do
        task_name="${result_task_names[$i]}"
        duration="${result_durations[$i]}"
        log_file="${result_logs[$i]}"
        status="${result_status[$i]}"

        # Extract success rate from the log (strip ANSI color codes)
        success_rate="-"
        episodes_done="-"
        if [ -f "$log_file" ]; then
            last_rate_line=$(sed 's/\x1b\[[0-9;]*m//g' "$log_file" | grep -oP 'Success rate: \K.*' | tail -1)
            if [ -n "$last_rate_line" ]; then
                suc_num=$(echo "$last_rate_line" | grep -oP '^\d+' | head -1)
                total_num=$(echo "$last_rate_line" | grep -oP '/\K\d+' | head -1)
                rate_pct=$(echo "$last_rate_line" | grep -oP '=> \K[\d.]+')
                if [ -n "$suc_num" ] && [ -n "$total_num" ]; then
                    success_rate="${rate_pct}%"
                    episodes_done="${suc_num}/${total_num}"
                    total_success=$((total_success + suc_num))
                    total_episodes=$((total_episodes + total_num))
                fi
            fi
        fi

        # Check whether all 100 episodes ran
        if [ "$episodes_done" != "-" ]; then
            total_ep=$(echo "$episodes_done" | cut -d'/' -f2)
            if [ "$total_ep" = "100" ]; then
                complete_mark="YES"
            else
                complete_mark="NO(${total_ep}/100)"
                all_complete=false
            fi
        else
            complete_mark="NO(0/100)"
            all_complete=false
        fi

        printf "%-30s %-10s %-12s %-10s %-10s\n" "$task_name" "$duration" "$complete_mark" "$episodes_done" "$success_rate"
    done

    echo "--------------------------------------------------------------------------------"
    if [ $total_episodes -gt 0 ]; then
        overall_rate=$(awk "BEGIN {printf \"%.1f\", $total_success/$total_episodes*100}")
    else
        overall_rate="0.0"
    fi
    printf "Summary: total %ds, success %d/%d, overall rate %s%%\n" "$total_duration" "$total_success" "$total_episodes" "$overall_rate"
    if $all_complete; then
        echo "All tasks fully executed 100 episodes"
    else
        echo "Warning: some tasks did not complete 100 episodes; check logs"
    fi
    echo "============================================"
} | tee "$stats_file"

# ============================================================
# Phase 5: handle inference servers (stop the remaining ones)
# ============================================================
if $keep_inference; then
    alive_count=0
    for slot in $(seq 0 $((num_slots-1))); do
        [ "${inference_pids[$slot]}" != "0" ] && alive_count=$((alive_count + 1))
    done
    echo -e "\033[36m========== Sim done, ${alive_count} inference servers kept resident ==========\033[0m"
    echo -e "\033[36mStop inference manually: kill \$(cat ${inference_pid_file})\033[0m"
else
    echo -e "\033[36m========== Stopping remaining inference servers ==========\033[0m"
    for slot in $(seq 0 $((num_slots-1))); do
        inf_pid=${inference_pids[$slot]}
        if [ "$inf_pid" != "0" ] && kill -0 "$inf_pid" 2>/dev/null; then
            kill "$inf_pid" 2>/dev/null
            echo -e "\033[36m  slot $slot inference server (PID: ${inf_pid}) stopped\033[0m"
        fi
    done
    echo -e "\033[32mAll inference servers stopped\033[0m"
fi

echo ""
echo -e "\033[32m========== All done ==========\033[0m"
echo -e "Run dir: ${run_dir}"
echo -e "Stats file: ${stats_file}"
echo -e "Inference logs: ${run_dir}/inference_logs/"
echo -e "Sim logs: ${run_dir}/eval_logs/"
echo -e "Eval results & videos: ${run_dir}/eval_results/"
if $keep_inference; then
    echo -e "Inference PIDs: ${inference_pid_file} (resident)"
else
    echo -e "Inference PIDs: ${inference_pid_file} (stopped)"
fi
echo -e "Sim PIDs: ${eval_pid_file}"
