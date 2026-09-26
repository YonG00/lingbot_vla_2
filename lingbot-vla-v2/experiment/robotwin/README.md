# Generate Lerobot Dataset from RoboTwin Data

This guide explains how to process raw data from **RoboTwin** and convert it into the **LerobotDataset** format following the official RoboTwin instructions.

## 1. Clone the Official RoboTwin Repository
```bash
git clone git@github.com:RoboTwin-Platform/RoboTwin.git
cd RoboTwin
git checkout 13c3c47ff4312dd62484bcd51be034af55c062d1
```

## 2. Create Required Directories
Navigate to the `policy/pi0` directory inside the cloned RoboTwin repository and create the folders:

```bash
cd ./policy/pi0
mkdir processed_data training_data
```

## 3. Convert RoboTwin Raw Data to HDF5

Download [official dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset) and unzip the dataset to '/path/to/RoboTwin/data'

**Example:**
```bash
data
└── adjust_bottle
    └── aloha-agilex_clean_50
```

Use the provided script [process_data_pi0.sh](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/policy/pi0/process_data_pi0.sh):

```bash
cd policy/pi0
bash process_data_pi0.sh ${task_name} ${task_config} ${expert_data_num}
```

**Example (clean demo):**
```bash
bash process_data_pi0.sh adjust_bottle aloha-agilex_clean_50 50
```

**Example (randomized demo):**
```bash
bash process_data_pi0.sh adjust_bottle aloha-agilex_randomized_500 50
```

If successful, the output folder:
```
processed_data/${task_name}-${task_config}-${expert_data_num}/
```

## 4. Prepare Training Data

Copy the required processed datasets into `training_data/${model_name}`:

```bash
cp -r processed_data/${task_name}-${task_config}-${expert_data_num} \
      training_data/${model_name}/
```

## 5. Ensure Sufficient Disk Space

The generated **LerobotDataset** will be stored under:

```
$XDG_CACHE_HOME/huggingface/lerobot/${repo_id}
```

By default, `XDG_CACHE_HOME` points to `~/.cache`, which must have sufficient free space.  
If space is low, change the cache location:

```bash
export XDG_CACHE_HOME=/path/to/your/cache
```

## 6. Generate LerobotDataset v2.1 Format

Run [generate.sh ](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/policy/pi0/generate.sh) to convert the HDF5 datasets to Lerobot.

Parameters:
- **hdf5_path**: Path to the HDF5 training data (e.g., `./training_data/${model_name}/`)
- **repo_id**: Name for the dataset (e.g., `my_repo`)

```bash
bash generate.sh ${hdf5_path} ${repo_id}
```

**Example:**
```bash
bash generate.sh ./training_data/demo_clean/ demo_clean_repo
```

Output:
```
${XDG_CACHE_HOME}/huggingface/lerobot/${repo_id}
```

---

## 7. Simulated Evaluation (Inference + RoboTwin Sim)

After training, evaluate your checkpoint on the 50 RoboTwin tasks (100 episodes each) with
[`start_robotwin_infer_and_eval.sh`](./start_robotwin_infer_and_eval.sh). It starts
`num_gpus * num_per_gpu` resident inference servers (one port each), then runs the sim tasks
in a **queue-scheduled** fashion against free slots — finishing a task frees its slot and
starts the next. All flags below also accept the equivalent environment variable
(`MODEL_PATH`, `EVAL_WORKDIR`, `OUTPUT_BASE`, `CONDA_SH`, `QWEN3VL_PATH`, `INFERENCE_ENV`, `SIM_ENV`).

### Prerequisites

1. **RoboTwin repo** cloned and its dependencies installed (steps 1–2 above). The launcher
   needs the repo **root** path (the one containing `envs/`, `assets/`, `task_config/`, `script/`).
2. **Two conda environments**:
   - inference side — e.g. `lingbotvla` (PyTorch + this repo's model code).
   - sim side — `RoboTwin` (sapien / mplib / curobo / open3d …), built against numpy 1.26.x.
3. **Qwen3-VL backbone** checkpoint used by the VLA vision-language encoder (`QWEN3VL_PATH`).
4. **Your trained HF checkpoint** (`model_path`), e.g. `.../global_step_xxxxx/hf_ckpt`.

> [!IMPORTANT]
> Release validation of the published checkpoint uses **FP32 inference**. The lower-memory
> BF16 mode is useful for pipeline checks, but can produce materially different success
> rates.

> The launcher **auto-copies** the eval client (`eval_policy_client_lingbotvla.py` + the small
> `deploy/` helpers) from this repo into `<RoboTwin>/script/`, and **self-heals** the curobo
> embodiment `.yml` and editable-install `.pth` paths to point at the RoboTwin checkout you
> pass. No manual path setup is needed when relocating RoboTwin.

### Run the full benchmark (50 tasks)

> Substitute every `/path/to/...` and the conda env / `conda.sh` path for your machine.

```bash
# from this repo's root (so inference_workdir = current working dir)
QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct \
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
    --model_path     /path/to/your/checkpoint/hf_ckpt \
    --eval_workdir   /path/to/RoboTwin \
    --output_base    /path/to/VLABenchmarkResult \
    --conda_sh       /path/to/miniconda3/etc/profile.d/conda.sh \
    --inference_env  lingbotvla \
    --sim_env        RoboTwin \
    --task_config    demo_clean \
    --num_tasks 50 --num_gpus 4 --num_per_gpu 1
```

Use `--task_config demo_randomized` for the randomized benchmark.

**GPU / concurrency**
- `num_gpus` × `num_per_gpu` = number of concurrent sim slots (one inference server per slot).
- `--num_per_gpu 1` is the safe starting point. In our current software stack, one FP32
  policy server plus its simulator uses roughly 32 GB, so leave additional headroom.
- A 24 GB GPU generally requires BF16 (`--use_bf16 True --use_fp32 False`), which does not reproduce the published FP32 benchmark. Increasing `num_per_gpu` can OOM; the script retries each task up to 3 times, but persistent OOM skips the task.

### Smoke test (1 task, 1 GPU)

Verify the pipeline end-to-end without waiting for the full run:

```bash
QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct \
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
    --model_path   /path/to/your/checkpoint/hf_ckpt \
    --eval_workdir /path/to/RoboTwin \
    --conda_sh     /path/to/miniconda3/etc/profile.d/conda.sh \
    --task_config  demo_clean \
    --num_tasks 1 --num_gpus 1 --num_per_gpu 1
```

The run dir is printed at startup (`Run directory: ...`). You should see `Success rate: N/N =>
...` lines appear in the task log. Each task evaluates **100 episodes**, so even a
single-task smoke takes tens of minutes — kill it once you've seen successes, then launch
the full run.

### Output layout

```
<output_base>/<exp>_<step>k_<task_config>_<timestamp>/
├── stats.txt                 # final per-task table + overall success rate
├── inference_logs/           # one log per inference server / port
├── eval_logs/                # one log per task (per-step progress, success rate)
└── eval_results/             # per-task videos: episodeN_success.mp4 ...
```

### Useful flags

| flag | meaning |
|------|---------|
| `--no_video` | disable per-episode video recording (faster, no videos saved) |
| `--keep_inference` | leave inference servers resident after sim finishes |
| `--start_port` | base port for inference servers (default 9330, slot *i* uses base + i) |
| `--use_length` | action-chunk length forwarded to the policy (default 50) |
| `--robo_name` | robot config name (default `robotwin`) |
| `--task_config` | RoboTwin setting: `demo_clean` or `demo_randomized` |
| `--use_bf16` / `--use_fp32` | inference precision; release reproduction uses `False` / `True` |
| `--use_compile` | enable lazy `torch.compile` (default `True`; first request takes longer) |
| `--inference_script` | inference-side module (default `deploy/lingbot_vla_v2_policy.py`) |

### Monitor / stop

```bash
RUN=/path/to/VLABenchmarkResult/<exp>_<step>k_<timestamp>

# overall progress (done/skip/fail summary)
sed 's/\x1b\[[0-9;]*m//g' $RUN/eval_logs/*.log | grep -E "Success rate" | tail

# per-task latest success rate
for f in $RUN/eval_logs/*.log; do
  printf "%-24s %s\n" "$(basename $f .log)" \
    "$(grep -aoE 'Success rate: [0-9]+/[0-9]+ => [0-9.]+%' "$f" | tail -1)"
done
```

Stop everything with `Ctrl-C` (the launcher traps it and kills all child processes), or kill
the launcher plus any lingering `deploy.lingbot_vla_v2_policy` / `eval_policy_client_lingbotvla.py`
processes.