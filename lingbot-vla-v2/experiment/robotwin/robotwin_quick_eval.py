#!/usr/bin/env python3
import argparse, csv, json, math, os, random, shutil, signal, socket, subprocess, time
from datetime import datetime
from pathlib import Path
import pandas as pd
import yaml

DEFAULT_CURRICULUM = Path("/data/code/lingbot-vla-v2/configs/curriculum/robotwin_curriculum_v1.yaml")
DEFAULT_MANIFEST = Path("/data/results/robotwin_dataset_analysis/curriculum_v1/curriculum_manifest_v1.csv")
DEFAULT_OUTPUT = Path("/data/results/robotwin_quick_eval")
LEVELS = ["L1", "L2", "L3", "L4"]
STAGES = ["BASE"] + [f"C{i}" for i in range(1, 9)]

def load_yaml(path):
    if not path.is_file(): raise FileNotFoundError(f"Curriculum config not found: {path}")
    cfg = yaml.safe_load(path.read_text())
    if "skill_levels" not in cfg or "curriculum" not in cfg: raise RuntimeError("Invalid curriculum YAML")
    return cfg

def load_manifest(path):
    if not path.is_file(): raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    required = {"episode_index", "task", "skill_level", "length", "first_stage", "first_stage_num"}
    missing = required - set(df.columns)
    if missing: raise RuntimeError(f"Manifest missing columns: {sorted(missing)}")
    return df

def get_level_tasks(cfg):
    out = {}
    for level in LEVELS:
        item = cfg["skill_levels"][level]
        tasks = list(item["tasks"])
        if len(tasks) != int(item["task_count"]): raise RuntimeError(f"{level} task_count mismatch")
        out[level] = tasks
    all_tasks = sum(out.values(), [])
    if len(all_tasks) != 50 or len(set(all_tasks)) != 50: raise RuntimeError("Expected 50 unique curriculum tasks")
    return out

def validate_manifest(df, level_tasks):
    yaml_tasks = set(sum(level_tasks.values(), []))
    manifest_tasks = set(df["task"].unique())
    if yaml_tasks != manifest_tasks:
        raise RuntimeError(f"YAML/manifest mismatch: missing={sorted(yaml_tasks-manifest_tasks)}, extra={sorted(manifest_tasks-yaml_tasks)}")
    counts = df.groupby("task").size()
    if not (counts == 50).all(): raise RuntimeError(f"Expected 50 episodes/task:\n{counts[counts != 50]}")
    yaml_level = {task: level for level, tasks in level_tasks.items() for task in tasks}
    manifest_level = df.groupby("task")["skill_level"].first().to_dict()
    bad = {t: (yaml_level[t], manifest_level[t]) for t in yaml_tasks if yaml_level[t] != manifest_level[t]}
    if bad: raise RuntimeError(f"Skill-level mismatch: {bad}")

def current_level(stage, cfg):
    return None if stage == "BASE" else cfg["curriculum"][stage]["add"]["level"]

def sample_tasks(stage, level, level_tasks, rng):
    if stage == "BASE":
        return [{"task": rng.choice(level_tasks[l]), "sample_type": "base_level"} for l in LEVELS]
    n = int(level[1])
    if n == 1:
        return [{"task": t, "sample_type": "current"} for t in rng.sample(level_tasks[level], 4)]
    current = rng.sample(level_tasks[level], 3)
    history_pool = sum([level_tasks[f"L{i}"] for i in range(1, n)], [])
    return [{"task": t, "sample_type": "current"} for t in current] + [{"task": rng.choice(history_pool), "sample_type": "history"}]

def assign_settings(samples, rng):
    clean_idx = set(rng.sample(range(4), 2))
    for i, item in enumerate(samples):
        item["setting"] = "clean" if i in clean_idx else "randomized"
        item["task_config"] = "demo_clean" if i in clean_idx else "demo_randomized"
    return samples

def build_plan(stage, cfg, df, level_tasks, rng, trials, scale):
    level = current_level(stage, cfg)
    samples = assign_settings(sample_tasks(stage, level, level_tasks, rng), rng)
    max_len = df.groupby("task")["length"].max().astype(int).to_dict()
    task_level = {task: l for l, tasks in level_tasks.items() for task in tasks}

    plan = []
    for i, item in enumerate(samples, 1):
        task = item["task"]
        plan.append({
            **item, "slot": i, "level": task_level[task], "trials": trials,
            "dataset_max_steps": max_len[task], "horizon_scale": scale,
            "max_steps": math.ceil(max_len[task] * scale),
        })
    return level, plan

def validate_plan(plan):
    if len(plan) != 4 or len({x["task"] for x in plan}) != 4: raise RuntimeError("Quick Eval must contain 4 unique tasks")
    clean = sum(x["setting"] == "clean" for x in plan)
    random_n = sum(x["setting"] == "randomized" for x in plan)
    if (clean, random_n) != (2, 2): raise RuntimeError("Quick Eval must be exactly 2 clean + 2 randomized")

def print_plan(stage, level, seed, trials, scale, plan, path):
    print("\n" + "=" * 96)
    print("RoboTwin Quick Eval V1 - PLAN ONLY")
    print("=" * 96)
    print(f"stage={stage}  current_level={level or 'N/A'}  seed={seed}  trials/task={trials}  rollouts={len(plan)*trials}  horizon_scale={scale}\n")
    print(f"{'#':<3}{'task':<30}{'level':<7}{'type':<12}{'setting':<13}{'data_max':>10}{'max_steps':>12}")
    print("-" * 96)
    for x in plan:
        print(f"{x['slot']:<3}{x['task']:<30}{x['level']:<7}{x['sample_type']:<12}{x['setting']:<13}{x['dataset_max_steps']:>10}{x['max_steps']:>12}")
    print("-" * 96)
    print(f"plan saved: {path}")
    print("PLAN ONLY: no LingBot server or RoboTwin rollout was started.\n")


def port_open(port):
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0

def tail(path, n=40):
    try: return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except Exception: return ""

def preflight(args, plan):
    need = [args.model_path, args.qwen_path, args.eval_workdir, args.conda_sh, Path(__file__).with_name("eval_policy_client_quick.py")]
    missing = [str(x) for x in need if not Path(x).exists()]
    if missing: raise RuntimeError(f"Missing required paths: {missing}")
    if port_open(args.port): raise RuntimeError(f"Port {args.port} is already in use")
    if any(x["setting"] == "randomized" for x in plan):
        bg = args.eval_workdir / "assets/background_texture/unseen"
        if not bg.is_dir() or not any(bg.iterdir()): raise RuntimeError(f"Randomized eval requires background textures: {bg}")

def wait_server(proc, port, log_path, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None: raise RuntimeError(f"LingBot server exited early:\n{tail(log_path)}")
        if port_open(port): return
        time.sleep(1)
    raise TimeoutError(f"LingBot server did not open port {port} within {timeout}s")

def stop_process(proc):
    if not proc or proc.poll() is not None: return
    try: os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError: return
    try: proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass

def run_eval(args, stage, plan, eval_id):
    preflight(args, plan)
    wall_start = time.time()
    started_at = datetime.now().isoformat(timespec="seconds")
    run_dir = args.output_dir / "runs" / f"{eval_id}_{stage}_seed{args.seed}"
    log_dir, result_dir = run_dir / "logs", run_dir / "eval_results"
    log_dir.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)

    src = Path(__file__).with_name("eval_policy_client_quick.py")
    worker = args.eval_workdir / "script/eval_policy_client_quick.py"
    shutil.copy2(src, worker)

    env = os.environ.copy()
    env["QWEN3VL_PATH"] = str(args.qwen_path)
    env["SETUPTOOLS_SCM_PRETEND_VERSION"] = "0.0.0"

    server_log = log_dir / "inference.log"
    server_cmd = (
        f"source '{args.conda_sh}' && conda activate {args.inference_env} && "
        f"cd '{Path(__file__).resolve().parents[2]}' && "
        f"python -m deploy.lingbot_vla_v2_policy --model_path '{args.model_path}' "
        f"--use_length {args.use_length} --use_bf16 True --use_fp32 False --use_compile False --port {args.port}"
    )

    print(f"\nStarting LingBot server on :{args.port} ...")
    fh = open(server_log, "w")
    server = subprocess.Popen(["bash", "-lc", server_cmd], stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)

    rows = []
    try:
        load_start = time.time()
        wait_server(server, args.port, server_log)
        server_load_sec = time.time() - load_start
        print(f"LingBot server ready. load={server_load_sec:.1f}s\n")

        for i, x in enumerate(plan, 1):
            task, cfg = x["task"], x["task_config"]
            task_log = log_dir / f"{i:02d}_{task}_{x['setting']}.log"
            cmd = (
                f"source '{args.conda_sh}' && conda activate {args.sim_env} && "
                f"cd '{args.eval_workdir}' && "
                f"export PYTHONPATH=\"$(python -c 'import site;print(site.getsitepackages()[0])')${{PYTHONPATH:+:$PYTHONPATH}}\" && "
                f"PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 "
                f"python -u script/eval_policy_client_quick.py --config policy/ACT/deploy_policy.yml --overrides "
                f"--task_name {task} --task_config {cfg} --seed {args.seed} --policy_name lingbotvla "
                f"--ckpt_setting {stage} --port {args.port} --robo_name robotwin --test_num {x['trials']} "
                f"--max_steps {x['max_steps']} --eval_video_log False --output_dir '{result_dir}'"
            )

            print(f"[{i}/4] {task} | {x['level']} | {x['setting']} | trials={x['trials']} | max_steps={x['max_steps']}")
            task_start = time.time()
            with open(task_log, "w") as lf: rc = subprocess.run(["bash", "-lc", cmd], stdout=lf, stderr=subprocess.STDOUT).returncode
            duration_sec = time.time() - task_start
            if rc != 0: raise RuntimeError(f"{task} worker failed:\n{tail(task_log)}")

            result_file = result_dir / task / "_result.txt"
            if not result_file.is_file(): raise RuntimeError(f"Missing result file for {task}: {result_file}")
            rate = float([z.strip() for z in result_file.read_text().splitlines() if z.strip()][-1])
            success = int(round(rate * x["trials"]))
            rows.append({**x, "eval_id": eval_id, "stage": stage, "success": success, "rate": rate, "duration_sec": round(duration_sec, 2)})
            print(f"      result: {success}/{x['trials']} = {rate:.3f} | time={duration_sec/60:.1f} min")

    finally:
        stop_process(server); fh.close()

    csv_path = run_dir / "quick_eval.csv"
    fields = ["eval_id","stage","slot","task","level","sample_type","setting","success","trials","rate","dataset_max_steps","max_steps","duration_sec"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    total_success = sum(x["success"] for x in rows); total_trials = sum(x["trials"] for x in rows)
    finished_at = datetime.now().isoformat(timespec="seconds")
    total_wall_sec = time.time() - wall_start
    rollout_sec = sum(x["duration_sec"] for x in rows)
    summary = {
        "eval_id": eval_id, "stage": stage, "started_at": started_at, "finished_at": finished_at,
        "server_load_sec": round(server_load_sec, 2), "rollout_wall_sec": round(rollout_sec, 2),
        "total_wall_sec": round(total_wall_sec, 2), "success": total_success,
        "trials": total_trials, "rate": total_success / total_trials,
    }
    summary_path = run_dir / "quick_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nQuick Eval finished: {total_success}/{total_trials} = {total_success/total_trials:.3f}")
    print(f"Total time: {total_wall_sec/60:.1f} min | model load: {server_load_sec:.1f}s | rollouts: {rollout_sec/60:.1f} min")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")

def main():
    p = argparse.ArgumentParser(description="LingBot-VLA-v2 RoboTwin Quick Eval")
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--horizon_scale", type=float, default=1.5)
    p.add_argument("--curriculum", type=Path, default=DEFAULT_CURRICULUM)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--run", action="store_true", help="Execute the generated Quick Eval plan")
    p.add_argument("--model_path", type=Path)
    p.add_argument("--qwen_path", type=Path, default=Path("/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct"))
    p.add_argument("--eval_workdir", type=Path, default=Path("/data/code/RoboTwin-lingbot"))
    p.add_argument("--conda_sh", type=Path, default=Path("/data/miniconda3/etc/profile.d/conda.sh"))
    p.add_argument("--inference_env", default="lingbotvla")
    p.add_argument("--sim_env", default="RoboTwin")
    p.add_argument("--port", type=int, default=9330)
    p.add_argument("--use_length", type=int, default=25)
    args = p.parse_args()

    if args.trials <= 0 or args.horizon_scale <= 0: raise ValueError("trials and horizon_scale must be > 0")

    cfg = load_yaml(args.curriculum)
    df = load_manifest(args.manifest)
    level_tasks = get_level_tasks(cfg)
    validate_manifest(df, level_tasks)

    rng = random.Random(args.seed)
    level, plan = build_plan(args.stage, cfg, df, level_tasks, rng, args.trials, args.horizon_scale)
    validate_plan(plan)

    eval_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_dir = args.output_dir / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / f"{eval_id}_{args.stage}_seed{args.seed}.json"

    payload = {
        "protocol": "robotwin_quick_eval_v1", "eval_id": eval_id, "stage": args.stage,
        "current_level": level, "sampling_seed": args.seed, "trials_per_task": args.trials,
        "num_tasks": 4, "num_rollouts": 4 * args.trials, "horizon_scale": args.horizon_scale,
        "curriculum_source": str(args.curriculum), "manifest_source": str(args.manifest), "plan": plan,
    }
    plan_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print_plan(args.stage, level, args.seed, args.trials, args.horizon_scale, plan, plan_path)
    if args.run:
        if args.model_path is None: raise ValueError("--model_path is required with --run")
        run_eval(args, args.stage, plan, eval_id)

if __name__ == "__main__":
    main()
