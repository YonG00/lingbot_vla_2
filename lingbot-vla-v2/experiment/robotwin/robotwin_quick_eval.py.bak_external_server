#!/usr/bin/env python3
import argparse, csv, json, math, os, random, shutil, signal, socket, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen
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

def health_ok(port):
    try:
        with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=.5) as r:
            return r.status == 200
    except Exception:
        return False

def tail(path, n=40):
    try: return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except Exception: return ""

def preflight(args, plan):
    need = [args.model_path, args.qwen_path, args.eval_workdir, args.conda_sh,
            Path(__file__).with_name("eval_policy_client_quick.py")]
    missing = [str(x) for x in need if not Path(x).exists()]
    if missing: raise RuntimeError(f"Missing required paths: {missing}")
    if any(x["setting"] == "randomized" for x in plan):
        bg = args.eval_workdir / "assets/background_texture/unseen"
        if not bg.is_dir() or not any(bg.iterdir()):
            raise RuntimeError(f"Randomized eval requires background textures: {bg}")

def wait_server(proc, port, log_path, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"LingBot server exited early:\n{tail(log_path)}")
        if health_ok(port): return
        time.sleep(1)
    raise TimeoutError(f"LingBot server did not become healthy on port {port} within {timeout}s")

def stop_process(proc):
    if not proc or proc.poll() is not None: return
    try: os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError: return
    try: proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass

def discover_gpus(spec="auto"):
    cmd = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
           "--format=csv,noheader,nounits"]
    rows = []
    for line in subprocess.check_output(cmd, text=True).splitlines():
        x = [z.strip() for z in line.split(",", 3)]
        rows.append({"index": int(x[0]), "name": x[1],
                     "total_mb": int(x[2]), "free_mb": int(x[3])})
    if spec != "auto":
        wanted = {int(x) for x in str(spec).split(",") if x.strip()}
        rows = [x for x in rows if x["index"] in wanted]
        missing = wanted - {x["index"] for x in rows}
        if missing: raise RuntimeError(f"Requested GPUs not found: {sorted(missing)}")
    if not rows: raise RuntimeError("No usable NVIDIA GPU found")
    return rows

def _auto_int(v, name):
    if str(v).lower() == "auto": return None
    try: n = int(v)
    except Exception: raise ValueError(f"{name} must be 'auto' or integer")
    if n <= 0: raise ValueError(f"{name} must be > 0")
    return n

def build_resource_plan(args, plan):
    gpus = discover_gpus(args.gpus)
    tasks = len(plan)
    ranked = sorted(gpus, key=lambda x: (x["free_mb"], x["total_mb"], -x["index"]), reverse=True)

    requested_inf = _auto_int(args.num_inference_gpus, "num_inference_gpus")
    if len(gpus) == 1:
        infer, sim = [ranked[0]], [ranked[0]]
    else:
        if requested_inf is None:
            sim_n = min(tasks, len(gpus) - 1)
            inf_n = len(gpus) - sim_n
        else:
            inf_n = requested_inf
            if inf_n >= len(gpus):
                raise ValueError("num_inference_gpus must leave at least one simulation GPU")
        infer, sim = ranked[:inf_n], ranked[inf_n:]

    bad_inf = [x for x in infer if x["free_mb"] < args.inference_min_free_mb]
    if bad_inf:
        raise RuntimeError(
            "Insufficient free VRAM for inference GPU(s): " +
            ", ".join(f"GPU{x['index']}={x['free_mb']}MiB" for x in bad_inf)
        )

    caps = {}
    if len(gpus) == 1:
        caps[sim[0]["index"]] = 1
    else:
        for g in sim:
            usable = max(0, g["free_mb"] - args.sim_reserve_mb)
            caps[g["index"]] = min(args.max_sim_workers_per_gpu,
                                   usable // args.sim_worker_vram_mb)
        if not all(caps.values()):
            raise RuntimeError(f"Simulation GPU capacity is zero: {caps}")

    capacity = sum(caps.values())
    req_parallel = _auto_int(args.parallel, "parallel")
    parallel = min(tasks, capacity, req_parallel or tasks)
    if parallel < 1: raise RuntimeError("No simulation slot available")

    # Spread logical sim slots across GPUs instead of filling GPU0 first.
    sim_slots, used = [], {g["index"]: 0 for g in sim}
    while len(sim_slots) < parallel:
        progressed = False
        for g in sim:
            gid = g["index"]
            if used[gid] < caps[gid] and len(sim_slots) < parallel:
                sim_slots.append(gid); used[gid] += 1; progressed = True
        if not progressed: break

    # LPT scheduling: long jobs first, assign to least-loaded logical slot.
    jobs = [dict(x) for x in plan]
    slot_cost = [0] * len(sim_slots)
    slot_jobs = [[] for _ in sim_slots]
    for x in sorted(jobs, key=lambda z: z["max_steps"] * z["trials"], reverse=True):
        i = min(range(len(sim_slots)), key=lambda k: slot_cost[k])
        cost = x["max_steps"] * x["trials"]
        x["_logical_slot"] = i
        x["_sim_gpu"] = sim_slots[i]
        slot_jobs[i].append(x)
        slot_cost[i] += cost

    # Balance inference demand among resident servers.
    server_cost = [0] * len(infer)
    for x in sorted(jobs, key=lambda z: z["max_steps"] * z["trials"], reverse=True):
        i = min(range(len(infer)), key=lambda k: server_cost[k])
        x["_server_idx"] = i
        x["_infer_gpu"] = infer[i]["index"]
        server_cost[i] += x["max_steps"] * x["trials"]

    # slot_jobs currently contains the same mutable dict objects as jobs.
    return {
        "gpus": gpus,
        "inference_gpus": [x["index"] for x in infer],
        "simulation_gpus": [x["index"] for x in sim],
        "sim_capacity": caps,
        "parallel_tasks": parallel,
        "sim_slots": sim_slots,
        "slot_jobs": slot_jobs,
        "jobs": jobs,
    }

def print_resource_plan(r, args):
    print("\n" + "=" * 96)
    print("Quick Eval Multi-GPU Resource Plan")
    print("=" * 96)
    for g in r["gpus"]:
        print(f"GPU{g['index']}: {g['name']} | total={g['total_mb']} MiB | free={g['free_mb']} MiB")
    print(f"\nInference GPUs : {r['inference_gpus']}")
    print(f"Simulation GPUs: {r['simulation_gpus']}")
    print(f"Sim capacity   : {r['sim_capacity']}")
    print(f"Parallel tasks : {r['parallel_tasks']}")
    print(f"Max batch      : {args.max_batch} | batch wait={args.batch_wait_ms:g}ms")
    print("\nAssignments:")
    for x in sorted(r["jobs"], key=lambda z: z["slot"]):
        print(f"  {x['task']:<28} sim=GPU{x['_sim_gpu']}  infer=GPU{x['_infer_gpu']}  "
              f"port={args.port + x['_server_idx']}  trials={x['trials']}  max_steps={x['max_steps']}")
    print("=" * 96 + "\n")

def run_task_job(args, stage, x, run_dir, result_dir, server_specs, eval_id, model_label):
    task, cfg = x["task"], x["task_config"]
    srv = server_specs[x["_server_idx"]]
    task_log = run_dir / "logs" / f"{x['slot']:02d}_{task}_{x['setting']}.log"

    cmd = (
        f"source '{args.conda_sh}' && conda activate {args.sim_env} && "
        f"cd '{args.eval_workdir}' && "
        f"export CUDA_VISIBLE_DEVICES={x['_sim_gpu']} && "
        f"export PYTHONPATH=\"$(python -c 'import site;print(site.getsitepackages()[0])')${{PYTHONPATH:+:$PYTHONPATH}}\" && "
        f"PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 "
        f"python -u script/eval_policy_client_quick.py --config policy/ACT/deploy_policy.yml --overrides "
        f"--task_name {task} --task_config {cfg} --seed {args.seed} --policy_name lingbotvla "
        f"--ckpt_setting {stage} --port {srv['port']} --robo_name robotwin --test_num {x['trials']} "
        f"--max_steps {x['max_steps']} --eval_video_log False --output_dir '{result_dir}'"
    )

    print(f"[start] {task} | sim=GPU{x['_sim_gpu']} | infer=GPU{srv['gpu']}:{srv['port']} "
          f"| {x['setting']} | {x['trials']}x{x['max_steps']}")
    t0 = time.time()
    with open(task_log, "w") as lf:
        rc = subprocess.run(["bash", "-lc", cmd], stdout=lf,
                            stderr=subprocess.STDOUT).returncode
    duration = time.time() - t0

    if rc != 0: raise RuntimeError(f"{task} worker failed:\n{tail(task_log)}")
    result_file = result_dir / task / "_result.txt"
    if not result_file.is_file():
        raise RuntimeError(f"Missing result file for {task}: {result_file}")

    rate = float([z.strip() for z in result_file.read_text().splitlines() if z.strip()][-1])
    success = int(round(rate * x["trials"]))
    print(f"[done ] {task} -> {success}/{x['trials']}={rate:.3f} | {duration/60:.1f} min")

    return {
        **x, "eval_id": eval_id, "model_label": model_label, "stage": stage,
        "success": success, "rate": rate, "duration_sec": round(duration, 2),
        "sim_gpu": x["_sim_gpu"], "inference_gpu": srv["gpu"],
        "server_port": srv["port"],
    }

def run_eval(args, stage, plan, eval_id):
    preflight(args, plan)
    resources = build_resource_plan(args, plan)
    print_resource_plan(resources, args)

    for i in range(len(resources["inference_gpus"])):
        if port_open(args.port + i):
            raise RuntimeError(f"Port {args.port+i} is already in use")

    wall_start = time.time()
    started_at = datetime.now().isoformat(timespec="seconds")
    run_dir = args.output_dir / "runs" / f"{eval_id}_{stage}_seed{args.seed}"
    log_dir, result_dir = run_dir / "logs", run_dir / "eval_results"
    log_dir.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)

    src = Path(__file__).with_name("eval_policy_client_quick.py")
    shutil.copy2(src, args.eval_workdir / "script/eval_policy_client_quick.py")

    model_label = args.model_label or args.model_path.parent.name
    root = Path(__file__).resolve().parents[2]
    base_env = os.environ.copy()
    base_env["QWEN3VL_PATH"] = str(args.qwen_path)
    base_env["SETUPTOOLS_SCM_PRETEND_VERSION"] = "0.0.0"
    base_env["PYTHONUNBUFFERED"] = "1"

    servers, handles = [], []
    load_start = time.time()

    try:
        # Start all inference servers concurrently, one model copy per inference GPU.
        for i, gpu in enumerate(resources["inference_gpus"]):
            port = args.port + i
            log_path = log_dir / f"inference_gpu{gpu}_port{port}.log"
            cmd = (
                f"source '{args.conda_sh}' && conda activate {args.inference_env} && "
                f"cd '{root}' && "
                f"python -u -m deploy.lingbot_vla_v2_batch_policy "
                f"--model_path '{args.model_path}' --use_length {args.use_length} "
                f"--max_batch {args.max_batch} --batch_wait_ms {args.batch_wait_ms} "
                f"--use_bf16 true --use_fp32 false --use_compile false --port {port}"
            )
            env = base_env.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            fh = open(log_path, "w"); handles.append(fh)
            proc = subprocess.Popen(["bash", "-lc", cmd], stdout=fh,
                                    stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True)
            servers.append({"gpu": gpu, "port": port, "proc": proc,
                            "log": log_path, "started": time.time()})
            print(f"Starting inference server: GPU{gpu} :{port}")

        for srv in servers:
            wait_server(srv["proc"], srv["port"], srv["log"])
            srv["load_sec"] = round(time.time() - srv["started"], 2)
            print(f"  ready GPU{srv['gpu']}:{srv['port']} load={srv['load_sec']:.1f}s")

        model_load_wall_sec = time.time() - load_start
        rollout_start = time.time()

        # Each logical slot is serial; slots themselves run concurrently.
        def run_slot(slot_id):
            out = []
            for x in resources["slot_jobs"][slot_id]:
                out.append(run_task_job(args, stage, x, run_dir, result_dir,
                                        servers, eval_id, model_label))
            return out

        rows = []
        with ThreadPoolExecutor(max_workers=resources["parallel_tasks"]) as pool:
            futures = [pool.submit(run_slot, i)
                       for i in range(resources["parallel_tasks"])]
            for f in as_completed(futures):
                rows.extend(f.result())

        rollout_wall_sec = time.time() - rollout_start

    finally:
        for srv in servers: stop_process(srv["proc"])
        for fh in handles: fh.close()

    rows.sort(key=lambda x: x["slot"])
    csv_path = run_dir / "quick_eval.csv"
    fields = [
        "eval_id","model_label","stage","slot","task","level","sample_type","setting",
        "success","trials","rate","dataset_max_steps","max_steps","duration_sec",
        "sim_gpu","inference_gpu","server_port"
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    total_success = sum(x["success"] for x in rows)
    total_trials = sum(x["trials"] for x in rows)
    total_wall_sec = time.time() - wall_start

    resource_summary = {
        "gpu_inventory": resources["gpus"],
        "inference_gpus": resources["inference_gpus"],
        "simulation_gpus": resources["simulation_gpus"],
        "sim_capacity": resources["sim_capacity"],
        "parallel_tasks": resources["parallel_tasks"],
        "max_batch": args.max_batch,
        "batch_wait_ms": args.batch_wait_ms,
        "assignments": [
            {"task": x["task"], "sim_gpu": x["_sim_gpu"],
             "inference_gpu": x["_infer_gpu"],
             "server_port": args.port + x["_server_idx"]}
            for x in sorted(resources["jobs"], key=lambda z: z["slot"])
        ],
    }

    summary = {
        "eval_id": eval_id, "model_label": model_label, "stage": stage,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "resource_plan": resource_summary,
        "inference_servers": [
            {"gpu": x["gpu"], "port": x["port"], "load_sec": x["load_sec"]}
            for x in servers
        ],
        "model_load_wall_sec": round(model_load_wall_sec, 2),
        "rollout_wall_sec": round(rollout_wall_sec, 2),
        "task_duration_sum_sec": round(sum(x["duration_sec"] for x in rows), 2),
        "total_wall_sec": round(total_wall_sec, 2),
        "success": total_success, "trials": total_trials,
        "rate": total_success / total_trials,
    }

    summary_path = run_dir / "quick_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(f"\nQuick Eval finished: {total_success}/{total_trials} = {total_success/total_trials:.3f}")
    print(f"Parallel tasks: {resources['parallel_tasks']}")
    print(f"Model load wall: {model_load_wall_sec:.1f}s")
    print(f"Rollout wall: {rollout_wall_sec/60:.1f} min")
    print(f"Total wall: {total_wall_sec/60:.1f} min")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")

def main():
    p = argparse.ArgumentParser(description="LingBot-VLA-v2 RoboTwin Quick Eval")
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--horizon_scale", type=float, default=1.15)
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
    p.add_argument("--gpus", default="auto", help="auto or physical GPU list, e.g. 0,1,2,3")
    p.add_argument("--parallel", default="auto", help="auto or max concurrent task jobs")
    p.add_argument("--num_inference_gpus", default="auto", help="auto or integer")
    p.add_argument("--max_sim_workers_per_gpu", type=int, default=4)
    p.add_argument("--sim_worker_vram_mb", type=int, default=4500)
    p.add_argument("--sim_reserve_mb", type=int, default=4096)
    p.add_argument("--inference_min_free_mb", type=int, default=20000)
    p.add_argument("--max_batch", type=int, default=4)
    p.add_argument("--batch_wait_ms", type=float, default=20)
    p.add_argument("--model_label", default=None)
    p.add_argument("--show_resources", action="store_true")
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
        "protocol": "robotwin_quick_eval_v2_multigpu", "eval_id": eval_id, "stage": args.stage,
        "current_level": level, "sampling_seed": args.seed, "trials_per_task": args.trials,
        "num_tasks": 4, "num_rollouts": 4 * args.trials, "horizon_scale": args.horizon_scale,
        "curriculum_source": str(args.curriculum), "manifest_source": str(args.manifest), "plan": plan,
    }
    plan_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print_plan(args.stage, level, args.seed, args.trials, args.horizon_scale, plan, plan_path)
    if args.show_resources and not args.run:
        print_resource_plan(build_resource_plan(args, plan), args)
    if args.run:
        if args.model_path is None: raise ValueError("--model_path is required with --run")
        run_eval(args, args.stage, plan, eval_id)

if __name__ == "__main__":
    main()
