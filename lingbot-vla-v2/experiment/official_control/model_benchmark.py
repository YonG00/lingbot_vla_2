"""Unified single-task / whole-model evaluation with configurable parallel episodes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONFIGS = ('demo_clean', 'demo_randomized')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def plan(config):
    tasks = config['tasks']
    if not isinstance(tasks, list) or not tasks or len(set(tasks)) != len(tasks):
        raise ValueError('unique nonempty tasks required')
    if any(not isinstance(t, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', t) for t in tasks):
        raise ValueError('invalid task name')
    configs = config.get('task_configs', list(CONFIGS))
    if not isinstance(configs, list) or not configs or len(set(configs)) != len(configs) or any(c not in CONFIGS for c in configs):
        raise ValueError('invalid task_configs')
    n = config.get('episodes_per_task', 3)
    workers = config.get('episode_workers', 3)
    videos = config.get('video_episodes', 0)
    if type(n) is not int or n < 1 or type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('invalid episodes/workers')
    if type(videos) is not int or not 0 <= videos <= n:
        raise ValueError('invalid video_episodes')
    if config.get('expected_task_count', 50) != len(tasks):
        raise ValueError('task count differs from expected_task_count (default 50)')
    if config.get('seed_selection','inline') not in ('inline','expert','fixed'):
        raise ValueError('invalid seed_selection')
    for key, default, minimum in [('seed_search_start',100000,0),('seed_search_max_candidates',200,n),('seed_selection_timeout_seconds',1800,1)]:
        value=config.get(key,default)
        if type(value) is not int or value < minimum:
            raise ValueError('invalid '+key)
    seeds = config.get('scene_seeds', {})
    valid_keys = {c + '/' + t for c in configs for t in tasks}
    if not isinstance(seeds, dict) or set(seeds) - valid_keys:
        raise ValueError('unknown scene_seeds keys')
    items = []
    for c in configs:
        for task in tasks:
            key = c + '/' + task
            values = seeds.get(key, list(range(100000, 100000 + n)))
            if not isinstance(values, list) or len(values) != n or any(type(s) is not int or s < 0 for s in values) or len(set(values)) != n:
                raise ValueError('unique scene seeds required: ' + key)
            items.append(dict(key=key, task=task, task_config=c, seeds=values,
                              episodes=n, video_episodes=videos, workers=min(n, workers)))
    return items


def fingerprint(config, items):
    """Bind resume to effective settings and immutable checkpoint file metadata."""
    model = Path(config['model']).resolve()
    files = sorted(p for p in model.iterdir() if p.is_file())
    if not (model / 'config.json').is_file() or not any(p.suffix == '.safetensors' for p in files):
        raise ValueError('HF checkpoint required')
    snapshot = dict(config=config, items=items, model=str(model), files=[
        dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns,
             sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.suffix == '.json' else None)
        for p in files])
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()


def jobs_for(config, item, destination):
    from experiment.official_control.episode_parallel import split_jobs
    job = {key: config[key] for key in ('model', 'robotwin', 'qwen', 'sim_python')}
    job.update(task=item['task'], task_config=item['task_config'], label='model_benchmark',
               episodes=item['episodes'], video_episodes=item['video_episodes'],
               destination=str(destination), require_baseline=False)
    for key in ('render_sync', 'temporal_ensemble'):
        if key in config:
            job[key] = config[key]
    # Always shard even with one worker: explicit scene identities remain unchanged.
    shards = split_jobs([job], dict(single_task_episode_workers=2,
        single_task_episode_seeds=item['seeds']))
    if config.get('seed_selection','inline')=='inline' and item['key'] not in config.get('scene_seeds',{}):
        start=config.get('seed_search_start',100000)
        for i,job in enumerate(shards):
            job['scene_seed']=start+i;job['expected_seeds']=[start+i]
            job['seed_search']=dict(stride=item['episodes'],max_candidates=config.get('seed_search_max_candidates',200))
    return shards


def report(items, rows):
    groups = {}
    for name in [*CONFIGS, 'overall']:
        expected = [i for i in items if name == 'overall' or i['task_config'] == name]
        found = [r for r in rows if name == 'overall' or r['task_config'] == name]
        complete = [r for r in found if r['state'] == 'complete']
        episodes = sum(r['episodes'] for r in complete)
        successes = sum(r['successes'] for r in complete)
        groups[name] = dict(expected_items=len(expected), completed_items=len(complete),
            error_items=sum(r['state'] == 'error' for r in found),
            pending_items=len(expected)-len(found), evaluated_episodes=episodes,
            expected_episodes=sum(i['episodes'] for i in expected), successes=successes,
            success_rate=successes / episodes if episodes else None,
            rate_scope='completed_items_only')
    finished = len(rows) == len(items) and all(r['state'] == 'complete' for r in rows)
    return dict(schema_version=1, state='complete' if finished else 'incomplete',
                groups=groups, items=rows)



def reference_results(items, rows, elapsed_seconds):
    """Serialize the reference collect_results arithmetic; its script defines no JSON schema.

    Keep successes / episode_count and percentage (0-100), matching the reference
    Final success rate and overall success output. Runtime audit lives separately.
    """
    by_key = {row['key']: row for row in rows}
    tasks = []
    for item in items:
        row = by_key.get(item['key'])
        complete = row is not None and row['state'] == 'complete'
        count = row['episodes'] if complete else 0
        successes = row['successes'] if complete else 0
        if complete and (type(successes) is not int or not 0 <= successes <= count
                         or count != item['episodes']):
            raise ValueError('invalid task episode counts: ' + item['key'])
        tasks.append(dict(task_config=item['task_config'], task=item['task'],
            status='complete' if complete else ('error' if row else 'pending'),
            successes=successes if complete else None,
            episode_count=count if complete else None,
            success_rate_percent=100 * successes / count if count else None,
            elapsed_seconds=row.get('elapsed_seconds') if row else None,
            result_path=row.get('summary_path', row.get('evidence_dir')) if row else None,
            error=row.get('error') if row else None,
            seeds=row.get('seeds') if row else None))
    def collect(selected):
        complete = [row for row in selected if row['status'] == 'complete']
        # Same sum(successes) / sum(episodes) as upstream collect_results().
        successes = sum(row['successes'] for row in complete)
        count = sum(row['episode_count'] for row in complete)
        return dict(successes=successes, episode_count=count,
            success_rate_percent=100 * successes / count if count else None,
            expected_task_count=len(selected), completed_task_count=len(complete),
            error_task_count=sum(row['status']=='error' for row in selected),
            pending_task_count=sum(row['status']=='pending' for row in selected))
    complete = all(row['status']=='complete' for row in tasks)
    return dict(schema_version=2, status='complete' if complete else 'incomplete',
        rate_scope='all_tasks' if complete else 'completed_tasks_only',
        elapsed_seconds=elapsed_seconds, overall=collect(tasks),
        task_configs={name:collect([row for row in tasks if row['task_config']==name])
                      for name in CONFIGS if any(row['task_config']==name for row in tasks)},
        tasks=tasks)


def run(config, resume=False, opener=None, runner=None, stopper=None, selector=None):
    from experiment.official_control.shared_evaluation import open_session
    from experiment.official_control.parallel_evaluation import run_jobs
    from experiment.official_control.evaluate import stop
    opener = opener or (lambda job, directory: open_session(job, directory,
        server_entry=ROOT / 'experiment/robotwin/session_eval_server.py'))
    from experiment.official_control.expert_seed_selection import select_seeds
    selector = selector or select_seeds
    runner = runner or run_jobs
    stopper = stopper or stop
    items = plan(config)
    identity = fingerprint(config, items)
    directory = Path(config['destination'])
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'manifest.json'
    if manifest.exists():
        if not resume or json.loads(manifest.read_text())['fingerprint'] != identity:
            raise ValueError('resume requires identical checkpoint/settings manifest')
    elif any(directory.iterdir()):
        raise ValueError('nonempty destination without manifest')
    else:
        atomic_json(manifest, dict(fingerprint=identity, config=config, items=items))
    archive = directory / 'benchmark_audit.json'
    # Accept the former internal report on resume; public results now have a stable export.
    legacy = directory / 'benchmark_results.json'
    source = archive if archive.exists() else legacy
    old = json.loads(source.read_text()) if source.exists() else dict(items=[])
    if 'items' not in old:
        raise ValueError('resume audit missing; cannot restore from public results alone')
    complete = {row['key']: row for row in old['items'] if row['state'] == 'complete'}
    rows = list(complete.values())
    session = None
    invocation_start = time.monotonic()
    def publish():
        result = report(items, rows)
        result.update(model=config['model'], fingerprint=identity,
            inference='temporal_ensemble' if 'temporal_ensemble' in config else 'chunk',
            episodes_per_task=config.get('episodes_per_task',3),
            episode_workers=config.get('episode_workers',3),
            video_episodes=config.get('video_episodes',0))
        atomic_json(archive, result)
        public = reference_results(items, rows, time.monotonic()-invocation_start)
        public.update(model=config['model'], episodes_per_task=config.get('episodes_per_task',3),
                      episode_workers=config.get('episode_workers',3), video_episodes=config.get('video_episodes',0))
        atomic_json(directory / 'benchmark_results.json', public)
        atomic_json(directory / 'selected_scene_seeds.json', {row['key']:row['seeds'] for row in rows if row['state']=='complete'})
    try:
        for item in items:
            if item['key'] in complete:
                row = complete[item['key']]
                # Completion markers alone cannot authorize a resume skip.
                summary = json.loads(Path(row['summary_path']).read_text())
                if (summary['state'] != 'complete' or summary['task'] != item['task']
                    or summary.get('task_config') != item['task_config']
                    or summary['seeds'] != row['seeds'] or len(row['seeds']) != item['episodes']
                    or summary['model'] != config['model']
                    or summary['successes'] != row['successes']):
                    raise ValueError('resume summary mismatch: ' + item['key'])
                publish(); continue
            parent = directory / item['task_config'] / item['task']
            attempt = 1
            while (parent / f'attempt_{attempt:03d}').exists():
                attempt += 1
            output = parent / f'attempt_{attempt:03d}'
            output.mkdir(parents=True)
            start = time.monotonic()
            try:
                atomic_json(directory/'current_item.json', dict(state='selecting_expert_seeds',
                    key=item['key'], task=item['task'], task_config=item['task_config']))
                selected = selector(config,item,directory)
                if len(selected)!=item['episodes'] or len(set(selected))!=len(selected):
                    raise ValueError('seed selector did not return unique complete episodes')
                item = dict(item,seeds=selected)
                label='BENCHMARK_SCENE_CANDIDATES' if config.get('seed_selection','inline')=='inline' else 'BENCHMARK_VALID_SEEDS'
                print(label+' '+json.dumps(dict(key=item['key'],seeds=selected)),flush=True)
                jobs = jobs_for(config, item, output / 'episodes_result')
                if session is None:
                    # One model service for the whole benchmark; expert selection is separate.
                    session_dir = directory / 'sessions' / str(time.time_ns())
                    session_dir.mkdir(parents=True)
                    session = opener(jobs[0], session_dir)
                atomic_json(directory/'current_item.json',dict(state='evaluating',key=item['key'],
                    task=item['task'],task_config=item['task_config'],seeds=selected,workers=item['workers']))
                summaries = runner(jobs, output, item['workers'], opener=lambda *_: session,
                                   stopper=lambda _: None, scheduling='rolling')
                if len(summaries) != 1:
                    raise ValueError('expected one aggregated task result')
                summary = summaries[0]
                if config.get('seed_selection','inline')=='inline' and item['key'] not in config.get('scene_seeds',{}):
                    if len(summary['seeds'])!=item['episodes'] or len(set(summary['seeds']))!=item['episodes']:
                        raise ValueError('incomplete inline valid scenes')
                    item=dict(item,seeds=summary['seeds'])
                if (summary.get('task_config') != item['task_config'] or summary['seeds'] != item['seeds']
                    or summary['episodes'] != item['episodes'] or summary['state'] != 'complete'):
                    raise ValueError('task summary identity mismatch')
                rows.append(dict(key=item['key'], task=item['task'], task_config=item['task_config'],
                    state='complete', episodes=summary['episodes'], successes=summary['successes'],
                    success_rate=summary['success_rate'], seeds=summary['seeds'],
                    episode_results=summary['episode_results'], elapsed_seconds=time.monotonic()-start,
                    summary_path=str(output / 'episodes_result' / 'summary.json')))
            except Exception as error:
                rows.append(dict(key=item['key'], task=item['task'], task_config=item['task_config'],
                    state='error', error=repr(error), seeds=item['seeds'],
                    elapsed_seconds=time.monotonic()-start, evidence_dir=str(output)))
                if session is not None and session['process'].poll() is not None:
                    raise RuntimeError('shared inference server exited') from error
            finally:
                publish()
        return report(items, rows)['state'] == 'complete'
    finally:
        if session is not None:
            stopper(session['process'])
        publish()


def configure(config, mode=None, task=None, episodes=None, workers=None, task_config=None):
    config = dict(config)
    mode = mode or config.get('mode', 'benchmark')
    if mode not in ('single', 'benchmark'):
        raise ValueError('invalid evaluation mode')
    config['mode'] = mode
    config.setdefault('seed_selection','inline')
    if mode == 'single':
        task = task or config.get('task')
        if not task:
            raise ValueError('single mode requires --task or job.task')
        config.update(task=task, tasks=[task], expected_task_count=1)
        config.setdefault('task_configs', ['demo_clean'])
    elif task is not None:
        raise ValueError('--task requires single mode')
    else:
        config['expected_task_count'] = 50
    if task_config is not None:
        config['task_configs'] = list(CONFIGS) if task_config == 'both' else [task_config]
    if episodes is not None:
        config['episodes_per_task'] = episodes
    if workers is not None:
        config['episode_workers'] = workers
    plan(config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--mode', choices=('single','benchmark'))
    parser.add_argument('--task')
    parser.add_argument('--episodes', type=int, help='episodes per task/config (default 3)')
    parser.add_argument('--episode-workers', type=int, help='parallel episode limit, 1-8 (default 3)')
    parser.add_argument('--task-config', choices=('both', *CONFIGS))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = configure(json.loads(args.job.read_text()), args.mode, args.task,
        args.episodes, args.episode_workers, args.task_config)
    items = plan(config)
    if args.worker:
        status = Path(config['destination'] + '_pipeline') / 'status.json'
        try:
            atomic_json(status, dict(state='evaluating'))
            ok = run(config, resume=args.resume)
            atomic_json(status, dict(state='complete' if ok else 'incomplete',
                                     report=config['destination']+'/benchmark_results.json'))
            return 0 if ok else 1
        except BaseException as error:
            atomic_json(status, dict(state='failed', error=repr(error)))
            raise
    print(json.dumps(dict(items=len(items), episodes=sum(i['episodes'] for i in items),
        episodes_per_task=config.get('episodes_per_task',3),
        workers=config.get('episode_workers',3), video_episodes=config.get('video_episodes',0)), indent=2))
    if args.execute:
        launch = Path(config['destination'] + '_pipeline')
        launch.mkdir(parents=True, exist_ok=args.resume)
        # Do not permit two writers / model sessions for the same benchmark.
        lock = launch / 'launch.lock'
        import fcntl
        with lock.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            effective = launch / ('job_' + str(time.time_ns()) + '.json')
            atomic_json(effective, config)
            worker_args = [sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(effective),'--worker']
            if args.resume:
                worker_args.append('--resume')
            handle.flush()
            with (launch/'pipeline.log').open('a') as log:
                process = subprocess.Popen(['setsid','nohup',*worker_args], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    pass_fds=(handle.fileno(),))
            print('MODEL_BENCHMARK_STARTED',process.pid,str(launch))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
