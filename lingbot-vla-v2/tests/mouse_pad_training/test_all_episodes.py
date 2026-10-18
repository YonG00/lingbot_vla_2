"""全轨迹拟合实验：真实CPU数据/窗口/采样/调度链，无模型或CUDA。"""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import random
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from mouse_pad_training import test_recipe as mouse_fixtures
from lingbotvla.auto_learning.catalog import TaskCatalog
from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.data_audit import audit_mapping
from lingbotvla.auto_learning.orchestration.scheduler import Scheduler
from lingbotvla.auto_learning.ports import Backend, TrainResult
from lingbotvla.auto_learning.resolver import SampleResolver
from lingbotvla.auto_learning.sampling.sampler import BatchSampler
from lingbotvla.auto_learning.sampling.target_change_window import cache_path
from lingbotvla.auto_learning.state.persistence import config_fingerprint
from lingbotvla.auto_learning.state.registry import TaskRegistry
from lingbotvla.auto_learning.types import TrajectoryMetrics
from experiment.robotwin.prepare_target_change_windows import prepare

recipe, ROOT = mouse_fixtures.recipe, mouse_fixtures.ROOT


class AllEpisodesTests(unittest.TestCase):
    def setUp(self):
        base = mouse_fixtures.MouseRecipeTests()
        base.setUp()
        self.addCleanup(base.doCleanups)
        self.fixture, self.args = base.fixture, base.args
        self.args.train_all_episodes = True
        self.args.ee_position_weight = 10.0
        _, self.plan, _ = recipe.build_job(self.args)

    def test_plan_and_real_worker_rebuild_keep_all50_fullvision_and_p10(self):
        job, plan, snapshot = recipe.build_job(self.args)
        self.assertEqual(plan['counts']['place_mouse_pad'], {'train': 50, 'val': 0})
        self.assertEqual(plan['train_ids'], list(range(800, 850)))
        self.assertIsNone(plan['config']['pass_thresholds_file'])
        self.assertFalse(Path(job['output_dir']).exists())
        snapshot.parent.mkdir()
        snapshot.write_text(json.dumps(plan['training_config']))
        al_file = snapshot.with_suffix('.al.json')
        al_file.write_text(json.dumps(plan['config']))
        command = job['command'][2:]
        command[command.index('--al-config') + 1] = str(al_file)
        command[command.index('--execute')] = '--prepare-only'
        with contextlib.redirect_stdout(io.StringIO()):
            recipe.launcher.main(command)
        out = Path(job['output_dir'])
        self.assertEqual(json.loads((out/'train_ids.json').read_text()), list(range(800, 850)))
        actual = json.loads((out/'provenance.json').read_text())
        self.assertTrue(actual['config']['train_all_episodes'])
        cmd = actual['command']
        for key, value in [('ee_fk_position_loss_weight', '10.0'),
                           ('ee_fk_orientation_loss_weight', '0.8'),
                           ('ee_fk_gripper_loss_weight', '1.5'),
                           ('freeze_vit', 'false'), ('freeze_vision_encoder', 'false'),
                           ('train_expert_only', 'false'), ('use_lora', 'false'),
                           ('save_steps', '250'), ('max_steps', '500')]:
            # freeze_vit/use_lora are source snapshot values, retained by the worker.
            if '--train.' + key in cmd:
                self.assertEqual(cmd[cmd.index('--train.' + key)+1], value)
            else:
                self.assertIs(actual['training_config']['train'][key], False)

    def test_explicit_entry_defaults_and_preview_no_writes(self):
        spec = importlib.util.spec_from_file_location('mouse_all50', ROOT/
            'experiment/robotwin/recipes/20261018_place_mouse_pad_all50_9680_ee_lnfm.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with patch.object(module.recipe, 'main', return_value=0) as run:
            self.assertEqual(module.main(['--steps', '750']), 0)
        self.assertEqual(run.call_args.args[0],
                         ['--train-all-episodes', '--ee-position-weight', '10.0', '--steps', '750'])
        self.args.run_name = None
        job, _, _ = recipe.build_job(self.args)
        self.assertIn('all50_p10', job['output_dir'])
        self.assertFalse(Path(job['output_dir']).exists())

    def test_invalid_state_combinations_and_budget_fail_before_writing(self):
        base = self.plan['config']
        for key, value in [('pass_transition_enabled', True), ('task_scan_enabled', True),
                           ('review_after_task_transitions', 1), ('import_pool_state', True),
                           ('early_defer_on_overfit', True), ('defer_resample_retry', True),
                           ('pool_filter_by_gmean_ratio', True), ('max_global_steps', None),
                           ('max_global_steps', 10000), ('active_train_probe_trajs', 3),
                           ('train_all_episodes', 'true')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                AutoLearningConfig(**{**base, key: value})
        self.args.steps = 10000
        with self.assertRaisesRegex(ValueError, 'defer budget'):
            recipe.build_job(self.args)

    def make_data(self, *, all_episodes=True):
        cfg = AutoLearningConfig(**{**self.plan['config'], 'train_all_episodes': all_episodes,
            'pass_thresholds_file': None if all_episodes else
            str(ROOT / 'configs/auto_learning/pass_thresholds_gmean1_active4.json')})
        root = self.args.dataset_root
        arms = [[f'{prefix}_joint{i}' for i in range(1, 7)] for prefix in ('fl', 'fr')]
        robot = self.fixture.root/'robot.yml'
        robot.write_text('arm_joints_name: ' + json.dumps(arms))
        cfg.target_change_embodiment_config = str(robot)
        cfg.target_change_cache_dir = str(self.fixture.root/'window_cache')
        cfg.target_change_dataset_root = str(root.resolve())
        (root/'meta/info.json').write_text(json.dumps({'fps': 15, 'features': {}}))
        original = TaskCatalog.from_manifest(str(self.args.manifest))
        original = TaskCatalog({'place_mouse_pad': original.entry('place_mouse_pad')})
        catalog = original.for_training(train_all_episodes=all_episodes)
        episodes = list(catalog.entry('place_mouse_pad').train_traj_ids)
        rows = []
        for ep in range(800, 850):
            for frame in range(10):
                state = [0.] * 14
                state[6] = max(0., 1. - (frame-2)/4)
                rows.append(dict(index=(ep-800)*10+frame, episode_index=ep,
                    frame_index=frame, timestamp=frame/15, **{'observation.state': state}))
        data = root/'data'; data.mkdir(exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), data/'mouse.parquet')
        selected = [row for row in rows if row['episode_index'] in episodes]
        resolver = SampleResolver(episode_index=[r['episode_index'] for r in selected],
            frame_index=[r['index'] for r in selected], task_of_episode=catalog.task_of_episode)
        catalog = catalog.attach_samples(resolver)
        return cfg, original, catalog, resolver

    def test_all50_cpu_chain_prepares_every_episode_and_only_monitors_train(self):
        cfg, original, catalog, resolver = self.make_data()
        self.assertEqual(original.entry('place_mouse_pad').n_train, 40)
        self.assertEqual(original.entry('place_mouse_pad').n_val, 10)
        self.assertFalse(catalog.verify())
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(cfg, self.args.dataset_root, recipe.launcher.read_json(self.args.manifest),
                    ['place_mouse_pad'])
        cache = json.loads(cache_path(cfg.target_change_cache_dir, 'place_mouse_pad').read_text())
        self.assertEqual(cache['train_episode_ids'], list(range(800, 850)))
        self.assertEqual(cache['summary']['frames'], 500)
        meta = {ep: dict(episode_index=ep, length=10, tasks=['Place mouse on pad.'],
                        dataset_from_index=(ep-800)*10, dataset_to_index=(ep-799)*10)
                for ep in range(800, 850)}
        audit = audit_mapping(catalog, resolver, dataset_root=self.args.dataset_root,
            episode_getter=meta.__getitem__, report_file=self.fixture.root/'audit50.json',
            instruction_patterns={'place_mouse_pad': 'mouse'})
        self.assertEqual(audit['tasks'][0]['train_episodes'], 50)
        self.assertEqual(audit['tasks'][0]['val_episodes'], 0)
        calls = []
        def evaluate(task, split, ids):
            self.assertEqual(split, 'train_monitor')  # val/scout/confirm must never execute
            self.assertEqual(ids, [800, 801, 802, 803])
            calls.append((split, tuple(ids)))
            return TrajectoryMetrics(task, split, list(ids), {i:.01 for i in ids},
                .01, 0., None, None, False, 4, ee_metrics={'position_rmse_mm': 20.})
        batches = BatchSampler(cfg, resolver, catalog, random.Random(3))
        seen = set()
        def train(request, steps):
            self.assertEqual(set(request.probs), set(range(500)))
            compiled = batches.prepare(request)
            for k in range(steps):
                comp = batches.build(compiled, logical_step=request.start_step+k)
                seen.update(ref.episode_id for ref in comp.new)
            return TrainResult(loss=.05, steps=steps, samples_seen=steps*20,
                new_slot_counts={0:steps*20}, batches_built=steps, unique_batches=steps)
        backend = Backend(catalog=catalog, resolver=resolver,
            evaluator=NS(evaluate=evaluate, baseline_mse=lambda _:0.),
            trainer=NS(train_steps=train), scorer=NS(score=Mock(side_effect=AssertionError('no scoring'))))
        logger = Mock()
        sched = Scheduler(backend, cfg, logger=logger)
        sched.run()
        rec = sched.registry.get('place_mouse_pad')
        self.assertEqual(sched.state.global_step, 500)
        self.assertEqual(len(calls), 6)  # step0 and 100/200/300/400/500, one call each
        self.assertEqual(seen, set(range(800, 850)))
        self.assertEqual([row['step'] for row in rec.eval_history], [0,100,200,300,400,500])
        self.assertEqual({row['kind'] for row in rec.eval_history}, {'active_train'})
        self.assertIsNone(rec.current_val_gmean_mse)
        self.assertIsNone(rec.train_val_gap_ratio)
        self.assertFalse(rec.ever_passed)
        self.assertEqual({e['reason'] for e in sched.events if e.get('action')=='train_unit'},
                         {'仅训练集拟合监控，继续至最大训练步数；不判PASS/过拟合'})
        tags = [c.args[1] for c in logger.log_metrics.call_args_list]
        self.assertFalse(any('active_val' in t or 'threshold' in t or 'overfit' in t
                             or 'val_gmean' in t or 'gap' in t for t in tags))
        self.assertIn('task/place_mouse_pad/active_train_ee_position_rmse_mm', tags)

    def test_40_episode_cache_cannot_be_reused_as_50(self):
        cfg, _, _, _ = self.make_data(all_episodes=False)
        manifest = recipe.launcher.read_json(self.args.manifest)
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(cfg, self.args.dataset_root, manifest, ['place_mouse_pad'])
        cfg.train_all_episodes = True
        with self.assertRaisesRegex(ValueError, 'train episodes changed'):
            prepare(cfg, self.args.dataset_root, manifest, ['place_mouse_pad'])

    def test_defaults_keep_fingerprint_and_true_invalidates_resume(self):
        normal = AutoLearningConfig(**{**self.plan['config'], 'train_all_episodes': False,
            'pass_thresholds_file': 'unused.json'})
        self.assertNotIn('train_all_episodes', config_fingerprint(normal, ['place_mouse_pad']))
        full = AutoLearningConfig(**self.plan['config'])
        self.assertIs(config_fingerprint(full, ['place_mouse_pad'])['train_all_episodes'], True)
        original = TaskCatalog.from_manifest(str(self.args.manifest))
        with self.assertRaisesRegex(ValueError, 'all episodes in train'):
            TaskRegistry.from_catalog(original, full)

    def test_open_loop_validator_preserves_explicit_empty_val(self):
        # Execute the actual constructor from production AST in offline mode;
        # no torch, transformers or dataset imports, no implementation copy.
        path = ROOT/'lingbotvla/utils/open_loop_validation.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'OpenLoopValidator')
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__']
        defaults = [n for n in tree.body if isinstance(n, ast.AnnAssign)
                    and isinstance(n.target, ast.Name) and n.target.id in
                    ('DEFAULT_TRAIN_MONITOR_IDS', 'DEFAULT_VAL_IDS')]
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future, *defaults, cls], type_ignores=[]))
        ns = {}
        exec(compile(module, str(path), 'exec'), ns)
        common = dict(model_config=NS(), args=NS(data=NS(image_augment=False)),
                      processor=None, use_depth_align=False, logger=Mock())
        explicit = ns['OpenLoopValidator'](**common, train_monitor_ids=[800,801,802,803], val_ids=[])
        self.assertEqual(explicit.val_ids, [])
        self.assertEqual(explicit.train_monitor_ids, [800,801,802,803])
        default = ns['OpenLoopValidator'](**common)
        self.assertEqual(default.val_ids, ns['DEFAULT_VAL_IDS'])
        self.assertEqual(default.train_monitor_ids, ns['DEFAULT_TRAIN_MONITOR_IDS'])

    def test_panel_reads_real_train_metrics_in_both_views(self):
        path = ROOT/'tools/auto_learning_pipeline/al_panel.py'
        spec = importlib.util.spec_from_file_location('mouse_monitor_panel', path)
        panel = importlib.util.module_from_spec(spec); spec.loader.exec_module(panel)
        run = self.fixture.root/'panel_run'; run.mkdir()
        al = run/'al_config.json'; al.write_text(json.dumps(self.plan['config']))
        (run/'lingbotvla_cli.yaml').write_text(f'train:\n  auto_learning: {al}\n  max_steps: 500\n')
        points = {'active_train_gmean_mse': .01, 'unit_loss': .05,
                  'active_train_ee_position_rmse_mm': 20.,
                  'active_train_ee_orientation_mean_deg': 5.,
                  'active_train_ee_gripper_mae': .02}
        events = [dict(action='select', task='place_mouse_pad', step=0)]
        events += [dict(name=f'task/place_mouse_pad/{key}', value=value, step=100)
                   for key,value in points.items()]
        event_file = run/'auto_learning_events.jsonl'
        event_file.write_text('\n'.join(map(json.dumps, events))+'\n')
        series = panel.read_series(event_file)
        kwargs = dict(run='panel_run', target=1, passed=0, newly=0, boot=0, step=100,
            sit=3., latest_sit=3., elapsed_s=300., eta_s=1200., trainer=None, trainers=[],
            records={'place_mouse_pad': {'status':'TRAINING','overfit':False,'attempt_count':1}},
            thresholds={}, derived_status={'place_mouse_pad':'TRAINING'}, per_task={},
            series=series, dcp_name=None, dcp_step=None, current_task='place_mouse_pad')
        with patch.object(panel, 'ROOT', self.fixture.root):
            cfg = panel.run_cfg('panel_run')
        self.assertTrue(cfg['train_all_episodes'])
        for wide in (False, True):
            with self.subTest(wide=wide), contextlib.redirect_stdout(io.StringIO()) as stream:
                if wide:
                    panel.render_wide(**kwargs, cfg=cfg, args=NS(), derived_passed=[], reg_passed=0)
                else:
                    panel.print_compact(**kwargs, cfg=cfg)
                text = stream.getvalue()
                self.assertIn('train G 0.010000', text)
                self.assertIn('末端train', text)
                for misleading in ['末端val', '未过拟合', '门槛', '倍率', '过拟合 0']:
                    self.assertNotIn(misleading, text)


if __name__ == '__main__':
    unittest.main()
