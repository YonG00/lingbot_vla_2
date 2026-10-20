import itertools
import random
import unittest
import tempfile
import json
from pathlib import Path

from experiment.paired_control.methods import aligned_weights, make_sampler, validate_flags, validate_pair


class MethodContracts(unittest.TestCase):
    def test_prepare_keeps_frozen_source_and_non_method_parameters(self):
        from experiment.paired_control.prepare import prepare
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);source = root / 'source';source.mkdir()
            config = dict(model={'model_path': '/models/global_step_9680/hf_ckpt'},
                data={'num_workers': 8, 'cameras': ['top', 'left', 'right']},
                train=dict(max_steps=500, save_steps=250, micro_batch_size=28, global_batch_size=28,
                           gradient_accumulation_steps=1, enable_fp32=True, enable_mixed_precision=True,
                           use_compile=True, seed=3, lr=.0001, hf_export_dtype='fp32'))
            text = json.dumps(config)
            (source / 'trainer_config.json').write_text(text)
            (source / 'datasets.txt').write_text('robotwin /datasets/robotwin\n')
            (source / 'train_ids.json').write_text('[0,1]')
            out = root / 'enhanced'
            prepare(source, out, ee_loss=True, window_sampling=True, ln_fm=True)
            self.assertEqual((source / 'trainer_config.json').read_text(), text)
            actual = json.loads((out / 'trainer_config.json').read_text())
            self.assertEqual(actual['model'], config['model'])
            for key, value in config['train'].items():
                self.assertEqual(actual['train'][key], value, key)
            for key, value in config['data'].items():
                self.assertEqual(actual['data'][key], value, key)
            self.assertTrue(actual['train']['ee_fk_loss_enabled'])
            self.assertTrue(actual['train']['low_noise_fm_enabled'])
            self.assertIsNone(actual['train']['auto_learning'])
            from lingbotvla.auto_learning.config import AutoLearningConfig
            from lingbotvla.auto_learning.sampling.target_change_window import cache_path
            window_config = json.loads((out / 'window_preparation.json').read_text())
            AutoLearningConfig(**window_config)
            methods = json.loads((out / 'methods.json').read_text())
            self.assertEqual(Path(methods['window_cache']),
                             cache_path(window_config['target_change_cache_dir'], methods['task']),
                             'methods.json must point at the file the window preparer actually writes')

    def test_all_switch_combinations(self):
        for ee, window, ln in itertools.product((False, True), repeat=3):
            with self.subTest(ee=ee, window=window, ln=ln):
                if ee and not ln:
                    with self.assertRaises(ValueError):
                        validate_flags(ee, window, ln)
                else:
                    self.assertEqual(validate_flags(ee, window, ln),
                                     dict(ee_loss=ee, window_sampling=window, ln_fm=ln))

    def test_disabled_sampler_does_not_call_factory_or_consume_rng(self):
        before = random.getstate()
        def forbidden(*args):
            self.fail('disabled sampler invoked factory')
        self.assertIsNone(make_sampler(False, forbidden, None, 3))
        self.assertEqual(before, random.getstate())

    def test_mapping_is_by_episode_and_absolute_index(self):
        cache = {'episodes': {'50': {'frame_ids': [900, 901], 'weights': [1., 2.]}}}
        self.assertEqual(aligned_weights(cache, [(50, 901), (50, 900)]), [2., 1.])
        for coords in [[(50, 900)], [(50, 900), (50, 900)], [(51, 900), (51, 901)]]:
            with self.assertRaises(ValueError):
                aligned_weights(cache, coords)

    def test_pair_requires_three_identical_scene_seeds(self):
        first = dict(episodes=3, seeds=[100001, 100002, 100005], oidn_errors=0,
                     dtype='float32', action_chunk=50, successes=1, success_rate=1/3)
        second = dict(first, successes=2, success_rate=2/3)
        self.assertEqual(validate_pair(first, second)['enhanced_successes'], 2)
        for altered in [dict(second, seeds=[100001, 100002, 100006]),
                        dict(second, episodes=10), dict(second, oidn_errors=1)]:
            with self.assertRaises(ValueError):
                validate_pair(first, altered)


if __name__ == '__main__':
    unittest.main()
