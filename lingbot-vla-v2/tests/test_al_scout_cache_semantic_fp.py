"""CPU 单测：Scout 缓存指纹的 **AST 语义 hash** + **评测 source 清单补全** + **过户工具**。

背景（2026-10-10 用户要求）：
* 改一行注释 / 无害重构就换指纹 ⇒ 50 个已扫描结果全部作废（真机实测一天两次，重扫 ~5 分钟）；
* 但**漏列**评测链上的文件更危险（数字变了指纹不变 ⇒ 拿旧结果判 PASS）。
⇒ ① source 清单补全（宁可多列）；② ``.py`` 用 AST 语义 hash；③ 存量缓存可安全过户。

本文件只测纯 CPU 路径：``ast`` / 哈希 / 文件 mtime，不需要 torch、不碰 GPU、不连网。
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from lingbotvla.auto_learning.scout_cache import (
    EVAL_SOURCES,
    NO_SEMANTIC_HASH_MARKER,
    eval_source_paths,
    normalize_dtype,
    provenance,
    semantic_sha256,
    sha256_file,
    source_manifest,
    source_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# ① AST 语义 hash
# ---------------------------------------------------------------------------
BASE_PY = '''\
"""模块 docstring（改动它不应改变指纹）。"""
import os


class Thing:
    """类 docstring。"""

    LIMIT = 0.2

    def run(self, a):
        """函数 docstring。"""
        if a < self.LIMIT:
            return 'low'
        return 'high'
'''


def _write(tmp_path: Path, text: str, name: str = 'mod.py') -> Path:
    p = tmp_path / name
    p.write_text(text, encoding='utf-8')
    return p


def test_semantic_hash_ignores_comments_blank_lines_and_indent(tmp_path):
    """注释 / 空行 / 缩进风格 / docstring ⇒ **不变**。"""
    base = _write(tmp_path, BASE_PY, 'base.py')
    h0 = semantic_sha256(base)

    cosmetic = _write(tmp_path, '''\
# 新增注释：解释为什么 LIMIT 是 0.2
import os

# 又一行注释



class Thing:
    LIMIT = 0.2

    def run(self, a):
        # 内联注释
        if a < self.LIMIT:

            return 'low'

        return 'high'
''', 'cosmetic.py')
    assert semantic_sha256(cosmetic) == h0, '注释/空行/缩进变化不得改变语义 hash'

    docstringless = _write(tmp_path, '''\
import os


class Thing:

    LIMIT = 0.2

    def run(self, a):
        if a < self.LIMIT:
            return 'low'
        return 'high'
''', 'nodoc.py')
    assert semantic_sha256(docstringless) == h0, 'docstring 增删不得改变语义 hash'


def test_semantic_hash_changes_on_constant_change(tmp_path):
    """改常数（0.2 → 0.3）⇒ **必变**。"""
    base = _write(tmp_path, BASE_PY, 'base.py')
    other = _write(tmp_path, BASE_PY.replace('0.2', '0.3'), 'other.py')
    assert semantic_sha256(other) != semantic_sha256(base)


def test_semantic_hash_changes_on_branch_add_or_remove(tmp_path):
    """增删分支 / 换运算符 ⇒ **必变**。"""
    base = _write(tmp_path, BASE_PY, 'base.py')
    h0 = semantic_sha256(base)

    extra_branch = _write(tmp_path, BASE_PY.replace(
        "        if a < self.LIMIT:\n            return 'low'\n        return 'high'",
        "        if a < self.LIMIT:\n            return 'low'\n"
        "        if a > 1.0:\n            return 'clamp'\n        return 'high'"), 'branch.py')
    assert semantic_sha256(extra_branch) != h0

    flipped = _write(tmp_path, BASE_PY.replace('a < self.LIMIT', 'a <= self.LIMIT'), 'op.py')
    assert semantic_sha256(flipped) != h0

    renamed = _write(tmp_path, BASE_PY.replace("return 'low'", "return 'LOW'"), 'lit.py')
    assert semantic_sha256(renamed) != h0


def test_semantic_hash_falls_back_to_content_hash_on_syntax_error(tmp_path, caplog):
    """``ast.parse`` 失败 ⇒ 回退内容 hash + warning（**绝不静默**）。"""
    bad = _write(tmp_path, 'def broken(:\n    pass\n', 'bad.py')
    with caplog.at_level('WARNING', logger='lingbotvla.auto_learning.scout_cache'):
        h = semantic_sha256(bad)
    assert h == sha256_file(bad), '解析失败必须回退到内容 hash'
    assert any('ast.parse' in r.message for r in caplog.records), '回退必须有 warning'

    # 修好语法（注释/空行层面的变化不再失效，但内容是新的）⇒ 又走语义 hash
    fixed = _write(tmp_path, 'def broken():\n    pass\n', 'fixed.py')
    assert semantic_sha256(fixed) != h


def test_semantic_hash_escape_hatch(tmp_path, caplog):
    """显式逃生舱：文件含 marker ⇒ 退回内容 hash（并有日志）。"""
    marked = _write(tmp_path, f'# {NO_SEMANTIC_HASH_MARKER}\nX = 1\n', 'marked.py')
    same_semantics = _write(tmp_path, f'# {NO_SEMANTIC_HASH_MARKER}\n\n\nX   =    1\n', 'marked2.py')
    assert semantic_sha256(marked) == sha256_file(marked)
    assert semantic_sha256(marked) != semantic_sha256(same_semantics)


def test_semantic_hash_docstring_rules_are_pinned(tmp_path):
    """docstring 的**精确**语义（避免"以为这样不行"的误判）：

    模块 / 类 / 函数体的**第一条** ``Expr(Constant(str))`` 一律被去掉 ⇒
    加 docstring、删 docstring、**换 docstring 文字** 都**不**改变指纹。
    （已知取舍：docstring 文字本身是 ``__doc__`` 的值 —— 但它不参与任何评测计算，
      把它算进指纹只会造成误失效；要"文字一变就失效"请用逃生舱 marker。）
    """
    no_doc = 'def f():\n    return 1\n'
    with_doc = 'def f():\n    """A."""\n    return 1\n'
    p0 = _write(tmp_path, no_doc, 'p0.py')
    p1 = _write(tmp_path, with_doc, 'p1.py')
    assert semantic_sha256(p1) == semantic_sha256(p0), '加 docstring 不得改变指纹'

    other = _write(tmp_path, 'def f():\n    """B."""\n    return 1\n', 'p2.py')
    assert semantic_sha256(other) == semantic_sha256(p1), '换 docstring 文字不得改变指纹'

    mod_a = _write(tmp_path, '"""A."""\nX = 1\n', 'm0.py')
    mod_b = _write(tmp_path, '"""B."""\nX = 1\n', 'm1.py')
    mod_none = _write(tmp_path, 'X = 1\n', 'm2.py')
    assert semantic_sha256(mod_a) == semantic_sha256(mod_b) == semantic_sha256(mod_none)

    # 但**第二条**字符串表达式是语句，不是 docstring ⇒ 必须参与哈希
    attr0 = _write(tmp_path, 'X = 1\n"""attribute doc"""\n', 'a0.py')
    attr1 = _write(tmp_path, 'X = 1\n"""attribute doc changed"""\n', 'a1.py')
    assert semantic_sha256(attr0) != semantic_sha256(attr1)


def test_source_sha256_dispatches_by_suffix(tmp_path):
    py = _write(tmp_path, 'X = 1\n# comment\n', 'a.py')
    py2 = _write(tmp_path, 'X = 1\n', 'b.py')
    js = tmp_path / 'norm.json'
    js.write_text('{"a": 1}', encoding='utf-8')
    assert source_sha256(py) == source_sha256(py2), '.py 用语义 hash'
    assert source_sha256(js) == sha256_file(js), '非 .py 用内容 hash'
    js.write_text('{"a": 2}', encoding='utf-8')
    assert source_sha256(js) == sha256_file(js)


@pytest.mark.parametrize('rel', [
    'lingbotvla/utils/open_loop_validation.py',
    'lingbotvla/data/vla_data/multi_vla_dataset.py',
    'lingbotvla/data/vla_data/base_dataset.py',
])
def test_real_eval_files_hash_semantically(tmp_path, rel):
    """真实评测链文件：加注释 ⇒ hash 不变；改逻辑 ⇒ hash 变。

    （`open_loop_validation.py` 是今天两次误失效的元凶 —— 它一天改十几版。）
    """
    src = REPO_ROOT / rel
    assert src.is_file(), f'{rel} 不存在'
    original = semantic_sha256(src)

    commented = tmp_path / Path(rel).name
    text = src.read_text(encoding='utf-8')
    # 全文最稳的「无害改动」：在最前面插注释 + 末尾加空行（语法不变、语义不变）
    commented.write_text('# 新增一行说明性注释\n# 再来一行\n' + text + '\n\n',
                         encoding='utf-8')
    assert semantic_sha256(commented) == original, f'{rel}: 注释/空行变化不得改变指纹'

    tweaked = tmp_path / ('x_' + Path(rel).name)
    tweaked.write_text(src.read_text(encoding='utf-8') + '\n\n_EXTRA_FLAG = 1\n',
                       encoding='utf-8')
    assert semantic_sha256(tweaked) != original, f'{rel}: 新增模块级逻辑必须改变指纹'


# ---------------------------------------------------------------------------
# ② provenance 端到端
# ---------------------------------------------------------------------------
def _fp(tmp_path, *, dtype='bfloat16', seed=1234, comment='# a', weight=b'WA'):
    w = tmp_path / 'w.safetensors'
    w.write_bytes(weight)
    code = tmp_path / 'code.py'
    code.write_text(f'V = 1\n{comment}\n', encoding='utf-8')
    data = tmp_path / 'norm.json'
    data.write_text('{"n": 1}', encoding='utf-8')
    return provenance(weight_files=[w], sources={'code': code, 'norm': data},
                      options={'inference_dtype': dtype, 'noise_seed': seed})


def test_provenance_is_deterministic(tmp_path):
    assert _fp(tmp_path) == _fp(tmp_path)


def test_provenance_changes_when_options_change(tmp_path):
    base = _fp(tmp_path)
    assert _fp(tmp_path, dtype='float32') != base
    assert _fp(tmp_path, seed=4321) != base


def test_provenance_changes_when_weight_content_changes(tmp_path):
    base = _fp(tmp_path)
    assert _fp(tmp_path, weight=b'WB') != base


def test_provenance_semantic_for_py_but_content_for_data(tmp_path):
    """端到端：注释变化**不**换指纹；归一化数据变化**必须**换指纹。"""
    base = _fp(tmp_path, comment='# a')
    assert _fp(tmp_path, comment='# completely different comment') == base
    w = tmp_path / 'w.safetensors'
    w.write_bytes(b'WA')
    code = tmp_path / 'code.py'
    code.write_text('V = 1\n# a\n', encoding='utf-8')
    data = tmp_path / 'norm.json'
    data.write_text('{"n": 2}', encoding='utf-8')
    changed = provenance(weight_files=[w], sources={'code': code, 'norm': data},
                         options={'inference_dtype': 'bfloat16', 'noise_seed': 1234})
    assert changed != base
    code.write_text('V = 2\n# a\n', encoding='utf-8')
    logic_changed = provenance(weight_files=[w], sources={'code': code, 'norm': data},
                               options={'inference_dtype': 'bfloat16', 'noise_seed': 1234})
    assert logic_changed != changed


def test_provenance_dtype_alias_is_normalized(tmp_path):
    """``bf16`` 与 ``bfloat16`` 是同一个精度 ⇒ 必须算同一个指纹（否则预检与运行时对不上）。"""
    assert normalize_dtype('bf16') == normalize_dtype('bfloat16') == 'bfloat16'
    assert _fp(tmp_path, dtype='bf16') == _fp(tmp_path, dtype='bfloat16')


# ---------------------------------------------------------------------------
# ③ 评测 source 清单补全
# ---------------------------------------------------------------------------
EXPECTED_NEW_KEYS = {
    'multi_vla_dataset', 'base_dataset', 'dataset_builder', 'data_utils',
    'auto_learning_config', 'checkpoint_config', 'manifest', 'norm',
    'thresholds', 'baseline', 'eval', 'model', 'transform', 'eval_precision',
    'evaluator', 'gmean',
}
# 由 `EVAL_SOURCES`（仓库内文件）提供的部分；其余来自运行时给的 manifest/norm/... 路径
EXPECTED_REPO_KEYS = {
    'multi_vla_dataset', 'base_dataset', 'dataset_builder', 'data_utils',
    'eval', 'model', 'transform', 'eval_precision', 'evaluator', 'gmean',
    'scan_accel', 'eval_batch_policy',
}


def test_eval_source_paths_cover_new_files():
    paths = eval_source_paths(REPO_ROOT)
    assert EXPECTED_REPO_KEYS.issubset(set(paths)), sorted(EXPECTED_REPO_KEYS - set(paths))
    # 用户点名的三个文件 + 数据集构造链必须真的指向存在的文件（否则"补全"是空话）
    for key in ('multi_vla_dataset', 'base_dataset', 'dataset_builder',
                'data_utils', 'transform'):
        assert paths[key].is_file(), f'{key} → {paths[key]} 不存在'
    assert paths['multi_vla_dataset'].name == 'multi_vla_dataset.py'
    assert paths['base_dataset'].name == 'base_dataset.py'
    # 清单完整性：这几项都是"改了会影响评测数字"的文件，漏一个就会拿旧结果判 PASS
    for rel in ('lingbotvla/data/vla_data/multi_vla_dataset.py',
                'lingbotvla/data/vla_data/base_dataset.py',
                'lingbotvla/data/vla_data/utils.py',
                'lingbotvla/data/dataset.py'):
        assert (REPO_ROOT / rel).is_file(), f'{rel} 不在仓库里 ⇒ 清单指向了错的路径'


def test_source_manifest_full_key_set(tmp_path):
    """数据/配置文件（manifest/norm/thresholds/baseline/AL 配置/checkpoint 配置）必须齐。"""
    cfg = tmp_path / 'al.yaml'
    cfg.write_text('enabled: true\n', encoding='utf-8')
    data = {}
    for key in ('manifest', 'norm', 'thresholds', 'baseline'):
        data[key] = tmp_path / f'{key}.json'
        data[key].write_text('{}', encoding='utf-8')
    ckpt = tmp_path / 'ckpt'
    ckpt.mkdir()
    for key, name in (('checkpoint_config', 'config.json'),
                      ('checkpoint_tokenizer', 'tokenizer.json')):
        data[key] = ckpt / name
        data[key].write_text('{}', encoding='utf-8')
    data['auto_learning_config'] = cfg
    sources, missing = source_manifest(REPO_ROOT, data)
    assert missing == []
    assert EXPECTED_NEW_KEYS.issubset(set(sources)), sorted(EXPECTED_NEW_KEYS - set(sources))


def test_source_manifest_merges_repo_and_extra(tmp_path):
    cfg = tmp_path / 'al.yaml'
    cfg.write_text('auto_learning:\n  enabled: true\n', encoding='utf-8')
    manifest = tmp_path / 'split.json'
    manifest.write_text('{}', encoding='utf-8')
    sources, missing = source_manifest(REPO_ROOT, {'manifest': manifest,
                                                   'auto_learning_config': cfg})
    assert missing == []
    for key in ('multi_vla_dataset', 'base_dataset', 'dataset_builder', 'data_utils',
                'auto_learning_config', 'manifest', 'eval', 'model', 'evaluator'):
        assert key in sources, f'{key} 应进入指纹清单'
    # 每一项都必须是**存在**的文件（否则 provenance 会 FileNotFoundError）
    assert all(p.is_file() for p in sources.values())


def test_source_manifest_degrades_gracefully_when_file_missing(tmp_path):
    """依赖文件不存在 ⇒ 不崩、不 KeyError；``strict=True`` 才报错。"""
    seen = []
    sources, missing = source_manifest(
        REPO_ROOT, {'manifest': tmp_path / 'nope.json'},
        on_missing=lambda key, reason: seen.append((key, reason)))
    assert 'manifest' not in sources
    assert [k for k, _ in missing] == ['manifest']
    assert len(seen) == 1 and seen[0][0] == 'manifest'
    with pytest.raises(FileNotFoundError, match='manifest'):
        source_manifest(REPO_ROOT, {'manifest': tmp_path / 'nope.json'}, strict=True)


def test_source_manifest_skips_missing_repo_file_without_crash(tmp_path):
    """精简检出（缺可选 .py）⇒ 跳过 + 记录，不抛异常。"""
    fake_root = tmp_path / 'fake_repo'
    for _key, rel, _req in EVAL_SOURCES:
        if 'base_dataset' in rel:
            continue
        target = fake_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('X = 1\n', encoding='utf-8')
    sources, missing = source_manifest(fake_root, {})
    assert 'base_dataset' not in sources
    assert any(k == 'base_dataset' for k, _ in missing)


def test_eval_source_keys_are_unique():
    keys = [k for k, _, _ in EVAL_SOURCES]
    assert len(keys) == len(set(keys))


def test_build_py_uses_shared_source_manifest():
    """回归：``real/build.py`` 的 scout 块必须用**同一份**清单，不得再自己手写一份。

    手写清单一旦与 ``EVAL_SOURCES`` 分叉，``tools/scan_accel_preflight.py`` 认证过的
    指纹就不再等于运行时真正用的指纹（静默重扫），而补进去的文件也可能被下一个人删掉。
    """
    src = (REPO_ROOT / 'lingbotvla/auto_learning/real/build.py').read_text(encoding='utf-8')
    i = src.index("AL_SCOUT_CACHE_MODE")
    scout_block = src[i:i + 6000]
    assert 'source_manifest(' in scout_block, 'scout 块必须复用 scout_cache.source_manifest'
    assert "on_missing=" in scout_block, '缺文件必须走 on_missing（优雅降级 + warning）'
    assert "'multi_vla_dataset'" not in scout_block, '清单不得在 build.py 里再抄一份'
    # 配置文件本身与评测数据集构造链必须在指纹输入里（可执行断言的清单侧已验证）


# ---------------------------------------------------------------------------
# ④ 迁移工具
# ---------------------------------------------------------------------------
def _load_migrate_module():
    path = REPO_ROOT / 'tools' / 'al_scout_cache_migrate.py'
    spec = importlib.util.spec_from_file_location('al_scout_cache_migrate_under_test', path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


MIG = _load_migrate_module()


def _old_fingerprint(weight_files, sources, options):
    """**复刻旧算法**（全部内容 hash + 旧清单）—— 用来造"存量缓存"目录名。"""
    manifest = {
        'schema': 1,
        'weights': [(Path(p).name, sha256_file(p)) for p in sorted(weight_files, key=str)],
        'sources': {k: sha256_file(v) for k, v in sorted(sources.items())},
        'options': dict(options),
    }
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def _fake_env(tmp_path, *, n_records=5, cache_age_s=2.0):
    """造一份「依赖文件齐全 + 旧缓存目录」的现场（全部在 tmp_path 内，不碰真机）。"""
    ckpt = tmp_path / 'ckpt'
    ckpt.mkdir()
    (ckpt / 'model-00001-of-00001.safetensors').write_bytes(b'WEIGHTS')
    (ckpt / 'config.json').write_text('{"model": "x"}', encoding='utf-8')
    (ckpt / 'tokenizer.json').write_text('{"tok": 1}', encoding='utf-8')
    norm = tmp_path / 'norm.json'
    norm.write_text('{"norm": 1}', encoding='utf-8')
    manifest = tmp_path / 'split.json'
    manifest.write_text('{"tasks": []}', encoding='utf-8')
    baseline = tmp_path / 'baseline.json'
    baseline.write_text('{"b": 1}', encoding='utf-8')
    thresholds = tmp_path / 'thresholds.json'
    thresholds.write_text('{"t": 1}', encoding='utf-8')
    cfg = tmp_path / 'al.yaml'
    cfg.write_text('auto_learning:\n  enabled: true\n'
                   '  pass_thresholds_file: null\n'
                   '  global_scout_val_trajs: 2\n', encoding='utf-8')
    cache_root = tmp_path / 'cache'
    cache_root.mkdir()
    args = dict(cache_root=str(cache_root), checkpoint=str(ckpt), manifest=str(manifest),
                norm=str(norm), baseline=str(baseline), thresholds=str(thresholds),
                config=str(cfg), dtype='bfloat16', noise_seed=1234, scout_trajs=2,
                old_fingerprint=None, min_files=1, tolerance_s=0.0,
                allow_unchanged=False, yes=True, apply=False, dry_run=True,
                json_report=False)
    old_fp = _old_fingerprint(
        sorted(ckpt.glob('*.safetensors')),
        {'eval': REPO_ROOT / 'lingbotvla/utils/open_loop_validation.py',
         'model': REPO_ROOT / 'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py',
         'transform': REPO_ROOT / 'lingbotvla/data/vla_data/transform.py',
         'eval_precision': REPO_ROOT / 'lingbotvla/utils/eval_precision.py',
         'evaluator': REPO_ROOT / 'lingbotvla/auto_learning/evaluator.py',
         'gmean': REPO_ROOT / 'lingbotvla/auto_learning/decision/gmean.py',
         'manifest': manifest, 'norm': norm, 'thresholds': thresholds, 'baseline': baseline,
         'checkpoint_config': ckpt / 'config.json',
         'checkpoint_tokenizer': ckpt / 'tokenizer.json'},
        {'inference_dtype': 'bfloat16', 'noise_seed': 1234, 'scout_trajs': 2,
         'stride': 'per_episode', 'image_augment': False})
    old_dir = cache_root / old_fp
    old_dir.mkdir()
    for i in range(n_records):
        (old_dir / f'{i:064x}.json').write_text(
            json.dumps({'version': 1, 'fingerprint': old_fp, 'task': f't{i}',
                        'episode_ids': [51, 52], 'metrics': {}}), encoding='utf-8')
    # 旧目录的 mtime 必须晚于全部依赖文件（模拟"扫描发生在文件最后一次修改之后"）
    stamp = time.time() + cache_age_s
    os.utime(old_dir, (stamp, stamp))
    return args, old_dir, cache_root


def _run(args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code, report = MIG.run(MIG.build_parser().parse_args(_argv(args)))
    return code, report, buf.getvalue()


def _argv(args):
    out = []
    for key, value in args.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                out.append(flag)
            continue
        if value is None:
            continue
        out += [flag, str(value)]
    return out


def test_migrate_dry_run_does_not_touch_filesystem(tmp_path):
    args, old_dir, cache_root = _fake_env(tmp_path)
    before = sorted(p.name for p in old_dir.iterdir())
    code, report, _text = _run(args)
    assert code == 0, report
    assert report['status'] == MIG.DRY_RUN_READY
    assert report['old_json_count'] == 5 and report['dependencies_newer'] == []
    assert not Path(report['new_dir']).exists(), 'dry-run 不得创建新目录'
    assert sorted(p.name for p in old_dir.iterdir()) == before, 'dry-run 不得改动旧目录'
    assert sorted(p.name for p in cache_root.iterdir()) == [old_dir.name]


def test_migrate_refuses_when_dependency_newer_than_cache(tmp_path):
    """构造「依赖文件比缓存新」⇒ **拒绝迁移**（安全第一）。"""
    args, old_dir, cache_root = _fake_env(tmp_path, cache_age_s=0.0)
    dep = Path(args['norm'])
    time.sleep(0.01)
    dep.write_text('{"norm": 2}', encoding='utf-8')      # 扫描之后被改过
    os.utime(dep, None)
    code, report, text = _run(args)
    assert code == 2, report
    assert report['status'] == MIG.REJECT
    assert 'norm' in report['dependencies_newer']
    assert '拒绝过户' in report['reason']
    assert not Path(report['new_dir']).exists()
    assert sorted(p.name for p in cache_root.iterdir()) == [old_dir.name]


def test_migrate_refuses_when_dependency_missing(tmp_path):
    """依赖文件被删 ⇒ 拒绝（删掉的文件同样"变了"，而且算出的新指纹也不是运行时那个）。"""
    args, _old, cache_root = _fake_env(tmp_path)
    Path(args['baseline']).unlink()
    code, report, _text = _run(args)
    assert code == 2 and report['status'] == MIG.REJECT
    assert 'baseline' in report['missing_required']
    assert len(list(cache_root.iterdir())) == 1, '拒绝后缓存根目录不得多出任何东西'


def test_migrate_refuses_when_multiple_candidates(tmp_path):
    args, old_dir, cache_root = _fake_env(tmp_path)
    other = cache_root / ('b' * 64)
    other.mkdir()
    (other / 'x.json').write_text('{}', encoding='utf-8')
    code, report, _text = _run(args)
    assert code == 2 and report['status'] == MIG.REJECT
    assert '候选' in report['reason']


def test_migrate_refuses_when_new_dir_nonempty(tmp_path):
    args, old_dir, cache_root = _fake_env(tmp_path)
    code, report, _text = _run(args)
    assert code == 0
    new_dir = Path(report['new_dir'])
    new_dir.mkdir(parents=True, exist_ok=True)
    (new_dir / 'stale.json').write_text('{}', encoding='utf-8')
    code, report, _text = _run(args)
    assert code == 2 and '非空' in report['reason']


def test_migrate_apply_hardlinks_and_keeps_old_dir(tmp_path):
    """``--apply``：硬链接过户、旧目录保留、两侧文件数一致。"""
    args, old_dir, _root = _fake_env(tmp_path, n_records=5)
    args = dict(args, apply=True, dry_run=False)
    code, report, text = _run(args)
    assert code == 0 and report['status'] == MIG.APPLY_READY, text
    new_dir = Path(report['new_dir'])
    old_files = sorted(p.name for p in old_dir.iterdir())
    new_files = sorted(p.name for p in new_dir.iterdir())
    assert old_files == new_files and len(new_files) == 5
    assert report['result']['old_dir_kept'] is True
    assert report['result']['old_json'] == report['result']['new_json'] == 5
    assert report['result']['hardlinked'] == 5, '同设备必须走 os.link'
    for name in old_files:
        assert os.stat(old_dir / name).st_ino == os.stat(new_dir / name).st_ino, '必须是硬链接'
    # 二次运行：新目录已非空 ⇒ 拒绝（幂等、不重复搬）
    code2, report2, _t = _run(args)
    assert code2 == 2 and '非空' in report2['reason']


def test_migrate_rejects_without_new_fingerprint_inputs(tmp_path):
    args, _old, _root = _fake_env(tmp_path)
    args = dict(args, checkpoint=None)
    code, report, _text = _run(args)
    assert code == 2 and report['status'] == MIG.REJECT and report['new_fingerprint'] is None


def test_migrate_allow_unchanged_is_explicit(tmp_path):
    """mtime 判定被人工推翻时必须走 ``--allow-unchanged``（并在 warnings 里留痕）。"""
    args, old_dir, cache_root = _fake_env(tmp_path, cache_age_s=0.0)
    dep = Path(args['manifest'])
    time.sleep(0.01)
    dep.write_text('{"tasks": ["x"]}', encoding='utf-8')
    os.utime(dep, None)
    code, report, _text = _run(dict(args, apply=True, dry_run=False, allow_unchanged=True))
    assert code == 0 and report['status'] == MIG.APPLY_READY
    assert any('allow-unchanged' in w for w in report['warnings'])
    assert (Path(report['new_dir'])).is_dir()
    assert old_dir.is_dir()


def test_migrate_module_help_runs():
    """``--help`` 不应触碰任何环境（真机跑之前先能看用法）。"""
    with pytest.raises(SystemExit) as exc:
        MIG.build_parser().parse_args(['--help'])
    assert exc.value.code == 0
