"""`al_launch.py --env` 与 `AL_*` 透传回归（CPU；2026-10-10 加）。

背景（真机踩坑）
----------------
启动器给扫描 worker 与训练**白名单构造**环境变量（有意为之：可审计、不受调用者 shell 影响）。
但因此父进程 `export AL_HARDNESS_SHARD=1` **到不了训练进程** —— 实测 rank 的
`/proc/<pid>/environ` 里根本没有该变量，于是"开了分片"其实没生效，白跑一轮。

本用例直接对 `--dry-run` 的环境变量段断言（走真实 `base_env()`），不启动任何进程。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'experiment' / 'robotwin' / 'al_launch.py'


def _dry_run(extra_args):
    env = dict(os.environ)
    env.setdefault('PYTHONPATH', str(ROOT))
    proc = subprocess.run([sys.executable, str(LAUNCHER), '--dry-run', *extra_args],
                          capture_output=True, text=True, errors='replace',
                          timeout=300, cwd=str(ROOT), env=env)
    return proc


def _env_values(text: str) -> dict:
    """从 dry-run 输出的 `KEY=VALUE` 行里抓环境变量。"""
    out = {}
    for line in text.splitlines():
        m = re.match(r'^\s{2,}([A-Z][A-Z0-9_]*)=(.*)$', line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_env_flag_reaches_worker_and_train_plans():
    proc = _dry_run(['--env', 'AL_HARDNESS_SHARD=1', '--env', 'AL_HARDNESS_LOG_SEC=7'])
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    blob = proc.stdout + proc.stderr
    vals = _env_values(blob)
    assert vals.get('AL_HARDNESS_SHARD') == '1', f'--env 未进入环境变量段:\n{blob[-1500:]}'
    assert vals.get('AL_HARDNESS_LOG_SEC') == '7', f'--env 未进入环境变量段:\n{blob[-1500:]}'


@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_parent_al_env_is_passed_through():
    """父进程里已存在的 AL_* 必须自动透传（不被白名单吞掉）。"""
    env = dict(os.environ, AL_HARDNESS_SHARD='1', PYTHONPATH=str(ROOT))
    proc = subprocess.run([sys.executable, str(LAUNCHER), '--dry-run'],
                          capture_output=True, text=True, errors='replace',
                          timeout=300, cwd=str(ROOT), env=env)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    vals = _env_values(proc.stdout + proc.stderr)
    assert vals.get('AL_HARDNESS_SHARD') == '1', '父进程的 AL_* 没有透传到子进程环境'


@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_bad_env_flag_is_rejected():
    proc = _dry_run(['--env', '=novalue'])
    blob = proc.stdout + proc.stderr
    assert proc.returncode != 0, f'非法 --env 应被拒绝:\n{blob[-800:]}'
    assert 'env' in blob.lower()


# --------------------------------------------------------------------------- #
# 启动器空间检查：hardness-cache **不得**被 20G 下限拦死（2026-10-10 自锁事故）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_hardness_cache_is_not_blocked_by_min_free_gb(tmp_path):
    """真机教训：把体积只有几十 KB 的 hardness 缓存塞进「20G 下限」检查后，
    `/workspace` 只剩 19.7G ⇒ **任何启用缓存的启动都被自己拦死**（dry-run 退出码 2）。

    这里用 dry-run 断言：即使把下限抬到远高于该目录可用空间，hardness-cache 也不报错。
    """
    hc = tmp_path / 'hardness_cache'
    hc.mkdir()
    proc = _dry_run(['--hardness-cache', str(hc), '--min-free-gb', '100000'])
    blob = proc.stdout + proc.stderr
    # 其它三个目录会因 100000G 下限而报错 ⇒ 只看"有没有点名 hardness-cache"
    assert 'hardness-cache' not in blob, (
        f'hardness-cache 仍被空间检查拦下（不应参与 20G 下限）：\n{blob[-1200:]}')


@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_hardness_cache_missing_dir_is_allowed(tmp_path):
    """缓存目录不存在**允许**（`store()` 会 `mkdir(parents=True)` 自建）——不应因此拦启动。"""
    proc = _dry_run(['--hardness-cache', str(tmp_path / 'nope' / 'hc'), '--min-free-gb', '0'])
    blob = proc.stdout + proc.stderr
    assert 'hardness-cache 不可写' not in blob and '上层路径不是目录' not in blob, blob[-1200:]


@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_hardness_cache_default_is_persistent_volume():
    """默认必须是持久卷路径（用户 2026-10-10 定：持久化）。"""
    proc = _dry_run(['--min-free-gb', '0'])
    blob = proc.stdout + proc.stderr
    assert '/workspace/al/hardness_cache' in blob, blob[-1200:]


# --------------------------------------------------------------------------- #
# 指纹计算默认必须**关闭**（2026-10-10：缓存改为显式文件 + 模型名后，指纹不再参与寻址）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_fingerprint_computation_is_skipped_by_default():
    """默认不得去哈希 11.9 GiB 权重分片：dry-run 应打印「跳过计算」且不出现「正在哈希」。"""
    proc = _dry_run(['--min-free-gb', '0'])
    blob = proc.stdout + proc.stderr
    assert '正在哈希' not in blob, (
        '默认仍在哈希权重分片（11.9 GiB，每轮多花 1–3 分钟 I/O）——'
        '指纹已不参与缓存寻址，应改为 --compute-fingerprint 显式开启')
    assert 'fingerprint] 跳过计算' in blob or '跳过计算' in blob, blob[-1200:]


@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_explicit_compute_fingerprint_flag_is_accepted():
    """显式开关必须被接受（不报 argparse 错）——dry-run 下因路径不齐会跳过，但不得崩。"""
    proc = _dry_run(['--compute-fingerprint', '--min-free-gb', '0'])
    blob = proc.stdout + proc.stderr
    assert 'unrecognized arguments' not in blob, blob[-800:]
    assert '--compute-fingerprint' not in blob or 'error' not in blob.lower()


@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_deprecated_fingerprint_env_is_not_exported():
    """已废弃的 AL_SCOUT_CACHE_FINGERPRINT 不得再出现在子进程 env 里。"""
    proc = _dry_run(['--min-free-gb', '0'])
    blob = proc.stdout + proc.stderr
    assert 'AL_SCOUT_CACHE_FINGERPRINT=' not in blob, '仍在透传已废弃的指纹 env'
    # 新 env 必须都在
    for key in ('AL_SCOUT_CACHE_FILE=', 'AL_HARDNESS_CACHE_FILE=', 'AL_MODEL_NAME='):
        assert key in blob, f'{key} 未透传'


@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_scout_and_hardness_cache_files_are_distinct_paths():
    """两个缓存文件必须分属不同目录（避免互相覆盖）。"""
    proc = _dry_run(['--min-free-gb', '0'])
    blob = proc.stdout + proc.stderr
    assert '/workspace/al/scout_cache/scout.json' in blob
    assert '/workspace/al/hardness_cache/hardness.json' in blob


# --------------------------------------------------------------------------- #
# 残留标识符防回归（2026-10-10：cache_dir→cache_file 改名后漏改 14 处，真机才炸出 NameError）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_no_stale_cache_dir_identifier():
    """`cache_dir` 已被 `cache_file` 取代（缓存从目录改成单文件）。

    真机教训：改名时只改了部分引用，剩下的在 `--no-cache` 等**本地 dry-run 走不到**的分支里，
    直到真机启动才抛 `NameError: name 'cache_dir' is not defined`（退出码 6）。这条用源码级
    断言把所有残留一次挡住（`al_cfg: Path, cache_dir` 这类子串假阳性由 `cache_dir` 单词边界过滤）。
    """
    import re
    src = LAUNCHER.read_text(encoding='utf-8')
    hits = [m.start() for m in re.finditer(r'\bcache_dir\b', src)]
    assert not hits, (
        f'al_launch.py 仍残留 {len(hits)} 处 `cache_dir` 标识符（应全部为 cache_file）：'
        + str([src[max(0, h - 40):h + 20].replace("\n", "\\n") for h in hits[:3]]))


@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_scan_round_signature_uses_cache_file():
    """`_scan_round` 的形参必须叫 `cache_file`（与调用点一致）。"""
    src = LAUNCHER.read_text(encoding='utf-8')
    i = src.index('def _scan_round(')
    head = src[i:i + 400]
    assert 'cache_file: Path' in head, head[:300]


# --------------------------------------------------------------------------- #
# 结构性静态检查：调用不存在的方法 / 变量（2026-10-10 连续两次真机 NameError/AttributeError）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_coverage_method_calls_exist():
    """`Coverage` 上被调用的方法必须真实存在。

    真机教训（同类第二次）：`json_files()` 改名为 `record_count()` 后漏改调用点 ⇒
    真机 `AttributeError: 'Coverage' object has no attribute 'json_files'`（退出码 6），
    而这些分支本地 dry-run 走不到。这里用 AST 收集 `Coverage` 的成员名与源码里
    `cov*.method()` 的调用，逐一核对。
    """
    import ast
    import re
    src = LAUNCHER.read_text(encoding='utf-8')
    tree = ast.parse(src)
    members = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'Coverage':
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(item.name)
                elif isinstance(item, ast.Assign):        # 类级属性
                    for t in item.targets:
                        if isinstance(t, ast.Name):
                            members.add(t.id)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    members.add(item.target.id)
    assert members, '没解析到 Coverage 成员'

    called = set(re.findall(r'\b(?:cov_now|cov_before|final_cov|target_cov)\.([A-Za-z_]\w*)\(', src))
    known_family = {'summary', 'as_dict', 'record_count', 'json_files', 'add',
                    'covered', 'missing', 'reasons'}
    suspicious = {c for c in called if c in members or c in known_family}
    unknown = sorted(c for c in suspicious if c not in members)
    assert not unknown, f'Coverage 上被调用但不存在的方法：{unknown}（现有：{sorted(members)}）'




# --------------------------------------------------------------------------- #
# 重复常量同步校验（2026-10-10 真机事故：VERSION 漂移 ⇒ 真命中被判未命中、拒启动）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_cache_constants_match_scout_cache():
    """launcher 里复制的 scout 缓存常量必须与 `scout_cache.py` **逐字一致**。

    真机现象：`scout_cache.VERSION` 升到 2 后 launcher 仍是 1 ⇒ `entry_is_hit` 报
    `version=2 != 1` ⇒ 覆盖被判成 0/50 ⇒ launcher 以退出码 3 拒启动（"覆盖不完整"），
    而缓存文件其实完全有效。注释写了"必须一致"但**没人校验**，于是漂移了。
    """
    import importlib.util
    import re
    src = LAUNCHER.read_text(encoding='utf-8')
    sys.path.insert(0, str(LAUNCHER.parents[2]))
    for mod_name in ('scout_cache',):
        pass
    spec = importlib.util.spec_from_file_location(
        'scout_cache_probe', LAUNCHER.parents[2] / 'lingbotvla/auto_learning/scout_cache.py')
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)

    def const(name: str) -> str:
        m = re.search(rf'^{name}\s*=\s*([^\n#]+)', src, re.M)
        assert m, f'launcher 里找不到常量 {name}'
        return m.group(1).strip()

    assert const('VERSION') == str(sc.VERSION), (
        f'launcher VERSION={const("VERSION")} 与 scout_cache.VERSION={sc.VERSION} 不一致 ⇒ '
        '会把真命中判成未命中、拒绝启动')
    assert eval(const('MAX_JSON_BYTES')) == sc.MAX_JSON_BYTES, (   # noqa: S307 —— 源码常量，安全
        f'launcher MAX_JSON_BYTES={const("MAX_JSON_BYTES")} 与 scout_cache 的 {sc.MAX_JSON_BYTES} 不一致')


# --------------------------------------------------------------------------- #
# 按 rank 隔离编译缓存（2026-10-10 真机冻结事故的修法）
# --------------------------------------------------------------------------- #
def test_per_rank_compile_cache_isolation(tmp_path):
    """`_al_per_rank_compile_cache()` 必须按 LOCAL_RANK 加子目录、幂等、且有逃生开关。

    真机现象：7 个 rank 共享同一 `TORCHINDUCTOR_CACHE_DIR` 时，231 个编译 worker 烧 ~8 核
    却零产物，首步编译永久冻结（v28 冻在 4620 kernel、v29 冻在 5846 kernel，GPU 0%）。
    """
    import ast
    import subprocess
    import sys
    script = ROOT / 'tasks/vla/train_lingbotvla.py'
    assert script.is_file(), f'找不到 {script}'
    code = (
        "import ast, os\n"
        f"src = open({str(script)!r}).read()\n"
        "fn = next(n for n in ast.parse(src).body"
        " if isinstance(n, ast.FunctionDef) and n.name == '_al_per_rank_compile_cache')\n"
        "ns = {'os': os}\n"
        "exec(compile(ast.Module(body=[fn], type_ignores=[]), '<f>', 'exec'), ns)\n"
        "ns['_al_per_rank_compile_cache']()\n"
        "ns['_al_per_rank_compile_cache']()\n"        # 幂等
        "print(os.environ['TORCHINDUCTOR_CACHE_DIR'])\n"
    )
    env = {'PATH': '/usr/bin:/bin', 'LOCAL_RANK': '5',
           'TORCHINDUCTOR_CACHE_DIR': '/base/ti', 'TRITON_CACHE_DIR': '/base/tr'}
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         env=env, timeout=60)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip() == '/base/ti/rank5', out.stdout
    # 幂等：调用两次仍只有一个 rank5
    assert out.stdout.count('rank5') == 1

    env_shared = dict(env, AL_SHARED_COMPILE_CACHE='1')
    out2 = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                          env=env_shared, timeout=60)
    assert out2.stdout.strip() == '/base/ti', f'逃生开关失效：{out2.stdout}'

    env_norank = {k: v for k, v in env.items() if k != 'LOCAL_RANK'}
    out3 = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                          env=env_norank, timeout=60)
    assert out3.stdout.strip() == '/base/ti', f'无 LOCAL_RANK 时不应改动：{out3.stdout}'
