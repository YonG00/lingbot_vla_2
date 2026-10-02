#!/usr/bin/env python
"""多 checkpoint 评测调度器 + sentinel 清单的单元测试。

无需 GPU, 无需 pytest:
    python tests/test_multi_ckpt_eval.py
    python tests/test_multi_ckpt_eval.py -v      # 显示更多细节

覆盖:
  T1  make_batches: 1~5 个 checkpoint -> [1] [2] [2,1] [2,2] [2,2,1]
  T2  hf_ckpt 完整性判据的 8 种情形
  T3  discover_checkpoints 发现与排序
  T4  classify: 新增 / 更新 / 历史不变 / 不完整
  T5  plan_schedule + validate_plan: 端口段不重叠 / 输出不撞 / model_path 不串
  T5c video 开关双向显式传递 (--enable_video / --no_video)
  T6  dry-run: 不调用 subprocess, 不创建作业目录
  T7  preflight: 幂等同步共享文件 (原子替换)
  T8  stats.txt 解析
  T9  sentinel 清单: 行数 4/8/12/16 + 前缀累积 + 越级拦截
  T10 episodes (test_num) 透传 + eval client 不再硬编码
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiment" / "robotwin"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import robotwin_multi_ckpt_eval as mce                       # noqa: E402


# ---------------------------------------------------------------------------
# 测试替身: 造一棵假的 checkpoint 目录树
# ---------------------------------------------------------------------------

def make_ckpt(root: Path, exp: str, step: int, *, n_shards: int = 2,
              residue: bool = False, truncate: bool = False,
              drop_shard: bool = False, empty_shard: bool = False,
              drop_index: bool = False, drop_config: bool = False,
              shard_bytes: int = 1000) -> Path:
    """在 root/<exp>/checkpoints/global_step_<step>/hf_ckpt 下造一个假 checkpoint。"""
    d = root / exp / "checkpoints" / f"global_step_{step}" / "hf_ckpt"
    d.mkdir(parents=True, exist_ok=True)

    if not drop_config:
        (d / "config.json").write_text("{}", encoding="utf-8")

    names = [f"model-{i:05d}-of-{n_shards:05d}.safetensors" for i in range(1, n_shards + 1)]
    weight_map = {}
    total = 0
    for i, name in enumerate(names):
        # 索引**始终**声明全部分片 —— 这样 drop_shard 才能模拟
        # 「index 里写了, 但文件不存在」这种真实的中断场景。
        weight_map[f"weight.{i}"] = name
        total += shard_bytes
        if drop_shard and i == len(names) - 1:
            continue
        size = 0 if (empty_shard and i == 0) else shard_bytes
        (d / name).write_bytes(b"x" * size)

    if truncate:
        total += shard_bytes          # 声明比实际大 -> 疑似截断

    if not drop_index:
        (d / mce.INDEX_FILENAME).write_text(
            json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}),
            encoding="utf-8",
        )

    if residue:
        (d / ".model-00001-of-00002.safetensors.ABC123").write_bytes(b"y" * 16)

    return d


class Sandbox:
    """临时目录, 退出时清理。"""

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix="mce_test_"))
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


# ---------------------------------------------------------------------------
# T1 批次形状
# ---------------------------------------------------------------------------

def test_t1_make_batches():
    assert mce.make_batches(0, 2) == []
    assert mce.make_batches(1, 2) == [1]
    assert mce.make_batches(2, 2) == [2]
    assert mce.make_batches(3, 2) == [2, 1]
    assert mce.make_batches(4, 2) == [2, 2]
    assert mce.make_batches(5, 2) == [2, 2, 1]
    # 并发上限 >= 任务数时只有一批
    assert mce.make_batches(3, 4) == [3]
    # 串行 (max_parallel=1) 时逐个跑
    assert mce.make_batches(3, 1) == [1, 1, 1]
    # 非法参数
    for bad in (0, -1):
        try:
            mce.make_batches(3, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"max_parallel={bad} 应当报错")


# ---------------------------------------------------------------------------
# T2 完整性判据
# ---------------------------------------------------------------------------

def test_t2_completeness_cases():
    with Sandbox() as tmp:
        # --- 完整 ---
        d = make_ckpt(tmp, "expA", 1000)
        ok, reasons, warnings, total, _ = mce.check_hf_ckpt(d)
        assert ok and not reasons, f"完整 checkpoint 被判为不完整: {reasons}"
        assert total == 2000 and not warnings

        # --- 有残留临时文件 -> 仍然完整, 但告警 ---
        d = make_ckpt(tmp, "expA", 1100, residue=True)
        ok, reasons, warnings, _, _ = mce.check_hf_ckpt(d)
        assert ok and not reasons, f"有残留不该判为不完整: {reasons}"
        assert any("残留" in w for w in warnings), f"应给出残留告警: {warnings}"

        # --- 缺分片 ---
        d = make_ckpt(tmp, "expA", 1200, drop_shard=True)
        ok, reasons, _, _, _ = mce.check_hf_ckpt(d)
        assert not ok and any("缺少分片" in r for r in reasons), reasons

        # --- 空分片 ---
        d = make_ckpt(tmp, "expA", 1300, empty_shard=True)
        ok, reasons, _, _, _ = mce.check_hf_ckpt(d)
        assert not ok and any("分片为空" in r for r in reasons), reasons

        # --- 截断 (声明 > 实际) ---
        d = make_ckpt(tmp, "expA", 1400, truncate=True)
        ok, reasons, _, _, _ = mce.check_hf_ckpt(d)
        assert not ok and any("截断" in r for r in reasons), reasons

        # --- 缺 index ---
        d = make_ckpt(tmp, "expA", 1500, drop_index=True)
        ok, reasons, _, _, _ = mce.check_hf_ckpt(d)
        assert not ok and any("index.json" in r for r in reasons), reasons

        # --- 静默期: 刚写完, 视为仍在写入 ---
        d = make_ckpt(tmp, "expA", 1600)
        ok, reasons, _, _, _ = mce.check_hf_ckpt(d, min_age_seconds=3600)
        assert not ok and any("仍在写入" in r for r in reasons), reasons
        # 同一目录, 静默期为 0 -> 通过
        ok, reasons, _, _, _ = mce.check_hf_ckpt(d, min_age_seconds=0)
        assert ok and not reasons, reasons

        # --- _READY 标记可以绕过静默期 ---
        d = make_ckpt(tmp, "expA", 1700)
        (d / mce.READY_MARKER).write_text("")
        ok, reasons, _, _, _ = mce.check_hf_ckpt(d, min_age_seconds=3600)
        assert ok and not reasons, f"_READY 应绕过静默期: {reasons}"

        # --- 缺 config.json 只告警 ---
        d = make_ckpt(tmp, "expA", 1800, drop_config=True)
        ok, _, warnings, _, _ = mce.check_hf_ckpt(d)
        assert ok and any("config.json" in w for w in warnings), warnings

        # --- 不存在 ---
        ok, reasons, _, _, _ = mce.check_hf_ckpt(tmp / "nope")
        assert not ok and reasons


# ---------------------------------------------------------------------------
# T3 发现
# ---------------------------------------------------------------------------

def test_t3_discover():
    with Sandbox() as tmp:
        make_ckpt(tmp, "expA", 2000, n_shards=1)
        make_ckpt(tmp, "expA", 1000, n_shards=1)
        make_ckpt(tmp, "expB", 500, n_shards=1)
        make_ckpt(tmp, "expA", 3000, n_shards=1, drop_shard=True)

        found = mce.discover_checkpoints(tmp)
        tags = [c.tag for c in found]
        assert tags == ["expA_step1000", "expA_step2000", "expA_step3000", "expB_step500"], tags
        assert [c.complete for c in found] == [True, True, False, True]
        # 不完整的要带上原因
        bad = [c for c in found if not c.complete][0]
        assert any("缺少分片" in r for r in bad.reasons), bad.reasons

        # --ckpt-root 直接指到 output_dir 也要能找到
        found2 = mce.discover_checkpoints(tmp / "expB")
        assert [c.tag for c in found2] == ["expB_step500"], [c.tag for c in found2]

        # 空目录
        assert mce.discover_checkpoints(tmp / "expB" / "checkpoints") == []


# ---------------------------------------------------------------------------
# T4 增量分类
# ---------------------------------------------------------------------------

def test_t4_classify():
    with Sandbox() as tmp:
        make_ckpt(tmp, "expA", 1000, n_shards=1)
        make_ckpt(tmp, "expA", 2000, n_shards=1)
        make_ckpt(tmp, "expA", 3000, n_shards=1, drop_shard=True)
        ckpts = mce.discover_checkpoints(tmp)

        # 全新: 都是 new
        b = mce.classify(ckpts, {})
        assert [c.tag for c in b["new"]] == ["expA_step1000", "expA_step2000"]
        assert [c.tag for c in b["incomplete"]] == ["expA_step3000"]
        assert b["updated"] == [] and b["unchanged"] == []

        # 存盘后: 都变 unchanged
        state_file = tmp / "eval_state.json"
        mce.save_state(state_file, ckpts)
        b = mce.classify(ckpts, mce.load_state(state_file))
        assert b["new"] == [] and b["updated"] == []
        assert [c.tag for c in b["unchanged"]] == ["expA_step1000", "expA_step2000"]

        # 改动 step1000 -> updated
        d = tmp / "expA" / "checkpoints" / "global_step_1000" / "hf_ckpt"
        (d / "model-00001-of-00001.safetensors").write_bytes(b"z" * 5000)
        os.utime(d / "model-00001-of-00001.safetensors", (time.time(), time.time()))
        ckpts2 = mce.discover_checkpoints(tmp)
        b = mce.classify(ckpts2, mce.load_state(state_file))
        assert [c.tag for c in b["updated"]] == ["expA_step1000"], [c.tag for c in b["updated"]]

        # 新增 step4000 -> new
        make_ckpt(tmp, "expA", 4000, n_shards=1)
        ckpts3 = mce.discover_checkpoints(tmp)
        b = mce.classify(ckpts3, mce.load_state(state_file))
        assert [c.tag for c in b["new"]] == ["expA_step4000"], [c.tag for c in b["new"]]

        # 状态文件损坏 -> 当作空表, 不炸
        state_file.write_text("{ not json", encoding="utf-8")
        assert mce.load_state(state_file) == {}


# ---------------------------------------------------------------------------
# T5 调度计划
# ---------------------------------------------------------------------------

def _plans(tmp: Path, n_ckpt: int, max_parallel: int, **kw):
    for i in range(n_ckpt):
        make_ckpt(tmp, "expA", 1000 * (i + 1), n_shards=1)
    ckpts = [c for c in mce.discover_checkpoints(tmp) if c.complete]
    defaults = dict(
        max_parallel=max_parallel,
        conditions=["clean", "randomized"],
        num_gpus=4,
        num_per_gpu=1,
        start_port_base=9330,
        output_base=tmp / "out",
        task_list_file=tmp / "phase1_eval.txt",
    )
    defaults.update(kw)
    return mce.plan_schedule(ckpts, **defaults)


def test_t5_plan_and_validate():
    with Sandbox() as tmp:
        plans = _plans(tmp, 5, 2)
        assert len(plans) == 5
        assert [len(p.jobs) for p in plans] == [2] * 5        # clean + randomized
        assert [p.slot_index for p in plans] == [0, 1, 0, 1, 0]

        # 自检必须通过
        problems = mce.validate_plan(plans, max_parallel=2)
        assert not problems, problems

        # 端口段: slot0 = 9330~9333, slot1 = 9334~9337
        for plan in plans:
            want = (9330, 9333) if plan.slot_index == 0 else (9334, 9337)
            for job in plan.jobs:
                assert (job.start_port, job.end_port) == want, (job.start_port, job.end_port)
                assert job.gpus == [0, 1, 2, 3]

        # 输出 / 日志互不重叠
        outs = [j.output_base for p in plans for j in p.jobs]
        logs = [j.log_file for p in plans for j in p.jobs]
        assert len(outs) == len(set(outs)) and len(logs) == len(set(logs))

        # 同一个 ckpt 的 model_path 一致, 不同 ckpt 不串
        for plan in plans:
            assert len({j.hf_ckpt for j in plan.jobs}) == 1
        assert len({p.hf_ckpt for p in plans}) == 5

        # condition 内部串行: 顺序就是 clean -> randomized
        assert [j.condition for j in plans[0].jobs] == ["clean", "randomized"]
        assert plans[0].jobs[0].task_config == "demo_clean"
        assert plans[0].jobs[1].task_config == "demo_randomized"

        # 命令里带上了 --task_list_file, 且 video 开关显式关闭
        cmd = plans[0].jobs[0].cmd
        assert "--task_list_file" in cmd
        assert "--no_video" in cmd and "--enable_video" not in cmd
        assert cmd[cmd.index("--use_fp32") + 1] == "true"
        assert cmd[cmd.index("--use_bf16") + 1] == "false"

        # 单并发: 全部用 slot 0 的端口段
        plans1 = _plans(tmp / "single", 3, 1)
        assert [p.slot_index for p in plans1] == [0, 0, 0]
        assert {j.start_port for p in plans1 for j in p.jobs} == {9330}
        assert not mce.validate_plan(plans1, max_parallel=1)

        # num_per_gpu=2 -> 端口段宽度 8
        plans2 = _plans(tmp / "pergpu", 2, 2, num_per_gpu=2)
        bands = sorted({(j.start_port, j.end_port) for p in plans2 for j in p.jobs})
        assert bands == [(9330, 9337), (9338, 9345)], bands
        assert not mce.validate_plan(plans2, max_parallel=2)

        # 非法 condition 要报错
        try:
            _plans(tmp / "bad", 1, 1, conditions=["nope"])
        except ValueError:
            pass
        else:
            raise AssertionError("非法 condition 应当报错")


def test_t5c_video_flag_both_ways():
    """video 开关必须双向显式传递。

    背景: launcher 的 enable_video 默认 False, 且上游**没有**能把它打开的 CLI
    开关 (只有 --no_video) —— 如果调度器只在关闭时传 --no_video, 那 --video
    就是个无声的摆设。这个测试锁住"两个方向都显式传"。
    """
    with Sandbox() as tmp:
        off = _plans(tmp / "off", 1, 1, enable_video=False)[0].jobs[0].cmd
        assert "--no_video" in off
        assert "--enable_video" not in off

        on = _plans(tmp / "on", 1, 1, enable_video=True)[0].jobs[0].cmd
        assert "--enable_video" in on
        assert "--no_video" not in on

        # launcher 必须真的认识 --enable_video (否则 argparse 直接报 Unknown argument)
        launcher = mce.DEFAULT_LAUNCHER
        text = Path(launcher).read_text(encoding="utf-8")
        assert "--enable_video)" in text, "launcher 缺少 --enable_video 分支"
        assert "--enable_video" in text.split("-h|--help")[1], "launcher help 未列出 --enable_video"


def test_t5b_validate_catches_overlap():
    """人为造一个端口重叠的计划, validate_plan 必须抓到。"""
    with Sandbox() as tmp:
        plans = _plans(tmp, 2, 2)
        plans[1].jobs[0].start_port = plans[0].jobs[0].start_port
        plans[1].jobs[0].end_port = plans[0].jobs[0].end_port
        plans[1].jobs[1].start_port = plans[0].jobs[1].start_port
        plans[1].jobs[1].end_port = plans[0].jobs[1].end_port
        problems = mce.validate_plan(plans, max_parallel=2)
        assert any("重叠" in p for p in problems), problems


# ---------------------------------------------------------------------------
# T6 dry-run
# ---------------------------------------------------------------------------

def test_t6_dry_run_no_subprocess():
    with Sandbox() as tmp:
        make_ckpt(tmp, "expA", 1000, n_shards=1)
        make_ckpt(tmp, "expA", 2000, n_shards=1)
        tasks = tmp / "tasks.txt"
        tasks.write_text("# 注释\nlift_pot\nclick_bell\n", encoding="utf-8")
        out = tmp / "out"

        calls: list = []
        orig = mce.subprocess.run

        def _boom(*a, **k):
            calls.append(a)
            raise AssertionError("dry-run 不应调用 subprocess")

        mce.subprocess.run = _boom
        try:
            rc = mce.main([
                "--ckpt-root", str(tmp),
                "--task-list-file", str(tasks),
                "--output-base", str(out),
                "--dry-run", "--all",
            ])
        finally:
            mce.subprocess.run = orig

        assert rc == 0, rc
        assert not calls, "dry-run 期间不应有子进程"
        # 不应创建任何作业输出目录
        assert not (out / "expA_step1000").exists()
        assert not (out / "expA_step2000").exists()
        # 但计划产物应当落盘
        assert (out / "summary.txt").is_file()
        payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert payload["dry_run"] is True
        assert payload["batches"] == [2]
        assert payload["tasks"] == ["lift_pot", "click_bell"]
        assert len(payload["plans"]) == 2


# ---------------------------------------------------------------------------
# T7 preflight
# ---------------------------------------------------------------------------

def test_t7_preflight():
    with Sandbox() as tmp:
        inf = tmp / "inference"
        ev = tmp / "robotwin"
        (inf / "experiment" / "robotwin").mkdir(parents=True)
        (inf / "deploy").mkdir(parents=True)
        src_client = inf / "experiment" / "robotwin" / "eval_policy_client_lingbotvla.py"
        src_client.write_text("V1", encoding="utf-8")
        for name in ("__init__.py", "websocket_client_policy.py", "msgpack_numpy.py"):
            (inf / "deploy" / name).write_text(f"# {name}", encoding="utf-8")

        # 首次: 全部同步
        actions = mce.preflight(ev, inf)
        assert len(actions) == 4, actions
        assert (ev / "script" / "eval_policy_client_lingbotvla.py").read_text() == "V1"

        # 再次: 幂等, 无动作
        assert mce.preflight(ev, inf) == []

        # 源变了 -> 重新同步 (原子替换, 不留临时文件)
        src_client.write_text("V2", encoding="utf-8")
        actions = mce.preflight(ev, inf)
        assert len(actions) == 1, actions
        assert (ev / "script" / "eval_policy_client_lingbotvla.py").read_text() == "V2"
        leftovers = [p.name for p in (ev / "script").iterdir() if ".preflight." in p.name]
        assert not leftovers, leftovers


# ---------------------------------------------------------------------------
# T8 stats.txt 解析
# ---------------------------------------------------------------------------

def test_t8_stats_parsing():
    with Sandbox() as tmp:
        run = tmp / "expA_1k_demo_clean_20261002_120000"
        run.mkdir(parents=True)
        (run / "stats.txt").write_text(
            "============================================\n"
            "  Eval Result Stats\n"
            "  Tasks: 4\n"
            "============================================\n"
            "Summary: total 261s, success 3/6, overall rate 50.0%\n"
            "Warning: some tasks did not complete 100 episodes; check logs\n",
            encoding="utf-8",
        )
        got = mce._stats_from_run_dir(run)
        assert got["success"] == 3 and got["episodes"] == 6
        assert got["overall_rate"] == 50.0

        # 缺 stats.txt -> 不炸, 只返回路径
        empty = tmp / "empty_run"
        empty.mkdir()
        got = mce._stats_from_run_dir(empty)
        assert "stats_file" in got and "overall_rate" not in got


# ---------------------------------------------------------------------------
# T9 sentinel 清单
# ---------------------------------------------------------------------------

def test_t9_sentinel_lists():
    yaml_path = REPO_ROOT / "configs" / "curriculum" / "robotwin_curriculum_v1.yaml"
    if not yaml_path.is_file():
        print("        (跳过: 未找到课程配置)")
        return

    import copy

    from robotwin_curriculum import (                        # noqa: E402
        load_curriculum, load_phase_defs, load_sentinel, load_skill_levels,
        resolve_phase_sentinel,
    )

    cfg = load_curriculum(yaml_path)
    t2l = load_skill_levels(cfg)
    phases = load_phase_defs(cfg)
    sentinel = load_sentinel(cfg, t2l)

    assert set(sentinel) == {"L1", "L2", "L3", "L4"}
    assert all(len(v) == 4 for v in sentinel.values())

    # 行数 4 / 8 / 12 / 16 且为前缀累积
    expect = {"P1": 4, "P2": 8, "P3": 12, "P4": 16}
    prev: list[str] = []
    for key in ("P1", "P2", "P3", "P4"):
        tasks = resolve_phase_sentinel(sentinel, phases[key]["levels"])
        assert len(tasks) == expect[key], (key, len(tasks))
        assert tasks[:len(prev)] == prev, f"{key} 不是前缀累积: {tasks}"
        prev = tasks

    # 生成的清单文件与配置一致 (存在才校验)
    for key, n in expect.items():
        f = Path("/data/train/phases") / f"phase{key[1:]}_eval.txt"
        if not f.is_file():
            continue
        lines = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines()]
        body = [ln for ln in lines if ln and not ln.startswith("#")]
        assert len(body) == n, (f, len(body), n)
        assert body == resolve_phase_sentinel(sentinel, phases[key]["levels"]), f

    # 越级 -> 报错
    bad = copy.deepcopy(cfg)
    bad["evaluation"]["sentinel"]["L1"][0] = "adjust_bottle"    # 这是 L2 的任务
    try:
        load_sentinel(bad, t2l)
    except RuntimeError:
        pass
    else:
        raise AssertionError("sentinel 越级应当报错")

    # 数量不对 -> 报错
    bad = copy.deepcopy(cfg)
    bad["evaluation"]["sentinel"]["L2"].pop()
    try:
        load_sentinel(bad, t2l)
    except RuntimeError:
        pass
    else:
        raise AssertionError("sentinel 数量不符应当报错")


# ---------------------------------------------------------------------------
# T10 episodes 透传 (eval client 的 test_num)
# ---------------------------------------------------------------------------

def test_t10_episodes_passthrough():
    with Sandbox() as tmp:
        # 指定回合数 -> 命令行里出现 --test_num N
        # 用一个**非默认值** (5) 来证明它确实被透传, 而不是碰巧撞上默认值
        plans = _plans(tmp, 1, 1, episodes=5)
        cmd = plans[0].jobs[0].cmd
        assert "--test_num" in cmd, cmd
        assert cmd[cmd.index("--test_num") + 1] == "5"

        # 不指定 -> 不带 --test_num, 让 eval client 用自己的官方默认 (100)
        plans = _plans(tmp / "none", 1, 1)
        cmd = plans[0].jobs[0].cmd
        assert "--test_num" not in cmd, cmd

        # CLI 默认 3 (对齐 curriculum yaml 的 evaluation.protocols.mini_eval);
        # --episodes 0 表示不覆盖
        ap = mce.build_parser()
        assert ap.parse_args(["--ckpt-root", str(tmp)]).episodes == 3
        assert ap.parse_args(["--ckpt-root", str(tmp), "--episodes", "0"]).episodes == 0
        assert ap.parse_args(["--ckpt-root", str(tmp), "--episodes", "8"]).episodes == 8


def test_t10b_eval_client_test_num_overridable():
    """eval client 的 test_num 必须可覆盖, 不能是硬编码常量。"""
    path = REPO_ROOT / "experiment" / "robotwin" / "eval_policy_client_lingbotvla.py"
    if not path.is_file():
        print("        (跳过: 未找到 eval client)")
        return
    src = path.read_text(encoding="utf-8")
    assert 'usr_args.get("test_num"' in src, "eval client 未从 usr_args 读取 test_num"
    assert "test_num = 2" not in src, "eval client 里仍残留 test_num = 2 硬编码"
    # 默认值必须是官方 100, 不能是 2
    assert 'usr_args.get("test_num", 100)' in src, "eval client 的 test_num 默认值不是官方 100"


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  [OK]   {name}")
        except Exception as exc:                              # noqa: BLE001
            failed.append((name, exc))
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
    print()
    print(f"  {len(tests) - len(failed)}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
