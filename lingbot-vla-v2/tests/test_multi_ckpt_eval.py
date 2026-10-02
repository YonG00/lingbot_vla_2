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
  T6  dry-run: 不调用 subprocess, 不创建作业目录, 且**不覆盖**真跑的 summary.*
  T7  preflight: 幂等同步共享文件 (原子替换)
  T8  stats.txt 解析
  T9  sentinel 清单: 行数 4/8/12/16 + 前缀累积 + 越级拦截
  T10 episodes (test_num) 透传 + eval client 不再硬编码
  T11 静默期 (min_age_seconds) 默认关闭
  T11b 静默期补充: _READY 标记优先 / mtime 变老放行 / now= 注入 / complete 透出
  T12 失败的 checkpoint 不写进增量状态 (下次会重试)
  T13 逐任务表解析 + Level 标注 (task->Level 映射 / 交叉表 / 按 Level 汇总 / 优雅退化)
"""

from __future__ import annotations

import inspect
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
        # dry-run 的产物落到 summary.dryrun.*, **不碰**真跑的 summary.*
        assert (out / "summary.dryrun.txt").is_file()
        payload = json.loads((out / "summary.dryrun.json").read_text(encoding="utf-8"))
        assert payload["dry_run"] is True
        assert payload["batches"] == [2]
        assert payload["tasks"] == ["lift_pot", "click_bell"]
        assert len(payload["plans"]) == 2
        assert not (out / "summary.txt").exists(), "dry-run 不该创建 summary.txt"
        assert not (out / "summary.json").exists(), "dry-run 不该创建 summary.json"


def test_t6b_dry_run_does_not_clobber_real_report():
    """回归: dry-run **绝不能**覆盖真跑出来的 summary.txt/json。

    实测踩过 —— 在 base_1ckpt 上补跑一次 --dry-run 做计划预览, 就把 测1b 的
    真实结果 (含每 condition 耗时/成功率) 冲成了"未实际执行"的空壳,
    summary.json 里 duration_s/success 全变 None, 数据不可恢复。
    """
    with Sandbox() as tmp:
        make_ckpt(tmp, "expA", 1000, n_shards=1)
        tasks = tmp / "tasks.txt"
        tasks.write_text("lift_pot\n", encoding="utf-8")
        out = tmp / "out"
        out.mkdir(parents=True)

        # 先放一份"真跑"的产物
        real_txt = "真跑结果: 成功率 91.7% (11/12)\n"
        real_json = '{"dry_run": false, "elapsed_seconds": 424.0, "plans": []}\n'
        (out / "summary.txt").write_text(real_txt, encoding="utf-8")
        (out / "summary.json").write_text(real_json, encoding="utf-8")

        rc = mce.main([
            "--ckpt-root", str(tmp),
            "--task-list-file", str(tasks),
            "--output-base", str(out),
            "--dry-run", "--all",
        ])
        assert rc == 0, rc

        # 真跑产物**原封不动**
        assert (out / "summary.txt").read_text(encoding="utf-8") == real_txt
        assert (out / "summary.json").read_text(encoding="utf-8") == real_json
        # dry-run 自己的产物在别处
        assert (out / "summary.dryrun.txt").is_file()
        dry = json.loads((out / "summary.dryrun.json").read_text(encoding="utf-8"))
        assert dry["dry_run"] is True


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


def test_t11_min_age_default_off():
    """静默期 (min_age_seconds) 默认必须是 0 (关闭)。

    背景: 本机训练占满 4 卡, 与评测**时间上互斥** ⇒ 评测只能在训练退出后启动,
    此时确定没有写入者。若默认 120s, 刚训练完的最后一个 checkpoint 会被静默判为
    "仍在写入"而跳过 —— 那恰恰是最关心的那个, 且失败是静默的。
    ⇒ 默认关闭; 想启用 (轮询扫描场景) 时显式传正数。
    """
    ap = mce.build_parser()
    assert ap.parse_args(["--ckpt-root", "/tmp"]).min_age_seconds == 0.0, \
        "静默期默认值应为 0 (关闭)"
    assert ap.parse_args(
        ["--ckpt-root", "/tmp", "--min-age-seconds", "120"]
    ).min_age_seconds == 120.0, "显式传正数应生效"

    # 函数层默认也要一致, 否则 CLI 与实现会漂移
    sig = inspect.signature(mce.check_hf_ckpt)
    assert sig.parameters["min_age_seconds"].default == 0.0, \
        "check_hf_ckpt 的 min_age_seconds 默认值应为 0"

    # 语义确认: 0 = 不阻塞; 正数 = 拦截刚写过的目录
    with Sandbox() as tmp:
        d = make_ckpt(tmp, "expA", 1000)
        ok, reasons, _, _, _ = mce.check_hf_ckpt(d, min_age_seconds=0)
        assert ok and not reasons, f"min_age=0 不应阻塞: {reasons}"
        ok, reasons, _, _, _ = mce.check_hf_ckpt(d, min_age_seconds=3600)
        assert not ok and any("仍在写入" in r for r in reasons), reasons


def test_t11b_quiet_period_ready_marker_and_backdate():
    """静默期的补充判据: `_READY` 标记优先, 以及时间真的过去后放行。

    T11 只覆盖「0 不阻塞 / 正数拦截刚写过的」。这里补四件容易漏的事:
      1. `_READY` 存在时**跳过**静默期判定 (标记是权威信号, 见 scheduler 注释);
      2. mtime 变老后放行 —— 证明拦的是「新鲜度」, 不是「有没有 _READY」;
      3. `now=` 注入等效于时间流逝, 可以不碰 mtime 就测「将来」;
      4. `discover_checkpoints` 要把 complete 标志透出来 (不完整的**也要返回**,
         好让报告能说清「为什么没评测它」), 而不是静默丢掉。
    """
    with Sandbox() as tmp:
        d = make_ckpt(tmp, "expA", 1000)

        # 1) 新鲜 + 静默期 -> 拦
        ok, reasons, _, _, _ = mce.check_hf_ckpt(d, min_age_seconds=120)
        assert not ok and any("仍在写入" in r for r in reasons), reasons

        # 2) 同一目录放上 _READY -> 放行 (标记优先于静默期)
        (d / mce.READY_MARKER).write_text("", encoding="utf-8")
        ok, reasons, _, _, _ = mce.check_hf_ckpt(d, min_age_seconds=120)
        assert ok, f"_READY 应覆盖静默期: {reasons}"
        (d / mce.READY_MARKER).unlink()

        # 3) 无 _READY, 但 mtime 拨回 10 分钟前 -> 放行
        old = time.time() - 600
        for p in d.iterdir():
            os.utime(p, (old, old))
        os.utime(d, (old, old))
        ok, reasons, _, _, _ = mce.check_hf_ckpt(d, min_age_seconds=120)
        assert ok, f"静默期已过应放行: {reasons}"

        # 4) now= 注入: 不碰 mtime, 等价于把时钟往前拨 300s
        d2 = make_ckpt(tmp, "expB", 1000)
        ok, reasons, _, _, _ = mce.check_hf_ckpt(
            d2, min_age_seconds=120, now=time.time() + 300)
        assert ok, f"now 注入应等效于时间流逝: {reasons}"

        # 5) discover_checkpoints 透出 complete 标志 (两种取值都要看)
        found = {c.exp_name: c.complete
                 for c in mce.discover_checkpoints(tmp, min_age_seconds=120)}
        assert len(found) == 2, f"不完整的也要返回: {found}"
        assert found["expA"] is True, found      # mtime 已拨老
        assert found["expB"] is False, found     # 新鲜 -> 被静默期拦

        found0 = {c.exp_name: c.complete
                  for c in mce.discover_checkpoints(tmp, min_age_seconds=0)}
        assert all(found0.values()), f"min_age=0 时两个都该可用: {found0}"


def _fake_job(rc):
    return mce.Job(ckpt_tag="t", step=1, hf_ckpt="/x", condition="clean",
                   task_config="demo_clean", slot_index=0, gpus=[0],
                   start_port=9330, end_port=9333, output_base="/o",
                   log_file="/l", returncode=rc)


def test_t12_failed_ckpt_not_registered():
    """失败的 checkpoint 不得写进增量状态, 否则下次会被永久跳过。

    覆盖两层:
      1. ``job_failed``: ``returncode is None`` (线程内异常 / 未跑完) 也算失败;
      2. ``main``: 失败的 ckpt 不被 save_state 登记 → 下次仍判为「新增」并重试。
    """
    # ---- 1. job_failed 语义 ----
    assert mce.job_failed(_fake_job(1)) is True
    assert mce.job_failed(_fake_job(137)) is True
    assert mce.job_failed(_fake_job(None)) is True, "未跑完 (None) 必须算失败"
    assert mce.job_failed(_fake_job(0)) is False

    # ---- 2. main: 失败的不登记, 成功的不受影响 ----
    with Sandbox() as tmp:
        make_ckpt(tmp, "expA", 1000, n_shards=1)
        make_ckpt(tmp, "expA", 2000, n_shards=1)
        tasks = tmp / "tasks.txt"
        tasks.write_text("lift_pot\n", encoding="utf-8")
        out = tmp / "out"
        state = out / "eval_state.json"

        seen: list[list[str]] = []
        orig_sched = mce.run_scheduler

        def _run(plans, **kw):
            """不真起 launcher: 记录选中了谁, 再按预设结果填 returncode。"""
            seen.append([p.tag for p in plans])
            for p in plans:
                for j in p.jobs:
                    j.returncode = 1 if p.step == 1000 else 0
            return plans

        argv = ["--ckpt-root", str(tmp),
                "--task-list-file", str(tasks),
                "--output-base", str(out),
                "--no-preflight", "--all",
                "--max-parallel-checkpoints", "1"]

        mce.run_scheduler = _run
        try:
            rc = mce.main(argv)
            assert rc == 3, f"有失败作业时应返回 3, 实际 {rc}"

            saved = mce.load_state(state)
            assert "expA_step2000" in saved, "成功的 ckpt 应被登记"
            assert "expA_step1000" not in saved, \
                "失败的 ckpt 不得被登记 (否则下次永久跳过)"

            # 第二次: 去掉 --all 走增量, 让 step1000 这次成功
            def _run_ok(plans, **kw):
                seen.append([p.tag for p in plans])
                for p in plans:
                    for j in p.jobs:
                        j.returncode = 0
                return plans

            mce.run_scheduler = _run_ok
            rc2 = mce.main([a for a in argv if a != "--all"])
            assert rc2 == 0, f"全部成功时应返回 0, 实际 {rc2}"
        finally:
            mce.run_scheduler = orig_sched

        assert seen[0] == ["expA_step1000", "expA_step2000"], seen[0]
        # 关键断言: 上次失败的必须被重新选中, 上次成功的必须跳过
        assert seen[1] == ["expA_step1000"], \
            f"失败的 ckpt 应被重试且只重试它, 实际 {seen[1]}"

        # 修好后 step1000 也应被登记
        saved2 = mce.load_state(state)
        assert "expA_step1000" in saved2 and "expA_step2000" in saved2, saved2


def test_t12b_run_plan_records_thread_exception():
    """作业线程内抛异常时, 原因必须落到 job.error (否则 rc=None 会被当成成功)。"""
    plan = mce.CkptPlan(index=0, tag="expA_step1000", step=1000,
                        hf_ckpt="/x", slot_index=0)
    plan.jobs = [_fake_job(None), _fake_job(None)]

    orig = mce.run_job

    def _boom(job, **kw):
        raise RuntimeError("模拟 launcher 启动失败")

    mce.run_job = _boom
    try:
        out = mce.run_plan(plan, dry_run=False)
    finally:
        mce.run_job = orig

    assert out.jobs[0].error.startswith("RuntimeError:"), out.jobs[0].error
    assert out.jobs[0].returncode is None
    assert mce.job_failed(out.jobs[0]) is True, "异常的作业必须算失败"
    # 第一个作业炸了就停, 不再跑第二个 condition
    assert out.jobs[1].error == "", "首个作业异常后不应继续跑剩余 condition"
    assert out.jobs[1].returncode is None


# ---------------------------------------------------------------------------
# T13 逐任务表解析 + Level 标注
# ---------------------------------------------------------------------------

# 一份**真实格式**的 stats.txt (照抄 launcher L664-738 的 printf 格式)。
# 关键点: 横幅/表头/Summary 行都不许被误判成任务行。
REAL_STATS = """\
============================================
  Eval Result Stats
  Time: 2026-10-02 23:34:02
  Model: lingbot-vla-v2-6b-robotwin_50k
  Model path: /data/models/lingbot-vla-v2-6b-robotwin/x/global_step_50000/hf_ckpt
  Tasks: 4
  Task Config: demo_clean
  Inference: 4 GPU x 1/GPU = 4 slots
  Precision: use_bf16=False, use_fp32=True, use_compile=False
  Result: 4 done, 0 skipped
============================================

Task                           Time(s)    Done(100)    Success/Total Rate
--------------------------------------------------------------------------------
lift_pot                       230        NO(3/100)    3/3        100.0%
click_alarmclock               219        NO(3/100)    2/3        66.7%
turn_switch                    250        NO(3/100)    3/3        100.0%
place_shoe                     198        NO(3/100)    -          -
--------------------------------------------------------------------------------
Summary: total 897s, success 8/9, overall rate 88.9%
Warning: some tasks did not complete 100 episodes; check logs
============================================
"""


def _fake_plan(tag, step, cond, *, per_task, success, episodes, rate):
    """造一个已完成 (或失败) 的 plan, 只填渲染需要的字段。"""
    job = mce.Job(
        ckpt_tag=tag, step=step, hf_ckpt=f"/models/{tag}/hf_ckpt",
        condition=cond,
        task_config=mce.CONDITIONS[cond],
        slot_index=0, gpus=[0], start_port=9330, end_port=9333,
        output_base="/tmp/out", log_file="/tmp/out/log",
    )
    job.returncode = 0
    job.success, job.episodes, job.overall_rate = success, episodes, rate
    job.per_task = per_task
    return mce.CkptPlan(index=0, tag=tag, step=step,
                        hf_ckpt=job.hf_ckpt, slot_index=0, jobs=[job])


def test_t13_per_task_parsing():
    with Sandbox() as tmp:
        run = tmp / "expA_50k_demo_clean_20261002_233402"
        run.mkdir(parents=True)
        (run / "stats.txt").write_text(REAL_STATS, encoding="utf-8")
        got = mce._stats_from_run_dir(run)

        # 汇总行仍然正确
        assert got["success"] == 8 and got["episodes"] == 9, got
        assert got["overall_rate"] == 88.9, got
        # 头部元信息
        assert got["task_config_in_stats"] == "demo_clean", got
        assert got["run_time"] == "2026-10-02 23:34:02", got
        assert got["model_path_in_stats"].endswith("global_step_50000/hf_ckpt"), got

        # 逐任务表: 恰好 4 行, 横幅/表头/Summary 一个都没混进来
        rows = got["per_task"]
        assert len(rows) == 4, rows
        assert [r["task"] for r in rows] == [
            "lift_pot", "click_alarmclock", "turn_switch", "place_shoe"], rows
        assert rows[0] == {"task": "lift_pot", "duration_s": 230,
                           "done_mark": "NO(3/100)", "success": 3,
                           "episodes": 3, "rate": 100.0}, rows[0]
        assert rows[1]["rate"] == 66.7 and rows[1]["success"] == 2, rows[1]
        # 没跑出成功率的行 -> None, 不能当成 0%
        assert rows[3]["success"] is None and rows[3]["rate"] is None, rows[3]

        # 没有逐任务表时不要凭空造 per_task
        bare = tmp / "bare"
        bare.mkdir()
        (bare / "stats.txt").write_text(
            "Summary: total 1s, success 0/0, overall rate 0.0%\n", encoding="utf-8")
        assert "per_task" not in mce._stats_from_run_dir(bare)

        # 单行解析器的负样本: 表头 / 横幅 / 元信息行都不算任务行
        for junk in (
            "Task                           Time(s)    Done(100)    Success/Total Rate",
            "--------------------------------------------------------------------------------",
            "============================================",
            "  Result: 4 done, 0 skipped",
            "  Inference: 4 GPU x 1/GPU = 4 slots",
            "  Precision: use_bf16=False, use_fp32=True, use_compile=False",
            "Summary: total 897s, success 8/9, overall rate 88.9%",
            "  Time: 2026-10-02 23:34:02",
        ):
            assert mce._parse_task_row(junk) is None, junk


def test_t13b_level_mapping_and_render():
    with Sandbox() as tmp:
        # --- task -> Level 映射 ---
        yml = tmp / "curriculum.yaml"
        yml.write_text(
            "skill_levels:\n"
            "  L1:\n    tasks:\n      - lift_pot\n      - click_alarmclock\n"
            "  L2:\n    tasks:\n      - turn_switch\n"
            "  L3:\n    tasks: []\n"
            "  L4:\n    tasks:\n      - place_shoe\n",
            encoding="utf-8")
        levels = mce.load_task_levels(yml)
        assert levels == {"lift_pot": "L1", "click_alarmclock": "L1",
                          "turn_switch": "L2", "place_shoe": "L4"}, levels

        # 文件不存在 / 结构不符 -> 空 dict, 不抛异常 (评测不该被报告拖死)
        assert mce.load_task_levels(tmp / "nope.yaml") == {}
        bad = tmp / "bad.yaml"
        bad.write_text("dataset:\n  name: x\n", encoding="utf-8")
        assert mce.load_task_levels(bad) == {}

        # --- 渲染 ---
        rows = mce._stats_from_run_dir(
            (lambda d: (d.mkdir(parents=True, exist_ok=True),
                        (d / "stats.txt").write_text(REAL_STATS, encoding="utf-8"),
                        d)[-1])(tmp / "run50k"))["per_task"]
        plan = _fake_plan("robotwin@50k", 50000, "clean", per_task=rows,
                          success=8, episodes=9, rate=88.9)
        text = mce.render_summary([plan], skipped={}, dry_run=False,
                                  task_levels=levels)

        # 1) 逐任务表在, 且带 Level
        assert "逐任务成功率" in text, text
        assert "lift_pot[L1]" in text, text
        assert "turn_switch[L2]" in text, text
        # 2) 单元格 = 成功率(成功/总)
        assert "100%(3/3)" in text, text
        assert "67%(2/3)" in text, text
        # 3) 没数据的任务是 "-", 不能渲染成 0%
        assert "place_shoe[L4]" in text, text
        # 4) 总览表在, 且给出了 condition 级结果
        assert "总览" in text, text
        assert "88.9% (8/9)" in text, text
        # 5) Level 分组标题 (整行宽的横线夹 L1), 且在任务行之前
        assert "--- L1 ---" in text, text
        assert text.index("--- L1 ---") < text.index("lift_pot[L1]"), text
        # 6) 报告的 Level 来源要写**真实**路径, 不能硬编码 (--curriculum-yaml 可被覆盖)
        tagged = mce.render_summary([plan], skipped={}, dry_run=False,
                                    task_levels=levels, curriculum_yaml=yml)
        assert str(yml) in tagged, tagged
        assert "robotwin_curriculum_v1.yaml" not in tagged, tagged
        # 7) 这份 yaml 有 3 个 Level -> 每组末尾出 "Lx 小计"
        assert "L1 小计" in text and "L2 小计" in text, text
        # 8) 老版那三段重复信息必须消失: 不再逐个作业罗列 rc / run_dir / 按 Level
        assert "按 Level:" not in text, text
        assert "Clean vs Randomized 对比" not in text, text

        # 没有 Level 映射时优雅退化: 没有 Lv 列, 但表还在
        plain = mce.render_summary([plan], skipped={}, dry_run=False)
        assert "逐任务成功率" in plain, plain
        assert "[L1]" not in plain, plain
        assert "按 Level:" not in plain, plain
        assert "lift_pot" in plain, plain

        # dry-run 不渲染逐任务表 (没有数据)
        dry = mce.render_summary([plan], skipped={}, dry_run=True,
                                 task_levels=levels)
        assert "逐任务成功率" not in dry, dry
        assert "总览" not in dry, dry


def _job(tag="t", step=1, cond="clean", **kw):
    return mce.Job(ckpt_tag=tag, step=step, hf_ckpt="/x", condition=cond,
                   task_config=mce.CONDITIONS.get(cond, "demo_clean"),
                   slot_index=0, gpus=[0], start_port=9330, end_port=9333,
                   output_base="/tmp", log_file="/tmp/l", **kw)


def test_t13c_helpers_and_level_subtotal():
    # --- _agg: 累加; 无分母时给 None (不能被 _cell 渲染成 0%(0/0)) ---
    idx = {"a": {"task": "a", "success": 3, "episodes": 3},
           "b": {"task": "b", "success": 1, "episodes": 3},
           "c": {"task": "c", "success": None, "episodes": None}}
    assert mce._agg(idx, ["a", "b"]) == {"success": 4, "episodes": 6}
    assert mce._agg(idx, ["c"]) == {"success": None, "episodes": None}
    assert mce._agg(idx, ["a", "nope"]) == {"success": 3, "episodes": 3}
    assert mce._agg(idx, []) == {"success": None, "episodes": None}
    assert mce._cell(mce._agg(idx, [])) == "-"
    # 全 0 分也要保留 (0/3 是有效结果, 不是"没数据")
    zero = {"z": {"task": "z", "success": 0, "episodes": 3}}
    assert mce._cell(mce._agg(zero, ["z"])) == "0%(0/3)"

    # --- _fmt_dur: 用户要求「X分Ys」 ---
    assert mce._fmt_dur(None) == "-"
    assert mce._fmt_dur(45) == "45秒"
    assert mce._fmt_dur(59.6) == "1分0秒"          # 四舍五入到 60 就进位
    assert mce._fmt_dur(59.4) == "59秒"
    assert mce._fmt_dur(238) == "3分58秒"
    assert mce._fmt_dur(424) == "7分4秒"
    assert mce._fmt_dur(3600) == "60分0秒"

    # --- _job_ok / _overview_cell ---
    ok = _job()
    ok.returncode = 0
    ok.success, ok.episodes, ok.overall_rate = 11, 12, 91.7
    assert mce._job_ok(ok) and mce._overview_cell(ok) == "91.7% (11/12)"
    bad = _job()
    bad.returncode = 3
    bad.success, bad.episodes, bad.overall_rate = 1, 12, 8.3
    assert not mce._job_ok(bad) and mce._overview_cell(bad) == "失败"
    dead = _job()
    assert not mce._job_ok(dead) and mce._overview_cell(dead) == "未跑完"
    crashed = _job()
    crashed.returncode = 0
    crashed.error = "boom"
    assert not mce._job_ok(crashed) and mce._overview_cell(crashed) == "失败"
    assert mce._overview_cell(None) == "-"

    # --- 多 Level 时才出小计行; 且小计 = 该组任务的累加 ---
    rows = [
        {"task": "lift_pot", "success": 3, "episodes": 3, "rate": 100.0},
        {"task": "click_alarmclock", "success": 2, "episodes": 3, "rate": 66.7},
        {"task": "adjust_bottle", "success": 0, "episodes": 3, "rate": 0.0},
    ]
    plan = _fake_plan("ck", 1000, "clean", per_task=rows, success=5, episodes=9,
                      rate=55.6)
    two = mce.render_per_task_table(
        [plan], task_levels={"lift_pot": "L1", "click_alarmclock": "L1",
                             "adjust_bottle": "L2"})
    joined = "\n".join(two)
    assert "L1 小计" in joined and "L2 小计" in joined, joined
    assert "83%(5/6)" in joined, joined              # L1 = 5/6
    assert "0%(0/3)" in joined, joined               # L2 = 0/3 (保留 0, 不是 "-")
    assert "--- L1 ---" in joined and "--- L2 ---" in joined, joined

    # 单个 Level 时不出小计
    one = mce.render_per_task_table(
        [plan], task_levels={"lift_pot": "L1", "click_alarmclock": "L1",
                             "adjust_bottle": "L1"})
    assert "小计" not in "\n".join(one), "\n".join(one)

    # --- 总览表: 一行一 ckpt, 一列一 condition, 带耗时 ---
    a = _job("ckA", 1000, "clean")
    a.returncode, a.success, a.episodes, a.overall_rate, a.duration_s = \
        0, 11, 12, 91.7, 238.0
    b = _job("ckA", 1000, "randomized")
    b.returncode, b.success, b.episodes, b.overall_rate, b.duration_s = \
        0, 10, 12, 83.3, 186.0
    c = _job("ckB", 2000, "clean")
    c.returncode = 1
    plan_a = mce.CkptPlan(index=0, tag="ckA", step=1000, hf_ckpt="/x",
                          slot_index=0, jobs=[a, b])
    plan_b = mce.CkptPlan(index=1, tag="ckB", step=2000, hf_ckpt="/y",
                          slot_index=1, jobs=[c])
    ov = "\n".join(mce.render_overview([plan_a, plan_b], task_levels={}))
    assert "91.7% (11/12)" in ov and "83.3% (10/12)" in ov, ov
    assert "失败" in ov, ov
    assert "7分4秒" in ov, ov                        # 238 + 186
    assert ov.count("clean") and ov.count("randomized"), ov


def test_t13d_curriculum_yaml_cli_default():
    """--curriculum-yaml 必须存在且默认指向仓库里的课程 yaml。"""
    ap = mce.build_parser()
    args = ap.parse_args(["--ckpt-root", "/tmp", "--phase", "1"])
    assert args.curriculum_yaml == str(mce.DEFAULT_CURRICULUM_YAML), args.curriculum_yaml
    assert mce.DEFAULT_CURRICULUM_YAML.name == "robotwin_curriculum_v1.yaml"
    # 仓库里真的有这个文件 (报告 Level 的来源)
    assert mce.DEFAULT_CURRICULUM_YAML.is_file(), mce.DEFAULT_CURRICULUM_YAML
    # 真 yaml 能读出 50 个任务 (只在有 pyyaml 时校验)
    levels = mce.load_task_levels()
    try:
        import yaml                                      # noqa: F401
        has_yaml = True
    except Exception:                                    # noqa: BLE001
        has_yaml = False
    if has_yaml:
        assert len(levels) == 50, len(levels)
        assert levels["lift_pot"] == "L1", levels.get("lift_pot")
        assert levels["place_shoe"] == "L1", levels.get("place_shoe")
        assert set(levels.values()) == {"L1", "L2", "L3", "L4"}, set(levels.values())


def test_t13e_table_alignment():
    """表格必须**真的**对齐 —— 表头/数据/合计行的显示宽度要一致。

    这是回归测试: 初版用 len() 算宽度, 导致
      (a) `合计` 被当 2 列 (实际 4 列) → 整行右移
      (b) 分隔线比表头行短 → 视觉错位
    """
    # CJK 按 2 列算
    assert mce._dw("合计") == 4, mce._dw("合计")
    assert mce._dw("task") == 4
    assert mce._dw("中文abc") == 4 + 3
    assert mce._pad("合计", 6) == "合计  ", repr(mce._pad("合计", 6))
    assert mce._pad("ab", 5, ">") == "   ab"
    assert mce._pad("ab", 5, "^") == " ab  "
    # 宽度不够时不抛异常, 原样返回
    assert mce._pad("abcdef", 2) == "abcdef"
    # 仓库内路径显示相对路径, 仓库外保持绝对路径
    assert mce._short_path(mce.DEFAULT_CURRICULUM_YAML) == \
        "configs/curriculum/robotwin_curriculum_v1.yaml"
    assert mce._short_path("/tmp/elsewhere.yaml") == "/tmp/elsewhere.yaml"

    levels = {"lift_pot": "L1", "click_alarmclock": "L1",
              "turn_switch": "L1", "place_shoe": "L1"}
    rows = [
        {"task": "click_alarmclock", "duration_s": 220, "done_mark": "NO(3/100)",
         "success": 3, "episodes": 3, "rate": 100.0},
        {"task": "turn_switch", "duration_s": 225, "done_mark": "NO(3/100)",
         "success": 2, "episodes": 3, "rate": 66.7},
        {"task": "lift_pot", "duration_s": 230, "done_mark": "NO(3/100)",
         "success": 3, "episodes": 3, "rate": 100.0},
        {"task": "place_shoe", "duration_s": 230, "done_mark": "NO(3/100)",
         "success": None, "episodes": None, "rate": None},
    ]
    plan = _fake_plan("robotwin@50k", 50000, "clean", per_task=rows,
                      success=8, episodes=9, rate=88.9)
    lines = mce.render_per_task_table([plan], task_levels=levels,
                                      curriculum_yaml="x.yaml")

    # 表头行之后到「格式:」之前 = 表体 (排除纯横线的分隔行)
    def _is_sep(line: str) -> bool:
        return line.startswith("  ") and set(line.strip()) == {"-"}

    start = next(i for i, l in enumerate(lines) if l.lstrip().startswith("task[Lv]"))
    end = next(i for i, l in enumerate(lines) if l.lstrip().startswith("单元格 ="))
    table = [l for l in lines[start + 1:end]
             if l.startswith("  ") and l.strip() and not _is_sep(l)]
    # 4 任务 + 1 条 Level 分组标题 + 合计
    assert len(table) == 6, table
    assert any("L1" in l for l in table), table      # 分组标题还在

    widths = {mce._dw(l) for l in table}
    assert len(widths) == 1, (widths, table)
    width = widths.pop()

    # 分隔线的宽度也要和表体一致
    seps = [l for l in lines[start + 1:end] if _is_sep(l)]
    assert seps, lines
    assert all(mce._dw(s) == width for s in seps), (seps, width)

    # 数据行去掉尾部空格后仍等宽 (即每列右边界重合)
    end_pos = {mce._dw(l.rstrip()) for l in table[1:]}
    assert len(end_pos) == 1, (end_pos, table)
    # 数值列右对齐 -> 两个 "100%(3/3)" 结尾列位置相同
    joined = "\n".join(table)
    assert "100%(3/3)" in joined and "67%(2/3)" in joined, joined
    # 没跑出成功率的任务占位 "-", 不能是 0%
    assert "0%(0/" not in joined, joined
    # 合计行用中文, 显示宽度仍要与其它行一致 (len() 会少算 2 列)
    total_row = next(l for l in table if "合计" in l)
    assert mce._dw(total_row) == width, (mce._dw(total_row), width)


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
