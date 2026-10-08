"""build_gmean_thresholds.py 的**补充**边界测试（审查方添加，独立文件，不修改原作者测试）。

动机：原作者 `tests/test_gmean_thresholds.py` 实测覆盖率 86%，未覆盖的 29 条语句**几乎全是额外的
fail-fast 分支**（非法 JSON / 缺任务 / 指纹不符 / nmse 自相矛盾 / 空文件 / val_trajs<2 /
split 目录异常 / high_cv_policy 非法 / baseline 无指纹 / GMean 上溢 / --ref-log 缺 --split-dir /
写后校验失败）。本文件把这些分支逐一钉住，避免将来被"优化掉"。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lingbotvla.auto_learning.baseline import BaselineStore, FixedBaseline, task_fingerprint
from lingbotvla.auto_learning.decision.thresholds import ThresholdsError
from lingbotvla.auto_learning.tools import build_gmean_thresholds as G

CFG = "cfg-extra"


def _store(tmp_path: Path, *, tasks=("a", "b"), mse=0.4, fingerprint: bool = True) -> BaselineStore:
    p = tmp_path / "baseline.json"
    st = BaselineStore(path=str(p), config_fingerprint=CFG)
    for t in tasks:
        st.put(FixedBaseline(task=t, mse=mse, mu=(0.0,),
                             fingerprint=task_fingerprint(CFG, "s") if fingerprint else ""))
    st.save()
    return BaselineStore.load(str(p))


def _rows(rows, tmp_path: Path, name="ref.jsonl") -> str:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def _good_rows(task_a="a", task_b="b"):
    return ([{"task": task_a, "traj": i, "mse": 0.001} for i in range(10)] +
            [{"task": task_b, "traj": 100 + i, "mse": 0.002} for i in range(10)])


# ---------------------------------------------------------------- _number / JSONL 解析
def test_bool_mse_is_rejected(tmp_path):
    """JSON 真布尔（不是字符串 "true"）走 `_number` 的布尔分支。"""
    with pytest.raises(ThresholdsError, match="布尔值"):
        G.load_reference(_rows([{"task": "a", "traj": 1, "mse": True}], tmp_path), _store(tmp_path))


def test_malformed_json_line_is_rejected(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"task": "a", "traj": 1, "mse": 0.1}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ThresholdsError, match="JSON 解析失败"):
        G.load_reference(str(p), _store(tmp_path))


def test_non_object_json_line_is_rejected(tmp_path):
    p = tmp_path / "bad2.jsonl"
    p.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ThresholdsError, match="必须是 JSON object"):
        G.load_reference(str(p), _store(tmp_path))


def test_missing_or_empty_task_is_rejected(tmp_path):
    with pytest.raises(ThresholdsError, match="缺少非空 task"):
        G.load_reference(_rows([{"traj": 1, "mse": 0.1}], tmp_path), _store(tmp_path))
    # P4 观察（非缺陷，仍 fail-closed）：`task` 只做真值判断、没有像 `traj` 那样 strip()，
    # 所以全空白任务名走的是"不存在于 baseline store"这条分支，报错里名字不可见。
    with pytest.raises(ThresholdsError, match="不存在于 baseline store"):
        G.load_reference(_rows([{"task": "  ", "traj": 1, "mse": 0.1}], tmp_path), _store(tmp_path))


def test_unknown_task_is_rejected(tmp_path):
    with pytest.raises(ThresholdsError, match="不存在于 baseline store"):
        G.load_reference(_rows([{"task": "zzz", "traj": 1, "mse": 0.1}], tmp_path), _store(tmp_path))


def test_task_fingerprint_mismatch_is_rejected(tmp_path):
    rows = _good_rows()
    rows[0]["task_fingerprint"] = "WRONG"
    with pytest.raises(ThresholdsError, match="task_fingerprint"):
        G.load_reference(_rows(rows, tmp_path), _store(tmp_path))


def test_nmse_inconsistency_is_rejected(tmp_path):
    rows = _good_rows()
    rows[0]["nmse"] = 0.001                       # 实际应为 0.001/0.4 = 0.0025
    with pytest.raises(ThresholdsError, match="nmse 与 mse/baseline 不一致"):
        G.load_reference(_rows(rows, tmp_path), _store(tmp_path))


def test_empty_reference_file_is_rejected(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ThresholdsError, match="无有效数据"):
        G.load_reference(str(p), _store(tmp_path))


def test_reference_missing_a_task_is_rejected(tmp_path):
    with pytest.raises(ThresholdsError, match="参考数据缺任务"):
        G.load_reference(_rows(_good_rows(task_b="a"), tmp_path), _store(tmp_path))


# ---------------------------------------------------------------- --ref-log 模式
def _split(tmp_path: Path, tasks=("a", "b"), per_task=10, name="split") -> Path:
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    for i, t in enumerate(tasks):
        (d / f"{t}.val_ids.json").write_text(
            json.dumps(list(range(i * per_task, (i + 1) * per_task))), encoding="utf-8")
    return d


def _log(tmp_path: Path, ids_mse: dict, name="log.txt") -> str:
    p = tmp_path / name
    p.write_text("".join(f"MSE for trajectory {i}: {v}, MAE: 0.01\n" for i, v in ids_mse.items()),
                 encoding="utf-8")
    return str(p)


def test_ref_log_requires_val_trajs_ge_2(tmp_path):
    with pytest.raises(ThresholdsError, match="val_trajs 必须 >= 2"):
        G.load_reference_log(_log(tmp_path, {}), str(_split(tmp_path)), _store(tmp_path), val_trajs=1)


def test_split_task_not_in_baseline_is_rejected(tmp_path):
    d = _split(tmp_path, tasks=("a", "b", "zzz"))
    with pytest.raises(ThresholdsError, match="不在 baseline 中"):
        G.load_reference_log(_log(tmp_path, {}), str(d), _store(tmp_path))


def test_split_unreadable_json_is_rejected(tmp_path):
    d = _split(tmp_path)
    (d / "b.val_ids.json").write_text("{not-json", encoding="utf-8")
    with pytest.raises(ThresholdsError, match="无法读取"):
        G.load_reference_log(_log(tmp_path, {}), str(d), _store(tmp_path))


def test_split_insufficient_ids_is_rejected(tmp_path):
    d = tmp_path / "short"; d.mkdir()
    (d / "a.val_ids.json").write_text(json.dumps(list(range(3))), encoding="utf-8")
    (d / "b.val_ids.json").write_text(json.dumps(list(range(10, 20))), encoding="utf-8")
    with pytest.raises(ThresholdsError, match="val 轨迹不足"):
        G.load_reference_log(_log(tmp_path, {}), str(d), _store(tmp_path))


def test_split_illegal_traj_id_type_is_rejected(tmp_path):
    d = tmp_path / "illegal"; d.mkdir()
    (d / "a.val_ids.json").write_text(json.dumps(["x"] * 10), encoding="utf-8")
    (d / "b.val_ids.json").write_text(json.dumps(list(range(10, 20))), encoding="utf-8")
    with pytest.raises(ThresholdsError, match="非法轨迹编号"):
        G.load_reference_log(_log(tmp_path, {}), str(d), _store(tmp_path))


# ---------------------------------------------------------------- build_thresholds 参数/护栏
@pytest.mark.parametrize("policy", ["nope", "", None])
def test_invalid_high_cv_policy_is_rejected(tmp_path, policy):
    with pytest.raises(ThresholdsError, match="high_cv_policy"):
        G.build_thresholds({"a": [0.001] * 10, "b": [0.002] * 10}, _store(tmp_path),
                           multiplier=10, high_cv_policy=policy, reference="r")


def test_baseline_without_fingerprint_is_rejected(tmp_path):
    st = _store(tmp_path)
    st.config_fingerprint = ""
    with pytest.raises(ThresholdsError, match="缺 config_fingerprint"):
        G.build_thresholds({"a": [0.001] * 10, "b": [0.002] * 10}, st,
                           multiplier=10, reference="r")


def test_gmean_overflow_is_rejected(tmp_path):
    with pytest.raises(ThresholdsError, match="溢出"):
        G.build_thresholds({"a": [1e308] * 10, "b": [1e308] * 10}, _store(tmp_path),
                           multiplier=1e10, reference="r")


# ---------------------------------------------------------------- CLI 与写出
def test_cli_ref_log_without_split_dir_errors(tmp_path):
    with pytest.raises(SystemExit):
        G.main(["--ref-log", str(tmp_path / "x.log"),
                "--baseline", str(_store(tmp_path).path),
                "--reference", "r", "--multiplier", "10", "-o", str(tmp_path / "o.json")])


def test_write_verify_failure_keeps_old_output(tmp_path, monkeypatch):
    """写后回读校验失败（模拟 PassThresholds.load 返回不一致结果）⇒ 抛错且旧产物不被替换。"""
    st = _store(tmp_path)
    ref = _rows(_good_rows(), tmp_path)
    out = tmp_path / "out.json"
    out.write_text("OLD", encoding="utf-8")

    real_load = G.PassThresholds.load

    def fake_load(path, **kw):
        obj = real_load(path, **kw)
        obj.tasks = {}
        return obj

    monkeypatch.setattr(G.PassThresholds, "load", staticmethod(fake_load))
    with pytest.raises(ThresholdsError, match="写入后阈值表校验失败"):
        G.main(["--ref-per-traj", ref, "--baseline", str(st.path),
                "--reference", "r", "--multiplier", "10", "-o", str(out)])
    assert out.read_text(encoding="utf-8") == "OLD"
    assert not list(tmp_path.glob("out.json.*.tmp"))       # 临时文件已清理


def test_cli_ok_writes_and_reloads(tmp_path):
    st = _store(tmp_path)
    ref = _rows(_good_rows(), tmp_path)
    out = tmp_path / "ok.json"
    assert G.main(["--ref-per-traj", ref, "--baseline", str(st.path),
                   "--reference", "r", "--multiplier", "220", "-o", str(out)]) == 0
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["metric"] == "mse" and raw["stat"] == "geomean"
    assert raw["calibration"]["source_sha256"]


def test_log_with_unrelated_lines_is_tolerated(tmp_path):
    """日志里混入无关行（不匹配 MSE 模式）应被跳过而不是报错（覆盖解析循环的 continue 分支）。"""
    d = _split(tmp_path)
    log = tmp_path / "mixed.log"
    log.write_text(
        "[INFO] loading model ...\n"
        "MSE for trajectory 0: 0.001, MAE: 0.01\n"
        "some unrelated progress line\n"
        + "".join(f"MSE for trajectory {i}: {0.001 if i < 10 else 0.002}, MAE: 0.01\n" for i in range(1, 20)),
        encoding="utf-8")
    per_task = G.load_reference_log(str(log), str(d), _store(tmp_path), val_trajs=10)
    assert set(per_task) == {"a", "b"} and all(len(v) == 10 for v in per_task.values())


def test_module_entrypoint_end_to_end_subprocess(tmp_path):
    """按 GUIDE 的调用方式用 `python -m ...` 跑一遍（覆盖 `sys.exit(main())` 入口）。"""
    import subprocess
    import sys
    st = _store(tmp_path)
    ref = _rows(_good_rows(), tmp_path)
    out = tmp_path / "sub.json"
    r = subprocess.run(
        [sys.executable, "-m", "lingbotvla.auto_learning.tools.build_gmean_thresholds",
         "--ref-per-traj", ref, "--baseline", str(st.path),
         "--reference", "global_step_50000", "--multiplier", "220", "-o", str(out)],
        cwd=str(Path(__file__).resolve().parents[1]),
        env={**__import__("os").environ, "PYTHONPATH": "."},
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    assert "可用 2/2" in r.stdout and out.is_file()
