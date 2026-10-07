"""checkpoint 保存兜底 + 磁盘容量保护的单元测试。

无需 GPU, 无需 pytest:
    python tests/test_disk_guard.py

覆盖:
  T1  DCP save 正常
  T2  DCP save 抛异常 -> re-raise
  T3  HF async 正常
  T4  HF async 抛异常 -> hf_save_failed=True
  T5  多卡收到相同 stop / T5b 多卡收到相同 hf_failed reason
  T6  HF 失败后不再进入下一次 Checkpointer.save()
  T6b 余量不足 -> allow=False (reason=disk)
  T7  首个 checkpoint 放行 + max_used 只增不减
  T8  fail-open: 读盘失败 / 占用无效 / 收集异常 -> 放行

⚠️ 凡是断言依赖「剩余空间够不够」的用例（T5/T5b/T6/T6b/T7）都必须用
   `_FixedDisk(...)` 把 `disk_avail_gb` 钉住 —— 否则测试结果会随宿主机
   真实剩余空间变化（踩过：`/data` 只剩 37.7G 时 T7 假红）。
"""
import logging
import os
import sys
import tempfile

import torch.distributed as _real_dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingbotvla.utils import checkpoint_guard as cg                      # noqa: E402
from lingbotvla.utils.async_hf_checkpoint import (                       # noqa: E402
    AsyncHFCheckpointSaver,
    HFCheckpointResult,
)

TMP = tempfile.gettempdir()
_REAL_DIST = _real_dist          # T5 会替换 cg.dist, 之后必须恢复, 否则 dist_ready() 会炸


def _use_single_rank():
    """恢复真实 dist (未初始化) —— 即单卡语义。"""
    cg.dist = _REAL_DIST


class _FixedDisk:
    """把 `cg.disk_avail_gb` 固定成给定值（GB）。

    🔴 涉及容量判定的用例**必须**用它：否则断言依赖**宿主机真实剩余空间**。
    实测踩过：`/data` 剩 78G 时 T7 靠「78 >= 77」踩着 1G 余量通过；
    跑完一轮训练存档后只剩 37.7G，T7 立刻变红 —— 与被测代码毫无关系。
    单元测试不能依赖宿主磁盘状态。
    """

    def __init__(self, avail_gb):
        self.avail_gb = avail_gb
        self._orig = None

    def __enter__(self):
        self._orig = cg.disk_avail_gb
        cg.disk_avail_gb = lambda path: self.avail_gb
        return self

    def __exit__(self, *exc):
        cg.disk_avail_gb = self._orig
        return False



# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------

class FakeCheckpointer:
    """记录 save 调用; 可选地抛异常。"""

    def __init__(self, exc=None):
        self.exc = exc
        self.calls = []

    def save(self, save_path, state, global_steps=None):
        self.calls.append(global_steps)
        if self.exc is not None:
            raise self.exc


class RecordingSaver:
    """每次 drain 依次返回一批预设结果。"""

    def __init__(self, batches):
        self._batches = list(batches)

    def drain_pending_best_effort(self):
        return self._batches.pop(0) if self._batches else []


class BoomSaver:
    def drain_pending_best_effort(self):
        raise RuntimeError("drain exploded")


def _result(step=545, before=500.0, after=430.0, err="", failed=False):
    r = HFCheckpointResult(
        global_step=step, checkpoint_path="/x", hf_path="/x/hf_ckpt",
        error=err, hf_save_failed=failed,
    )
    r.disk_avail_before_gb = before
    r.disk_avail_after_gb = after
    r.checkpoint_used_gb = (before - after) if (before > 0 and after > 0) else -1.0
    return r


class FakeDist:
    """模拟 4 个 rank 的 broadcast_object_list。"""

    def __init__(self, rank, store):
        self.rank = rank
        self.store = store

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def get_rank(self):
        return self.rank

    def broadcast_object_list(self, payload, src=0):
        if self.rank == src:
            self.store["payload"] = list(payload)
        else:
            payload[:] = list(self.store["payload"])


# ---------------------------------------------------------------------------
# T1 / T2  DCP 保存
# ---------------------------------------------------------------------------

def test_t1_dcp_save_normal():
    ckpt = FakeCheckpointer()
    cg.dcp_save_or_abort(ckpt, TMP, {"model": None}, 545)     # 不应抛异常
    assert ckpt.calls == [545], ckpt.calls


def test_t2_dcp_save_failure_reraises():
    err = OSError(28, "No space left on device")
    ckpt = FakeCheckpointer(exc=err)
    try:
        cg.dcp_save_or_abort(ckpt, TMP, {"model": None}, 545)
    except OSError as got:
        assert got is err, "必须原样 re-raise 同一个异常"
    else:
        raise AssertionError("DCP 保存失败必须 re-raise, 不能吞掉")
    assert ckpt.calls == [545]


# ---------------------------------------------------------------------------
# T3 / T4  异步 HF 保存
# ---------------------------------------------------------------------------

def _make_saver():
    # dist 未初始化 -> _is_rank0() 为 True, 会建单线程 executor
    return AsyncHFCheckpointSaver(enabled=True, max_pending=1)


def _run_one(saver, ckpt_dir, before=500.0):
    return saver._run_hf_checkpoint(
        global_step=545,
        checkpoint_path=ckpt_dir,
        output_dir=ckpt_dir,
        ckpt_manager="dcp",
        save_ema=False,
        enable_fp32=True,
        model_assets=None,
        epoch=1,
        epoch_step=1,
        best_effort=True,
        disk_avail_before_gb=before,
    )


def test_t3_hf_async_normal():
    saver = _make_saver()
    saver._save_one_hf_checkpoint = lambda **kw: None        # 模拟写盘成功
    res = _run_one(saver, TMP)
    assert res.hf_success is True, res
    assert res.hf_save_failed is False, res
    assert res.error == "", res.error


def test_t4_hf_async_failure_sets_flag():
    saver = _make_saver()

    def _boom(**kw):
        raise OSError(28, "No space left on device")

    saver._save_one_hf_checkpoint = _boom
    res = _run_one(saver, TMP)
    assert res.hf_save_failed is True, "HF 失败必须打上 hf_save_failed 标记"
    assert res.hf_success is False
    assert "No space left" in res.error, res.error


# ---------------------------------------------------------------------------
# T5  多卡一致性
# ---------------------------------------------------------------------------

def test_t5_all_ranks_receive_same_stop():
    store = {}
    flags, maxes = [], []
    # 余量固定 10G ⇒ 1100G 的需求必然被拒（不依赖宿主机真实剩余空间）
    with _FixedDisk(10.0):
        for rank in range(4):
            cg.dist = FakeDist(rank, store)
            # 只有 rank0 有实测结果; 其余 rank 的 saver 不可用
            saver = RecordingSaver([[_result(before=2000.0, after=1000.0)]]) if rank == 0 else None
            allow, max_used, _, reason = cg.disk_guard_check(TMP, saver, 0.0, 1.1, label=f"rank{rank}")
            flags.append(allow)
            maxes.append(max_used)
            if rank == 0:
                assert reason == "disk", reason
            else:
                assert reason == "disk", f"rank{rank} 也必须收到同一 reason: {reason}"
    assert flags == [False] * 4, f"4 个 rank 必须一致 stop: {flags}"
    assert len(set(maxes)) == 1, f"4 个 rank 的 max_used 必须一致: {maxes}"


def test_t5b_hf_failure_reason_broadcast():
    store = {}
    with _FixedDisk(1000.0):          # 余量充足 ⇒ 必须走到 hf_failed 这条分支
        for rank in range(4):
            cg.dist = FakeDist(rank, store)
            saver = RecordingSaver([[_result(err="ENOSPC", failed=True)]]) if rank == 0 else None
            allow, _, _, reason = cg.disk_guard_check(TMP, saver, 0.0, 1.1)
            assert allow is False, f"rank{rank} 必须 stop"
            assert reason == "hf_failed", f"rank{rank} reason={reason}"


# ---------------------------------------------------------------------------
# T6  HF 失败后不再进入下一次 Checkpointer.save()
# ---------------------------------------------------------------------------

def test_t6_no_save_after_hf_failure():
    _use_single_rank()                       # 单卡
    ckpt = FakeCheckpointer()
    # 第 1 个检查点: 无历史 -> 放行 -> 存档 -> 之后 HF 失败
    # 第 2 个检查点: 收到 hf_save_failed -> allow=False -> 不得再存档
    saver = RecordingSaver([
        [],                                                   # ckpt@545 之前: 无历史
        [_result(step=545, err="ENOSPC", failed=True)],       # ckpt@1090 之前: HF 失败
    ])
    max_used = 0.0
    saved = []
    with _FixedDisk(1000.0):                  # 余量充足 ⇒ 拒绝只能来自 hf_failed
        for step in (545, 1090):
            allow, max_used, _, reason = cg.disk_guard_check(TMP, saver, max_used, 1.1, label=f"step {step}")
            if not allow:
                assert reason == "hf_failed", reason
                break
            cg.dcp_save_or_abort(ckpt, TMP, {"model": None}, step)
            saved.append(step)
    assert saved == [545], f"HF 失败后不得再存档, 实际存档: {saved}"
    assert ckpt.calls == [545], ckpt.calls


def test_t6b_disk_full_blocks_next_save():
    _use_single_rank()
    ckpt = FakeCheckpointer()
    saver = RecordingSaver([
        [_result(before=2000.0, after=1000.0)],               # 占用 1000G
    ])
    max_used = 0.0
    saved = []
    with _FixedDisk(10.0):                    # 余量只有 10G ⇒ 必然拒绝
        for step in (545, 1090):
            allow, max_used, _, reason = cg.disk_guard_check(TMP, saver, max_used, 1.1, label=f"step {step}")
            if not allow:
                assert reason == "disk", reason
                break
            cg.dcp_save_or_abort(ckpt, TMP, {"model": None}, step)
            saved.append(step)
    assert saved == [], f"余量不足时不应存档, 实际: {saved}"


# ---------------------------------------------------------------------------
# T7 / T8  正常路径与 fail-open
# ---------------------------------------------------------------------------

def test_t7_first_checkpoint_passes_and_tracks_max():
    _use_single_rank()
    # 余量固定 1000G ⇒ 与宿主机真实剩余空间解耦（曾因只剩 37.7G 而假红）
    with _FixedDisk(1000.0):
        allow, max_used, avail, reason = cg.disk_guard_check(
            TMP, RecordingSaver([[]]), 0.0, 1.1, label="first")
        assert allow is True and max_used == 0.0 and avail is not None and reason == ""
        assert abs(avail - 1000.0) < 0.01, avail

        allow, max_used, _, _ = cg.disk_guard_check(
            TMP, RecordingSaver([[_result(before=500.0, after=430.0)]]), 0.0, 1.1)
        assert allow is True and abs(max_used - 70.0) < 0.01, max_used

        # max 只增不减
        allow, max_used, _, _ = cg.disk_guard_check(
            TMP, RecordingSaver([[_result(before=100.0, after=90.0)]]), 500.0, 1.1)
        assert abs(max_used - 500.0) < 0.01, max_used


def test_t8_fail_open_paths():
    _use_single_rank()
    # 读盘失败
    allow, _, avail, reason = cg.disk_guard_check(
        "/nonexistent/xyz", RecordingSaver([[_result()]]), 0.0, 1.1)
    assert allow is True and avail is None and reason == ""
    # 占用实测无效
    allow, max_used, _, reason = cg.disk_guard_check(
        TMP, RecordingSaver([[_result(before=-1.0, after=430.0)]]), 0.0, 1.1)
    assert allow is True and max_used == 0.0 and reason == ""
    # 收集抛异常
    allow, _, avail, reason = cg.disk_guard_check(TMP, BoomSaver(), 0.0, 1.1)
    assert allow is True and avail is None and reason == ""


def test_t8b_disk_avail_gb_missing_path():
    assert cg.disk_avail_gb("/nonexistent/xyz") is None
    assert cg.disk_avail_gb(TMP) is not None


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main():
    if "-v" not in sys.argv:
        logging.getLogger("lingbotvla.utils.checkpoint_guard").setLevel(logging.CRITICAL)
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  [OK]   {name}")
        except Exception as exc:
            failed.append((name, exc))
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
    print()
    print(f"  {len(tests) - len(failed)}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
