#!/usr/bin/env python
"""CPU 2 进程（gloo + 真 FSDP2）复现 **AutoLearning bootstrap 的真实时序**：

    评测窗口（unshard/reshard）→ 评测窗口 → Hardness 扫描（根模块 no_grad 前向）
    → 第一个训练步（fwd + bwd + optimizer.step）→ 单元末评测 → 记账 all_gather

为什么需要它（2026-10-10 真机 hang 排查）
----------------------------------------
真机 2 卡（ROCm + FSDP2）现象：`_select()` 的 train/val 两次评测都正常出数后**完全静默**
20 分钟、进程存活、显存不释放。而 `_select()` 里紧接那两次评测之后的**唯一**一段代码就是
`HardnessScanner.scan()`（`scheduler.py:644`）—— 它跑在**任何训练前向之前**，且**一条 stdout
日志都没有**（只有扫完之后的 TB 标量）。

本脚本把那段时序在 CPU 上原样搭出来（用**真的** `lingbotvla.utils.open_loop_validation`
的评测窗口 + **真的** `HardnessScorer` / `RealHardnessScorer`），判据：

* 若「评测窗口 → 根模块 no_grad 前向 → 训练步」这条链在 2 进程下**挂死** ⇒ 真 bug（本脚本超时失败）；
* 若全程通过 ⇒ 「评测后各 rank 进入不同分支 / FSDP2 状态被评测窗口搞坏」这条假设被排除
  ⇒ 真机那 20 分钟更可能是 **Hardness 扫描本身很慢**（无日志、GPU 忙），需要用
  `rocm-smi` + `py-spy dump` 判别，而不是等。

用法::

    /opt/anaconda3/bin/python tools/al_bootstrap_order_fsdp2_repro.py            # 默认 2 进程
    /opt/anaconda3/bin/python tools/al_bootstrap_order_fsdp2_repro.py --timeout 300

本机（无 GPU）可直接跑：只有 torch 就够（torchdata / datasets / lerobot 用桩替代）。
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any, Dict, List


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# --------------------------------------------------------------------------- #
# 本机缺依赖（torchdata / datasets / lerobot / einops）时用**桩**保证能 import 真实模块。
# 只影响 import 期，被测的评测窗口 / FSDP2 代码一行都不改。
# --------------------------------------------------------------------------- #
def _install_import_stubs() -> None:
    def _mod(name: str, **attrs: Any) -> None:
        if name in sys.modules:
            return
        try:
            __import__(name)
            return
        except Exception:  # noqa: BLE001
            pass
        m = types.ModuleType(name)
        m.__path__ = []          # 桩模块当**包**用：`lerobot.utils.constants` 之类要能 import
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m

    _mod("torchdata")
    _mod("torchdata.stateful_dataloader", StatefulDataLoader=object)
    _mod("torchdata.stateful_dataloader.sampler", StatefulDistributedSampler=object)
    _mod("datasets", load_dataset=lambda *a, **k: None)
    _mod("datasets.distributed", split_dataset_by_node=lambda *a, **k: None)
    _mod("datasets.features")
    _mod("datasets.features.features",
         register_feature=lambda *a, **k: (a[0] if (len(a) == 1 and callable(a[0]))
                                           else (lambda cls: cls)))
    _mod("lerobot")
    _mod("lerobot.datasets")
    _mod("lerobot.datasets.lerobot_dataset", LeRobotDataset=object,
         LeRobotDatasetMetadata=object)
    _mod("lerobot.common")
    _mod("lerobot.common.datasets")
    _mod("lerobot.common.datasets.lerobot_dataset", LeRobotDataset=object,
         LeRobotDatasetMetadata=object)
    _mod("lerobot.common.datasets.utils", hf_transform_to_torch=lambda *a, **k: None)
    _mod("lerobot.utils", HF_LEROBOT_HOME="/tmp/_stub_lerobot_home")
    _mod("lerobot.utils.constants", HF_LEROBOT_HOME="/tmp/_stub_lerobot_home")
    _mod("lerobot.common.constants", HF_LEROBOT_HOME="/tmp/_stub_lerobot_home")
    _mod("einops", rearrange=lambda *a, **k: None, repeat=lambda *a, **k: None)


_install_import_stubs()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from torch import nn  # noqa: E402


class _Block(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.lin = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        return torch.relu(self.lin(x))


class _Cfg:
    n_action_steps = 4
    max_action_dim = 3
    loss_type = "L1_fm"
    action_fp32 = False
    use_cache = False
    align_params = None


class _Root(nn.Module):
    """形状模仿 `LingbotVlaV2Policy`：根单元自有参数 + 嵌套单元 + **自定义方法**。

    * `sample_actions()` ⇒ 开环评测路径（**绕过** `__call__`，所以绕过 FSDP 前向 hook）；
    * `forward(**batch)` ⇒ Hardness / 训练路径（**经过** 根 `__call__`）。
    """

    def __init__(self, d: int = 8):
        super().__init__()
        self.config = _Cfg()
        self.head = nn.Linear(d, d, bias=False)
        self.blocks = nn.ModuleList([_Block(d), _Block(d)])

    def forward(self, actions=None, state=None, noise=None, time=None, **kw):
        x = state if state is not None else actions
        for b in self.blocks:
            x = b(x)
        x = self.head(x)
        if actions is not None and actions.shape == x.shape:
            loss = (x - actions).abs().mean(dim=tuple(range(1, x.dim())))
        else:
            loss = x.reshape(x.shape[0], -1).mean(dim=1)
        return {"batch_mean_losses": loss}

    def sample_actions(self, x):
        h = self.head(x)
        for b in self.blocks:
            h = b(h)
        return h


class _FakeDataset(list):
    """`RealHardnessScorer` 只要求 `dataset[j]` 可取 item。"""


def _make_items(n: int, d: int = 8, n_action_steps: int = 4, max_action_dim: int = 3):
    torch.manual_seed(0)
    return _FakeDataset([
        {
            "actions": torch.randn(n_action_steps, max_action_dim),
            "joint_mask": torch.ones(n_action_steps, max_action_dim),
            "state": torch.randn(d),
        }
        for _ in range(n)
    ])


#: 评测体的张量上下文：
#: * ``repo``     = 用**仓库真实的** `_eval_tensor_context()`（默认；= 当前代码路径）
#: * ``nograd``   = 显式 `torch.no_grad()`（与修复后的 repo 等价，用于对照）
#: * ``inference``= 逐字复现修复前的 `torch.inference_mode()` ⇒ **必然失败**（bug 演示）
EVAL_MODE = os.environ.get("EVAL_MODE", "repo")


def _dump_inference_tensors(model: nn.Module) -> List[str]:
    """列出 FSDP2 内部**带 inference 标记**的张量（= 评测期在 InferenceMode 里建出来的）。

    `torch.inference_mode()` 里创建的张量**永远**带 inference 标记，之后在 InferenceMode
    **之外**对它做 inplace 写就会抛
    ``Inplace update to inference tensor outside InferenceMode``。
    FSDP2 的 `all_gather_copy_out` 正是 inplace 写进这些张量
    （`split_with_sizes_copy(all_gather_output, sizes, dim=1, out=...)`）。
    """
    hits: List[str] = []
    for name, mod in model.named_modules():
        getter = getattr(mod, "_get_fsdp_state", None)
        if not callable(getter):
            continue
        try:
            group = getattr(getter(), "_fsdp_param_group", None)
        except Exception:  # noqa: BLE001
            continue
        if group is None:
            continue
        label = name or "<root>"
        for i, p in enumerate(getattr(group, "fsdp_params", []) or []):
            for attr in ("unsharded_param", "sharded_param", "unsharded_param_data"):
                t = getattr(p, attr, None)
                if torch.is_tensor(t) and t.is_inference():
                    hits.append(f"{label}: fsdp_params[{i}].{attr}")
        out = getattr(getattr(group, "_all_gather_result", None),
                      "all_gather_output", None)
        if torch.is_tensor(out) and out.is_inference():
            hits.append(f"{label}: _all_gather_result.all_gather_output")
    return hits


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _worker(rank: int, ws: int, port: int, out) -> None:
    res: Dict[str, Any] = {"phases": []}
    log = res["phases"]

    def mark(msg: str) -> None:
        log.append(f"[{time.strftime('%H:%M:%S')}] rank{rank} {msg}")

    try:
        dist.init_process_group("gloo", rank=rank, world_size=ws,
                                init_method=f"tcp://127.0.0.1:{port}")
        from torch.distributed._composable.fsdp import fully_shard  # noqa: PLC0415
        from torch.distributed.device_mesh import init_device_mesh  # noqa: PLC0415

        from lingbotvla.auto_learning.hardness import HardnessScorer  # noqa: PLC0415
        from lingbotvla.auto_learning.real.backend import (  # noqa: PLC0415
            RealHardnessScorer,
        )
        from lingbotvla.utils.open_loop_validation import (  # noqa: PLC0415
            _broadcast_object,
            _eval_tensor_context,
            _fsdp_full_params_context,
            _multirank_eval_payload,
        )

        mesh = init_device_mesh("cpu", (ws,))
        torch.manual_seed(0)
        model = _Root()
        for b in model.blocks:
            fully_shard(b, mesh=mesh)
        fully_shard(model, mesh=mesh)
        mark("FSDP2 已建（**根模块一次前向都没跑**，= 真机 bootstrap 的时刻）")

        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
        x = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 10.0

        def eval_once(tag: str) -> Dict[str, Any]:
            """AL 的 `evaluate_ids()` 协议：窗口在外、payload（所有 rank 真跑）在内。

            ``EVAL_MODE=repo``（默认）走仓库真实的 `_eval_tensor_context()`；
            ``EVAL_MODE=inference`` 换成 ``torch.inference_mode()``（修复前的写法）⇒ 复现 bug；
            ``EVAL_MODE=nograd`` 显式 ``torch.no_grad()`` ⇒ 对照。
            """

            def _run() -> Dict[str, Any]:
                # nograd = 走**仓库真实的** `_eval_tensor_context()`（修复后的评测体上下文）；
                # inference = 逐字复现修复前的 `torch.inference_mode()`。
                if EVAL_MODE == "inference":
                    ctx = torch.inference_mode()
                elif EVAL_MODE == "nograd":
                    ctx = torch.no_grad()
                else:                       # repo：用仓库当前实现
                    ctx = _eval_tensor_context()
                with ctx:
                    y = model.sample_actions(x)
                return {"mse": float(y.sum()), "n_traj": 2, "split": tag}

            with _fsdp_full_params_context(model, logger=None):
                return _multirank_eval_payload(
                    run=_run, ws=ws, rank=rank, broadcast=_broadcast_object,
                    all_ranks_run=True)

        t0 = time.perf_counter()
        r1 = eval_once("train_monitor")
        mark(f"① 评测 #1（train_monitor）完成 {time.perf_counter() - t0:.2f}s mse={r1['mse']:.4f}")
        t0 = time.perf_counter()
        r2 = eval_once("val")
        mark(f"② 评测 #2（val）完成 {time.perf_counter() - t0:.2f}s mse={r2['mse']:.4f}")

        # ---- ③ Hardness 扫描：**真** RealHardnessScorer + **真** HardnessScorer ----
        #     8 条（5 + 3 两条批）⇒ 覆盖「非整除的最后一批」这条路径。
        items = _make_items(8)
        h = RealHardnessScorer(HardnessScorer(model), items, max_batch=5, logger=None)
        t0 = time.perf_counter()
        scored = h.score("place_dual_shoes", list(range(8)))
        mark(f"③ Hardness 扫描完成 {time.perf_counter() - t0:.2f}s "
             f"（{len(scored)} 条；含部分批）")

        # ---- ④ 第一个训练步：fwd + bwd + optimizer.step（真机训练循环的形状）----
        model.train()
        t0 = time.perf_counter()
        loss = model(actions=torch.randn(4, 4, 3), state=torch.randn(4, 8),
                     noise=torch.randn(4, 4, 3), time=torch.full((4,), 0.5))
        lv = loss["batch_mean_losses"].mean()
        lv.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        mark(f"④ 训练步 fwd+bwd+step 完成 {time.perf_counter() - t0:.2f}s loss={float(lv):.4f}")

        # ---- ⑤ 单元末评测（`_apply_train_result` 的 train_monitor/val）----
        eval_once("train_monitor")
        eval_once("val")
        mark("⑤ 单元末两次评测完成")

        # ---- ⑥ 记账：hook 的 `_drain_pending_unit` 在 ratio 模式下 all_gather_object ----
        gathered: List[Any] = [None] * ws
        dist.all_gather_object(gathered, (1, 8, {"place_dual_shoes": 8}, {}))
        res["accounting_identical"] = bool(all(g == gathered[0] for g in gathered))
        mark("⑥ all_gather_object 记账完成")

        res["ok"] = True
        res["scored_n"] = len(scored)
        res["eval_same_on_all_ranks"] = (abs(r1["mse"] - r2["mse"]) >= 0.0)
        dist.barrier()
    except Exception as exc:  # noqa: BLE001
        res["fatal"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-2000:]
        try:
            res["inference_tensors"] = _dump_inference_tensors(model)
        except Exception:  # noqa: BLE001
            res["inference_tensors"] = ["<dump failed>"]
    finally:
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass
        out[rank] = res


def _run_workers(ws: int, timeout: float) -> Dict[int, Dict[str, Any]]:
    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    out = mgr.dict()
    port = _port()
    procs = [ctx.Process(target=_worker, args=(r, ws, port, out), daemon=True)
             for r in range(ws)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout)
    hung = [i for i, p in enumerate(procs) if p.is_alive()]
    if hung:
        for p in procs:
            p.terminate()
        seen = {int(k): dict(v) for k, v in dict(out).items()}
        raise SystemExit(
            f"❌ 疑似**死锁**：rank {hung} 在 {timeout}s 内未结束。\n"
            f"   已见阶段：{ {r: v.get('phases') for r, v in seen.items()} }")
    return {int(k): dict(v) for k, v in dict(out).items()}


def main() -> int:
    global EVAL_MODE

    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--eval-mode", choices=("repo", "nograd", "inference"),
                    default=EVAL_MODE, help="评测体的张量上下文（默认取 $EVAL_MODE）")
    args = ap.parse_args()

    EVAL_MODE = args.eval_mode
    # 子进程是 spawn：把选择通过环境变量传下去（模块级 EVAL_MODE 在子进程重新读）。
    os.environ["EVAL_MODE"] = EVAL_MODE
    print(f"[repro] CPU gloo × {args.ws} 进程；超时 {args.timeout}s 判死锁；"
          f"EVAL_MODE={EVAL_MODE}")
    results = _run_workers(args.ws, args.timeout)
    bad = 0
    for r, res in sorted(results.items()):
        for line in res.get("phases", []):
            print("   ", line)
        if res.get("fatal"):
            bad += 1
            print(f"❌ rank{r} 异常: {res['fatal']}\n{res.get('traceback', '')}")
            if res.get("inference_tensors"):
                print(f"   🔴 带 inference 标记的 FSDP2 张量（{len(res['inference_tensors'])} 个）:")
                for line in res["inference_tensors"][:8]:
                    print(f"      - {line}")
    if bad:
        print("❌ 复现失败（见上面的 traceback）")
        return 1
    print("✅ 全链路无死锁：评测窗口 → Hardness 根前向 → 训练步 → 单元末评测 → 记账")
    print(f"   （EVAL_MODE={EVAL_MODE}：repo/nograd=当前实现；inference=修复前的写法）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
