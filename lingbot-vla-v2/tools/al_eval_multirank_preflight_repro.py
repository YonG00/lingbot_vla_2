#!/usr/bin/env python
"""CPU N 进程（gloo + **真 FSDP2** + **真 `OpenLoopValidator`**）复现/守门多卡开环评测的
「rank 不对称 ⇒ 集合通信错配 ⇒ NCCL 看门狗 600 s 后 abort」这一类故障。

真机现象（2026-10-10，AMD 8×W7900D / ROCm / 8 卡 FSDP2，正式 AL 训练）
--------------------------------------------------------------------
* `global ratio plan …` / `已就绪：50 任务 …` 正常；
* 约 12 分钟后在**评测**里整体崩掉，日志只有::

      [rank1] ProcessGroupNCCL.cpp:744 [Rank 1] Some NCCL operations have failed or timed out.
      [rank1] ProcessGroupNCCL.cpp:758 [Rank 1] To avoid data inconsistency, we are taking the entire process down.
      [rank1] [PG ID 0 PG GUID 0(default_pg) Rank 1] Process group watchdog thread terminated …
      [open_loop] ⚠️ [al_adjust_bottle_val_e10f709e] 评测失败: DistBackendError: NCCL communicator was aborted on rank 1.

* 关键判读：① 挂掉的集合通信在 **default_pg**（= 评测末尾的 `broadcast_object_list`，
  FSDP2 的 all-gather 走的是 device mesh 自己的 process group）；② `e10f709e` 是
  `adjust_bottle` 前 2 条 val 回合的指纹（= bootstrap 的**第一个 scout 评测**，
  `TASK_ORDER[0]`）⇒ 谁都没走到那次广播 ⇒ **有 rank 没到**。

本脚本把「有 rank 没到」的可复现成因在 CPU 上搭出来（判据：不许挂死）
----------------------------------------------------------------------
1. ``--inject <rank>:prepare`` —— 某 rank 在**准备阶段**（建评测数据集）失败。
   旧行为：该 rank 被 `_multirank_eval_payload` 静默吞掉后去等末尾广播，
   其余 rank 卡在 all-gather ⇒ 死锁（本脚本超时判失败）；
   新行为：一致性预检把「谁失败 + 为什么」在**进入集合通信之前**摊开，所有 rank 一起抛。
2. ``--inject <rank>:skew`` —— 某 rank 的评测回合集合/推理次数与别人不同
   （模拟「按 rank 各自抽样 / 索引口径分叉」）。旧行为同样死锁；新行为由预检
   直接点名 ``n_starts 不一致``。
3. ``--inject <rank>:delay`` —— 某 rank 在**窗口之前**慢一拍（`--delay-sec`，默认 8s；
   模拟真机 rank4：某张卡在做一件非集合通信的本地重活）。
   新顺序下这**不是死锁**：其余 rank 停在「一致性预检 A」的集合点上，
   逐 rank 阶段日志 + 停滞告警会把「谁在等、等在哪」写清楚（真机 rank4 当时一行日志都没有）。
4. ``--inject <rank>:warmup`` —— 某 rank 的**对称预热**（窗口之外的 1 次真实推理）
   报告本地失败。新顺序下由「一致性预检 B」在**真实 unshard 窗口之前**点名
   ⇒ 所有 rank 一起快速失败（预热窗口本身进出仍然对称）；旧顺序（预热在窗口内）
   这种失败会把全组拖到看门狗超时。
4b. ``--inject <rank>:warmup_skip`` —— 某 rank **不进**预热窗口（不对称）。
   预热窗口是集合通信，一个 rank 进、另一个不进必死锁 ⇒ 「一致性预检 A」的
   ``warmup`` 字段直接点名，所有 rank 在**任何 FSDP 集合通信之前**一起停。
5. ``--inject <rank>:forward`` —— 某 rank 在**推理中途**（已进入集合通信区）失败。
   预检救不了这种（它发生在集合通信里），但 `AL_EVAL_FAIL_FAST_NONZERO=1`
   会让该 rank **立刻抛**（不等广播）⇒ 日志里留下完整栈，而不是 600 s 后只剩
   `NCCL communicator was aborted`。

`--preflight off`（= `AL_EVAL_PREFLIGHT=0`）可逐字回到旧行为，用来对照「新代码真的
把死锁变成了报错」；`--warmup off`（= `AL_EVAL_WARMUP=0`）对照「预热关掉后逐字回旧流程」。

用法::

    PY=/opt/anaconda3/bin/python
    $PY tools/al_eval_multirank_preflight_repro.py --ws 8                 # 健康路径（应全绿）
    $PY tools/al_eval_multirank_preflight_repro.py --ws 8 --inject 3:prepare
    $PY tools/al_eval_multirank_preflight_repro.py --ws 8 --inject 3:prepare --preflight off
    $PY tools/al_eval_multirank_preflight_repro.py --ws 8 --inject 3:skew
    $PY tools/al_eval_multirank_preflight_repro.py --ws 8 --inject 3:warmup
    $PY tools/al_eval_multirank_preflight_repro.py --ws 8 --inject 3:warmup_skip
    $PY tools/al_eval_multirank_preflight_repro.py --ws 8 --inject 3:delay --delay-sec 8
    $PY tools/al_eval_multirank_preflight_repro.py --ws 8 --inject 3:forward --fail-fast on

退出码：0 = 全部 rank 正常结束（或**按预期**一起失败）；2 = 疑似死锁；3 = 断言不符。
本机（无 GPU）可直接跑：只要 torch（torchdata / datasets / lerobot 用桩替代）。
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
from typing import Any, Dict, List, Tuple


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# --------------------------------------------------------------------------- #
# 本机缺重依赖时用桩保证能 import 真实模块（只影响 import 期，被测代码一行不改）
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
        m.__path__ = []
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from torch import nn  # noqa: E402


# --------------------------------------------------------------------------- #
# 故障注入开关（spawn 子进程靠环境变量取值）
# --------------------------------------------------------------------------- #
def _inject() -> Dict[str, Any]:
    raw = os.environ.get("AL_REPRO_INJECT", "") or ""
    if not raw:
        return {}
    rank_s, _, phase = raw.partition(":")
    return {"rank": int(rank_s), "phase": phase or "prepare"}


#: 进程内的推理调用计数（`--inject <rank>:forward` 用）
_FWD_CALLS = {"n": 0}
#: 本进程是否正处在「窗口前对称预热」的那次推理里（预热也算一次 `sample_actions`）
_IN_WARMUP = {"on": False}
#: 本进程做过几次预热（应为 1 —— 一次/进程）
_WARMUPS = {"n": 0}


def _delay_sec() -> float:
    try:
        return max(0.0, float(os.environ.get("AL_REPRO_DELAY_SEC", "8") or 8))
    except (TypeError, ValueError):
        return 8.0


class _Cfg:
    """仿 `LingbotVLAV2Config` 里评测真正读到的几个字段。"""

    chunk_size = 2
    n_action_steps = 2
    max_action_dim = 3
    action_fp32 = False
    use_cache = False


class _Block(nn.Module):
    """嵌套 FSDP2 单元（真机 = 每层 decoder 一个）。"""

    def __init__(self, d: int = 8):
        super().__init__()
        self.lin1 = nn.Linear(d, d, bias=False)
        self.lin2 = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return torch.relu(self.lin2(torch.relu(self.lin1(x))))


class _Root(nn.Module):
    """形状仿 `LingbotVlaV2Policy`：根单元自有参数 + 嵌套单元 + **自定义方法**。

    `sample_actions()` 是自定义方法（绕过根模块 `__call__`）⇒ 与真机一样，
    FSDP2 的 all-gather 靠嵌套单元自己的前向 hook 触发。
    """

    def __init__(self, d: int = 8, n_blocks: int = 3):
        super().__init__()
        self.config = _Cfg()
        self.head = nn.Linear(d, d, bias=False)
        self.blocks = nn.ModuleList([_Block(d) for _ in range(n_blocks)])

    def forward(self, actions=None, state=None, **kw):
        x = state.float()
        for b in self.blocks:
            x = b(x)
        return {"batch_mean_losses": self.head(x).pow(2).mean(dim=1)}

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state,
                       noise=None, image_grid_thw=None):
        _FWD_CALLS["n"] += 1
        inj = _inject()
        # ⚠️ 预热（窗口外的第 1 次推理）**不算**在 `:forward` 注入的计数里：
        #    该注入要模拟的是「真实评测体中途失败」（已进集合通信区），
        #    即预热之后的那 1 次 ≠ 第 2 次。
        if (not _IN_WARMUP["on"] and inj.get("rank") == _RANK
                and inj.get("phase") == "forward" and _FWD_CALLS["n"] >= 3):
            raise RuntimeError(
                f"INJECTED forward failure on rank{_RANK} at inference #{_FWD_CALLS['n']}"
                "（模拟：推理中途 OOM / HIP 错误 / 数据解码失败）")
        h = self.head(state.float())
        for _ in range(2):                      # 模拟 denoise 步数
            for b in self.blocks:
                h = b(h)
        t, d = self.config.n_action_steps, self.config.max_action_dim
        v = h[:, : t * d].reshape(h.shape[0], t, d)
        return v if noise is None else v + noise


class _FakeLeRobot:
    def __init__(self, ep_map: List[int]):
        self.hf_dataset = {"episode_index": list(ep_map)}


class _FakeInner:
    def __init__(self, ep_map: List[int]):
        self.dataset = _FakeLeRobot(ep_map)


class _FakeFT:
    """仿 `FeatureTransform`：评测只用到 `unapply` / `org_features` / `image_augment`。"""

    image_augment = False
    org_features = {"actions": ["action.arm"]}

    def unapply(self, item: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(item)
        acts = item["actions"]
        arr = acts.detach().cpu().numpy() if hasattr(acts, "detach") else np.asarray(acts)
        out["action.arm"] = np.asarray(arr, dtype=np.float32)
        return out


class _FakeDS:
    """仿评测子集数据集：`len(ds)` 与 `_episode_index_map` 的 `episode_index` 等长。"""

    def __init__(self, ep_map: List[int], *, cam: int = 2, c: int = 3, hw: int = 8,
                 d_state: int = 8, chunk: int = 2, d_act: int = 3):
        self._ep_map = list(ep_map)
        self._datasets = [_FakeInner(self._ep_map)]
        self.strict_getitem = False
        self.cam, self.c, self.hw, self.d_state, self.chunk, self.d_act = cam, c, hw, d_state, chunk, d_act

    def __len__(self) -> int:
        return len(self._ep_map)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        g = torch.Generator().manual_seed(1000 + int(i))
        return {
            "images": torch.randn(self.cam, self.c, self.hw, self.hw, generator=g),
            "img_masks": torch.ones(self.cam, dtype=torch.bool),
            "lang_tokens": torch.randint(0, 10, (5,), generator=g),
            "lang_masks": torch.ones(5, dtype=torch.bool),
            "state": torch.randn(self.d_state, generator=g),
            "actions": torch.randn(self.chunk, self.d_act, generator=g),
            "action_is_pad": torch.zeros(self.chunk, dtype=torch.bool),
            "image_grid_thw": None,
        }


class _Logger:
    """极简 logger（评测代码只用 info_rank0 / info / warning）。"""

    def __init__(self, rank: int):
        self.rank = rank
        self.lines: List[str] = []

    def _emit(self, level: str, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}][{level}][rank{self.rank}] {msg}"
        self.lines.append(line)
        if level == "WARN":
            print(line, flush=True)

    def info_rank0(self, msg, *a, **k):
        if self.rank == 0:
            self._emit("INFO", str(msg))

    def info(self, msg, *a, **k):
        self._emit("INFO", str(msg))

    def warning(self, msg, *a, **k):
        self._emit("WARN", str(msg))


_RANK = 0


def _audit_collectives(store: List[Tuple[str, int]]) -> None:
    """给 `torch.distributed` 的集合通信函数挂计数器（逐 rank 记录序列）。"""
    def wrap(name: str, numel_idx: int = 0):
        orig = getattr(dist, name, None)
        if orig is None:
            return

        def _wrapped(*a, **k):
            try:
                n = int(a[numel_idx].numel())
            except Exception:  # noqa: BLE001
                n = -1
            store.append((name, n))
            return orig(*a, **k)

        setattr(dist, name, _wrapped)

    wrap("all_gather_into_tensor")
    wrap("broadcast_object_list", numel_idx=-1)
    wrap("all_gather_object", numel_idx=-1)


def _make_validator(model: nn.Module, out_dir: Path, logger: Any):
    """用**真** `OpenLoopValidator`（只把数据集/索引换成假件 + 注入故障）。"""
    from lingbotvla.utils.open_loop_validation import OpenLoopValidator

    args = types.SimpleNamespace(
        data=types.SimpleNamespace(image_augment=False, chunk_size=2, num_episode=None),
        model=types.SimpleNamespace(tokenizer_path="stub"),
        train=types.SimpleNamespace(
            output_dir=str(out_dir), global_rank=_RANK, data_parallel_mode="fsdp2",
            eval_inference_dtype="auto", use_bf16=False, micro_batch_size=2,
        ),
    )
    v = OpenLoopValidator(model=model, model_config=_Cfg(), args=args, processor=None,
                          use_depth_align=False, writer=None, logger=logger,
                          train_monitor_ids=[1, 2], val_ids=[1, 2, 3, 4], device="cpu")
    ft = _FakeFT()

    def _dataset(episode_ids_file: str):        # noqa: ANN202
        inj = _inject()
        if inj.get("rank") == _RANK and inj.get("phase") == "prepare":
            raise RuntimeError(f"INJECTED prepare failure on rank{_RANK}"
                               "（模拟：评测数据集构建/解码/磁盘失败）")
        if inj.get("rank") == _RANK and inj.get("phase") == "delay":
            # 窗口之前的本地重活（真机 rank4：非集合通信、分钟级、且旧代码里一行日志都没有）
            print(f"[repro][rank{_RANK}] INJECTED pre-window delay {_delay_sec():.1f}s "
                  "（模拟：某 rank 在窗口前做本地重活 —— 旧顺序下其余 rank 会卡在 unshard）",
                  flush=True)
            time.sleep(_delay_sec())
        ep_map = [0] * 5 + [1] * 5               # 2 个回合 × 5 帧
        if inj.get("rank") == _RANK and inj.get("phase") == "skew":
            ep_map = ep_map + [2] * 5            # 多一个回合 ⇒ 推理次数与别人不同
        ds = _FakeDS(ep_map)
        v._ft_by_path[episode_ids_file] = ft
        if v._ft is None:
            v._ft = ft
        return ds

    v._dataset = _dataset                       # type: ignore[assignment]
    _orig_warmup = v._eval_warmup
    _orig_plan = v._eval_warmup_plan

    def _plan(prep):                            # noqa: ANN202
        inj = _inject()
        if inj.get("rank") == _RANK and inj.get("phase") == "warmup_skip":
            # 模拟「某 rank 在窗口之前不走预热」这种**不对称**：预热窗口是集合通信，
            # 一个 rank 进、另一个不进就是死锁 ⇒ 预检 A 的 `warmup` 字段必须点名。
            print(f"[repro][rank{_RANK}] INJECTED warmup plan=False（不对称：本 rank 不预热）",
                  flush=True)
            return False
        return _orig_plan(prep)

    def _warmup(prep, tag, phases=None):        # noqa: ANN202
        inj = _inject()
        planned = bool(_orig_plan(prep))        # ⚠️ 只有「计划要预热」才算一次真实预热
        print(f"[repro][rank{_RANK}] 窗口前预热调用（计划={planned}）", flush=True)
        if not planned:
            return _orig_warmup(prep, tag, phases)
        if inj.get("rank") == _RANK and inj.get("phase") == "warmup":
            # 模拟「预热那 1 次推理在**本地**失败（集合通信之外被发现）」：
            # 真实预热照跑（窗口进出对称），然后报告错误 ⇒ 预检 B 点名 + 窗口外一起失败。
            _WARMUPS["n"] += 1
            _IN_WARMUP["on"] = True
            try:
                _orig_warmup(prep, tag, phases)
            finally:
                _IN_WARMUP["on"] = False
            print(f"[repro][rank{_RANK}] INJECTED warmup failure（预热已跑完，报告本地失败）",
                  flush=True)
            return f"INJECTED warmup failure on rank{_RANK}"
        _WARMUPS["n"] += 1
        _IN_WARMUP["on"] = True
        try:
            return _orig_warmup(prep, tag, phases)
        finally:
            _IN_WARMUP["on"] = False

    v._eval_warmup_plan = _plan                 # type: ignore[assignment]
    v._eval_warmup = _warmup                    # type: ignore[assignment]
    return v


# --------------------------------------------------------------------------- #
# 「无副作用」审计：预热 + 评测前后，RNG / 参数 / 梯度 / training 标志 / inference 张量
# --------------------------------------------------------------------------- #
def _snapshot_model(model: nn.Module) -> Dict[str, Any]:
    flags = [(n, bool(m.training)) for n, m in model.named_modules()]
    sums = {}
    for n, p in model.named_parameters():
        with torch.no_grad():
            t = p.detach()
            local = t.to_local() if hasattr(t, "to_local") else t
            sums[n] = float(local.float().sum())
    return {"flags": flags, "sums": sums, "rng": torch.get_rng_state().clone()}


def _audit_model(model: nn.Module, snap: Dict[str, Any]) -> List[str]:
    """返回「副作用」清单（空 = 干净）。判据对齐用户要求：不抽 RNG、不留梯度、不改参数、无 inference 张量。"""
    bad: List[str] = []
    now = _snapshot_model(model)
    if not torch.equal(now["rng"], snap["rng"]):
        bad.append("torch RNG 被推进（评测/预热必须恢复 RNG）")
    if now["flags"] != snap["flags"]:
        bad.append("模块 training 标志被改动")
    for k, v in snap["sums"].items():
        if abs(now["sums"].get(k, float("nan")) - v) > 1e-6:
            bad.append(f"参数被改动：{k}")
    for n, p in model.named_parameters():
        if p.grad is not None:
            bad.append(f"留下了梯度：{n}")
        for cand in (p, getattr(p, "to_local", lambda: p)()):
            try:
                if bool(cand.is_inference()):
                    bad.append(f"参数成了 inference 张量：{n}")
                    break
            except Exception:  # noqa: BLE001
                pass
    return bad


def _worker(rank: int, ws: int, port: int, out) -> None:
    global _RANK
    _RANK = rank
    res: Dict[str, Any] = {"phases": [], "audit": [], "results": {}}
    logger = _Logger(rank)

    def mark(msg: str) -> None:
        res["phases"].append(f"[{time.strftime('%H:%M:%S')}] rank{rank} {msg}")
        # 每走一步都推给父进程 ⇒ 死锁被强杀时也能看到「各 rank 最后停在哪」
        try:
            out[rank] = dict(res)
        except Exception:  # noqa: BLE001
            pass

    try:
        dist.init_process_group("gloo", rank=rank, world_size=ws,
                                init_method=f"tcp://127.0.0.1:{port}")
        from torch.distributed._composable.fsdp import fully_shard
        from torch.distributed.device_mesh import init_device_mesh

        _audit_collectives(res["audit"])
        mesh = init_device_mesh("cpu", (ws,))
        torch.manual_seed(0)
        model = _Root()
        for b in model.blocks:
            fully_shard(b, mesh=mesh)
        fully_shard(model, mesh=mesh)
        mark("FSDP2 已建（根模块一次前向都没跑 = 真机 bootstrap 时刻）")

        out_dir = Path(os.environ["AL_REPRO_OUT"])
        out_dir.mkdir(parents=True, exist_ok=True)
        validator = _make_validator(model, out_dir, logger)
        snap = _snapshot_model(model)

        # ---- ① 第一次 scout 评测（真机挂死的那一次：2 条回合）----
        t0 = time.perf_counter()
        r1 = validator.evaluate_ids([1, 5], "al_adjust_bottle_val_e10f709e")
        res["results"]["scout"] = {"mse": float(r1["mse"]), "n": int(r1["n"]),
                                   "n_chunks": int(r1["n_chunks"])}
        mark(f"① scout 评测完成 {time.perf_counter() - t0:.2f}s "
             f"mse={float(r1['mse']):.4f} chunks={int(r1['n_chunks'])}")
        # ---- ①b 无副作用审计（预热 + 评测跑完之后）----
        bad = _audit_model(model, snap)
        res["side_effects"] = bad
        res["warmups"] = int(_WARMUPS["n"])
        mark(("✅ 副作用审计 OK（RNG/参数/梯度/training 标志/inference 张量 全部干净；"
              f"本进程预热 {_WARMUPS['n']} 次）") if not bad
             else f"❌ 副作用审计发现 {len(bad)} 项：{bad}")

        # ---- ② confirm 评测（4 条回合，另一组 ids ⇒ 另一个 tag）----
        r2 = validator.evaluate_ids([1, 5, 20, 21], "al_adjust_bottle_val_9cdb4cbf")
        mark(f"② confirm 评测完成 mse={float(r2['mse']):.4f} chunks={int(r2['n_chunks'])}")

        # ---- ③ 训练步（单元级：评测后必须还能训 —— 上一轮修的就是这条）----
        model.train()
        loss = model(state=torch.randn(2, 8))["batch_mean_losses"].mean()
        loss.backward()
        mark(f"③ 训练步完成 loss={float(loss.detach()):.4f}")

        # ---- ④ 再评测一次（训练前向之后，根单元已 init）----
        r3 = validator.evaluate_ids([1, 5], "al_adjust_bottle_val_e10f709e")
        mark(f"④ 训练后再评测完成 mse={float(r3['mse']):.4f} chunks={int(r3['n_chunks'])}")

        # ---- ⑤ 集合通信计数逐 rank 对拍（对称性的硬证据）----
        marks = [None] * ws
        dist.all_gather_object(marks, len(res["audit"]))
        res["collective_counts"] = marks
        mark(f"⑤ 本 rank 集合通信调用数 = {len(res['audit'])}；全 rank = {marks}")

        # ---- ⑥ 逐 rank 阶段日志：本 rank 的**轨迹文件**里必须已经写下「进入评测」
        #          （且早于 unshard）—— 真机 rank4 那种「一行都没有」在这里就会暴露 ----
        trace = out_dir / "_open_loop_phase" / f"rank{rank}.log"
        txt = trace.read_text(encoding="utf-8") if trace.exists() else ""
        res["trace_lines"] = len(txt.splitlines())
        res["stall_lines"] = [ln for ln in txt.splitlines() if "仍停在" in ln]
        res["trace_seen"] = {
            "enter": "▶ 进入评测" in txt,
            "warmup": "预热" in txt,
            "preflight": "一致性预检" in txt,
            "unshard": "unshard（取全参数）完成" in txt,
            "enter_before_unshard": ("▶ 进入评测" in txt and "▶ unshard（取全参数）完成" in txt
                                     and txt.index("▶ 进入评测") < txt.index("▶ unshard（取全参数）完成")),
        }
        mark(f"⑥ 阶段日志轨迹文件 {trace.name}：{res['trace_lines']} 行；"
             f"{res['trace_seen']}")
        res["ok"] = True
    except Exception as exc:  # noqa: BLE001
        res["fatal"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-4000:]
        mark(f"❌ rank{rank} 结束于异常: {type(exc).__name__}")
    finally:
        res["logger_lines"] = logger.lines
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass
        out[rank] = res


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run_workers(ws: int, timeout: float, env: Dict[str, str]) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    out = mgr.dict()
    port = _port()
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    procs = [ctx.Process(target=_worker, args=(r, ws, port, out), daemon=True)
             for r in range(ws)]
    try:
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout)
        hung = [i for i, p in enumerate(procs) if p.is_alive()]
        if hung:
            for p in procs:
                p.terminate()
        res = {int(k): dict(v) for k, v in dict(out).items()}
    finally:
        # ⚠️ 不 shutdown 的话 Manager 的服务进程会把本进程挂在退出阶段
        #    （表现为工具「跑完了但不返回」）
        try:
            mgr.shutdown()
        except Exception:  # noqa: BLE001
            pass
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(5)
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return res, hung


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=90.0,
                    help="join 超时（秒）；超时即判「疑似死锁」")
    ap.add_argument("--inject", default="",
                    help="<rank>:<prepare|skew|delay|warmup|warmup_skip|forward>")
    ap.add_argument("--preflight", choices=("on", "off"), default="on",
                    help="off ⇒ AL_EVAL_PREFLIGHT=0（回到旧行为对照）")
    ap.add_argument("--warmup", choices=("on", "off"), default="on",
                    help="off ⇒ AL_EVAL_WARMUP=0（关掉窗口前对称预热，对照旧流程）")
    ap.add_argument("--fail-fast", choices=("on", "off"), default="off",
                    help="on ⇒ AL_EVAL_FAIL_FAST_NONZERO=1（非 0 rank 在集合通信区内失败即抛）")
    ap.add_argument("--delay-sec", type=float, default=8.0,
                    help="`--inject <rank>:delay` 在**窗口之前**停留的秒数")
    ap.add_argument("--stall-sec", type=float, default=20.0,
                    help="AL_EVAL_STALL_SEC（停滞告警阈值）")
    args = ap.parse_args()

    out_root = REPO / ".pytest_tmp" / f"al_eval_preflight_{os.getpid()}"
    env = {
        "AL_REPRO_OUT": str(out_root),
        "AL_REPRO_INJECT": args.inject,
        "AL_REPRO_DELAY_SEC": str(args.delay_sec),
        "AL_EVAL_PREFLIGHT": "1" if args.preflight == "on" else "0",
        "AL_EVAL_WARMUP": "1" if args.warmup == "on" else "0",
        "AL_EVAL_FAIL_FAST_NONZERO": "1" if args.fail_fast == "on" else "0",
        "AL_EVAL_STALL_SEC": str(args.stall_sec),
    }
    print(f"[repro] CPU gloo × {args.ws}；超时 {args.timeout}s 判死锁；"
          f"inject={args.inject or '-'} preflight={args.preflight} warmup={args.warmup} "
          f"fail_fast={args.fail_fast}")

    results, hung = _run_workers(args.ws, args.timeout, env)
    for line in sorted({ln for r in results.values() for ln in r.get("phases", [])}):
        print("   ", line)

    fatal = {r: v.get("fatal") for r, v in results.items() if v.get("fatal")}
    # `collective_counts` 是 all_gather 出来的「每个 rank 的集合通信次数」⇒ 取任意一份即可
    marks_by_rank: Dict[int, int] = {}
    for _r, _v in sorted(results.items()):
        cc = _v.get("collective_counts")
        if cc:
            marks_by_rank = {i: int(n) for i, n in enumerate(cc)}
            break
    inject_phase = (args.inject.split(":")[-1] if args.inject else "")
    injected_rank = (args.inject.split(":")[0] if args.inject else "")

    if fatal:
        print("❌ 有 rank 以异常结束：")
        for r, msg in sorted(fatal.items()):
            print(f"   rank{r}: {msg}")
            print("   " + "\n   ".join((results[r].get("traceback") or "").strip().splitlines()[-12:]))
        want_all = (inject_phase in ("prepare", "skew", "warmup", "warmup_skip")
                    and args.preflight == "on")
        if want_all and len(fatal) == args.ws:
            joined = " ".join(fatal.values())
            key = "一致性预检未通过" if "预检" in joined else "评测"
            print(f"✅ 符合预期：{args.ws} 个 rank **一起**失败（{key}）—— 没有死锁、"
                  "真因在日志里（见上面的 `[open_loop][multirank] ❌ …`）")
            return 0
        if inject_phase == "forward" and args.fail_fast == "on" and hung:
            print(f"✅ 符合预期（fail-fast）：rank{injected_rank} 在**集合通信区内**"
                  "失败 ⇒ 不等广播、**立刻抛出**（上面有完整栈）；其余 rank 仍会卡住，"
                  "但 torchrun 会立刻收掉整个作业 ⇒ 不需要等 600 s 看门狗。")
            return 0
        print("⚠️ 失败 rank 数 = "
              f"{len(fatal)}/{args.ws}（注入 {args.inject}，preflight={args.preflight}，"
              f"死锁 rank={hung}）")
        return 3

    if hung:
        print(f"❌ 疑似**死锁**：rank {hung} 在 {args.timeout}s 内未结束。")
        print("   已见阶段：")
        for r, v in sorted(results.items()):
            print(f"     rank{r}: {v.get('phases', [])[-1:]}")
        print("   （这正是真机 8 卡的失败形态：其余 rank 卡在 all-gather，"
              "失败/落后的 rank 等不到 —— 直到 NCCL 看门狗 600 s 后 abort）")
        return 2

    # ---- 逐 rank 阶段日志（真机 rank4「一行都没有」的守门）----
    seen = {r: (v.get("trace_seen") or {}) for r, v in sorted(results.items())}
    missing_enter = [r for r, s in seen.items() if not s.get("enter")]
    if missing_enter:
        print(f"❌ 这些 rank 连「▶ 进入评测」都没有落盘：{missing_enter} ⇒ "
              "『某 rank 无阶段日志』的老问题回来了")
        return 3
    bad_order = [r for r, s in seen.items() if not s.get("enter_before_unshard")]
    if bad_order:
        print(f"❌ 这些 rank 的「进入评测」不在 unshard 之前：{bad_order}")
        return 3
    print(f"✅ 逐 rank 阶段日志：全部 {args.ws} 个 rank 都有『▶ 进入评测』，"
          f"且都写在『unshard 窗口』之前（逐 rank 轨迹文件 rank*.log）")
    if args.warmup == "on":
        no_warm = [r for r, s in seen.items() if not s.get("warmup")]
        if no_warm:
            print(f"❌ 这些 rank 没有预热阶段日志：{no_warm}（预热必须**每个 rank** 都做）")
            return 3
        counts = {r: int(v.get("warmups") or 0) for r, v in results.items()}
        if set(counts.values()) != {1}:
            print(f"❌ 预热次数不对称（应各 rank 恰好 1 次/进程）：{counts}")
            return 3
        print(f"✅ 窗口前对称预热：各 rank 恰好 1 次（{counts}），"
              "且都在真实 unshard 窗口之前")
    # ---- 无副作用审计 ----
    side = {r: (v.get("side_effects") or []) for r, v in sorted(results.items())}
    if any(side.values()):
        print(f"❌ 副作用审计失败：{side}")
        return 3
    print(f"✅ 无副作用审计通过（{args.ws} 个 rank）：RNG 未被推进、参数未变、无梯度、"
          "training 标志未改、无 inference 张量")

    if marks_by_rank:
        print(f"   集合通信调用数（逐 rank）：{marks_by_rank}")
        if len(set(marks_by_rank.values())) != 1:
            print("❌ 各 rank 的集合通信调用数**不一致** ⇒ 这正是死锁的成因")
            return 3
    if inject_phase == "delay":
        # 延迟注入：不是死锁 —— 其余 rank 停在**窗口前**的集合点上，日志里能看到
        stall = [ln for v in results.values() for ln in (v.get("stall_lines") or [])]
        if not stall:
            print("⚠️ 没有捕获到停滞告警（把 --stall-sec 调到小于 --delay-sec 再试）")
        else:
            print(f"✅ 符合预期（窗口前延迟）：rank{injected_rank} 慢 {args.delay_sec}s，"
                  "其余 rank 停在**窗口前**的集合点（『一致性预检』），停滞告警已指名位置：")
            for ln in sorted(set(stall))[:4]:
                print("     " + ln.strip())
        print("    ⇒ 真机 rank4 那种『一行阶段日志都没有』不会再发生："
              "谁停在哪个阶段、停了多久，逐 rank 可见")
        return 0
    print("✅ 全链路无死锁：scout 评测 → confirm 评测 → 训练步 → 再评测；"
          "各 rank 集合通信次数一致、结果一致")
    same = {tuple(sorted((k, str(v.get("results", {}).get("scout", {}).get("n_chunks")))
                         for k, v in results.items()))}
    if len({len(v.get("results") or {}) for v in results.values()}) != 1:
        print("❌ 各 rank 拿到的评测结果条数不一致")
        return 3
    del same
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
