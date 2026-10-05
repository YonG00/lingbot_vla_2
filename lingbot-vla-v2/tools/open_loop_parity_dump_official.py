#!/usr/bin/env python3
"""官方 `scripts/open_loop_eval.py` 的**逐值 dump 版** —— 供与 in-process 评测对拍。

做法：把官方脚本当**模块**加载（`__name__ != "__main__"` ⇒ 不跑 `main()`），
然后 patch 三个点：
  1. `plot_trajectory_results` —— 它恰好收到 gt/pred/state 三个数组，正是要对拍的量
  2. `prepare_eval_observation` —— dump 归一化**前**的原始 state + 帧号（定位"模型输入不同"的来源）
  3. `sample_actions_batch`  —— dump 真正喂给模型的输入 dict
另外把 `torch.randn` 包一层，只拦截 shape `(1, horizon, max_action_dim)`，
改用与 in-process 侧**同一个 seed 的 CUDA generator** ⇒ 噪声逐位相同。

🔴 **顺序必须与 in-process 侧完全一致**：先 5 条 train-monitor、再 10 条 val。
   噪声按"第几次 randn"消费 ⇒ 顺序不同 ⇒ 噪声错位 ⇒ pred 不可比。
   **不要**改成"每条轨迹重置 generator" —— 那与 in-process「整个集合连续消费」的语义不符。

用法：
    python tools/open_loop_parity_dump_official.py --ckpt <hf_ckpt> [--dump-dir /data/tmp/dump_official]

配套：`tools/open_loop_parity_compare.py`（逐值对比）、
      `tools/open_loop_eval_inprocess.py --dump-dir`（我方 dump）
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATA = "/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30"
# 与 lingbotvla/utils/open_loop_validation.py 的 DEFAULT_* 保持一致，且**顺序**要一致
TRAIN_MONITOR_IDS = [50, 63, 76, 87, 99]
VAL_IDS = [51, 52, 56, 66, 73, 75, 78, 84, 94, 97]
EVAL_SEED = 1234

_orig_randn = torch.randn
_gen = {"g": None, "n": 0}
NOISE_SHAPE: tuple = ()


def patched_randn(*args, **kwargs):
    if len(args) == 1 and not isinstance(args[0], int):
        raw = args[0]
    elif args:
        raw = args
    else:
        raw = kwargs.get("size")
    try:
        shape = tuple(raw) if not isinstance(raw, int) else (raw,)
    except TypeError:
        shape = None
    if shape == NOISE_SHAPE and "generator" not in kwargs:
        if _gen["g"] is None:
            g = torch.Generator(device="cuda")
            g.manual_seed(EVAL_SEED)
            _gen["g"] = g
        _gen["n"] += 1
        kw = dict(kwargs)
        kw["generator"] = _gen["g"]
        out = _orig_randn(*args, **kw)
        print(f"[dump] fixed-noise draw#{_gen['n']} -> {out.flatten()[:3].tolist()}", flush=True)
        return out
    return _orig_randn(*args, **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="hf_ckpt 目录")
    ap.add_argument("--dump-dir", default="/data/tmp/dump_official")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--plots", default="/data/eval_results/open_loop/parity_dump_official")
    ap.add_argument("--horizon", type=int, default=50, help="= chunk_size / use_length")
    ap.add_argument("--max-action-dim", type=int, default=55)
    ap.add_argument("--max-infer-time", type=int, default=10)
    a = ap.parse_args()

    global NOISE_SHAPE
    NOISE_SHAPE = (1, a.horizon, a.max_action_dim)
    dump = Path(a.dump_dir)
    dump.mkdir(parents=True, exist_ok=True)

    torch.randn = patched_randn

    # ---- 把官方脚本当模块加载 ------------------------------------------------
    ole_path = REPO / "scripts/open_loop_eval.py"
    spec = importlib.util.spec_from_file_location("ole", str(ole_path))
    ole = importlib.util.module_from_spec(spec)
    sys.modules["ole"] = ole
    spec.loader.exec_module(ole)

    _orig_plot = ole.plot_trajectory_results

    def dump_plot(state_joints_across_time, gt_action_across_time,
                  pred_action_across_time, traj_id, action_keys,
                  action_horizon, save_plot_path):
        np.save(dump / f"traj{traj_id}_gt.npy", gt_action_across_time)
        np.save(dump / f"traj{traj_id}_pred.npy", pred_action_across_time)
        np.save(dump / f"traj{traj_id}_state.npy", state_joints_across_time)
        print(f"[dump] traj{traj_id}: gt={gt_action_across_time.shape} "
              f"pred={pred_action_across_time.shape}", flush=True)
        return _orig_plot(
            state_joints_across_time=state_joints_across_time,
            gt_action_across_time=gt_action_across_time,
            pred_action_across_time=pred_action_across_time,
            traj_id=traj_id, action_keys=action_keys,
            action_horizon=action_horizon, save_plot_path=save_plot_path)

    ole.plot_trajectory_results = dump_plot

    from deploy.lingbot_vla_v2_policy import LingBotVlaV2InferencePolicy  # noqa: E402

    _cur = {"traj": None, "chunk_i": 0}
    _orig_est = ole.evaluate_single_trajectory

    def patched_est(policy, dataset, traj_id, *args, **kwargs):
        _cur["traj"] = traj_id
        _cur["chunk_i"] = 0
        return _orig_est(policy, dataset, traj_id, *args, **kwargs)

    ole.evaluate_single_trajectory = patched_est

    _orig_peo = ole.prepare_eval_observation

    def patched_peo(policy, traj):
        idx = _cur.get("chunk_i", 0) * a.horizon
        t = _cur["traj"]
        try:
            st = traj.get("observation.state")
            if st is not None:
                arr = st.detach().float().cpu().numpy() if hasattr(st, "detach") else np.asarray(st)
                np.save(dump / f"traj{t}_chunk{idx}_rawstate.npy", arr)
            for k in ("index", "episode_index", "frame_index", "timestamp"):
                if k in traj:
                    v = traj[k]
                    print(f"[dump] traj{t} chunk{idx} raw {k}="
                          f"{v.item() if hasattr(v, 'item') else v}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[dump] raw dump 失败: {exc}", flush=True)
        return _orig_peo(policy, traj)

    ole.prepare_eval_observation = patched_peo

    _orig_sab = LingBotVlaV2InferencePolicy.sample_actions_batch

    def patched_sab(self, observation, *args, **kwargs):
        idx = _cur.get("chunk_i", 0) * a.horizon
        _cur["chunk_i"] = _cur.get("chunk_i", 0) + 1
        p = dump / f"traj{_cur['traj']}_chunk{idx}"
        for key in ("images", "img_masks", "lang_tokens", "lang_masks",
                    "state", "image_grid_thw"):
            v = observation.get(key)
            if v is None:
                continue
            arr = v.detach().float().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            np.save(f"{p}_in_{key}.npy", arr)
        print(f"[dump] traj{_cur['traj']} chunk idx={idx} 输入已存", flush=True)
        return _orig_sab(self, observation, *args, **kwargs)

    LingBotVlaV2InferencePolicy.sample_actions_batch = patched_sab

    # ---- 复刻官方 __main__ 的流程 -------------------------------------------
    Path(a.plots).mkdir(parents=True, exist_ok=True)
    PolicyServer = ole.load_policy_server("auto", a.ckpt)
    model = PolicyServer(path_to_pi_model=a.ckpt, robot_norm_path=None,
                         use_length=a.horizon, use_bf16=False, use_fp32=True,
                         chunk_ret=True, use_compile=False)
    model.reset("robotwin")
    trajs = TRAIN_MONITOR_IDS + VAL_IDS      # 🔴 顺序不能改，见文件头说明
    print(f"[dump] 轨迹顺序 = {trajs}", flush=True)
    ole.main(model, "robotwin", a.data, trajs, a.horizon, a.plots, a.max_infer_time)
    print(f"[dump] 完成，输出目录 {dump}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
