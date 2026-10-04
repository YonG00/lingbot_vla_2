#!/usr/bin/env python3
"""**离线（解耦）版** in-process open-loop 评测。

从**已存在的 ckpt** 建模型（复用 deploy 的建模型流程：config 读 `lingbotvla_cli.yaml`、
强制 `attention_implementation='eager'` + `use_cache=True`），然后用与训练中评测
**完全相同**的代码路径（`build_vla_dataset` + `OpenLoopValidator._run`）算指标。

为什么需要它
------------
训练中的评测（`OpenLoopValidator`）**被绑在训练进程上**：它要拿训练里的 model / args /
data config，所以想评一份权重就必须「起训练 → 训到那一步 → 存一份 ckpt（或就地评）」。
⇒ 和官方 `scripts/open_loop_eval.py` 对拍时，每次都要白存 72G、多花约 7 分钟。

解耦之后：
* 不需要训练进程、不需要新存档 ⇒ 对拍两边**读同一个 ckpt**（"权重是否相同"这个问题消失）
* 任何已有 ckpt 都能直接评（旧的 779/1558、官方 50k 都行）
* 能把「训练中评测」与「离线评测」的差异单独隔离出来

用法
----
    cd /data/code/lingbot-vla-v2
    export QWEN3VL_PATH=/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct
    /data/miniconda3/envs/lingbotvla/bin/python -u tools/open_loop_eval_inprocess.py \
        --ckpt /data/outputs/smoke/parity2/checkpoints/global_step_2/hf_ckpt \
        --train-ids /data/outputs/smoke/ids/train50.json \
        --val-ids   /data/outputs/smoke/ids/val51.json \
        --dump-dir /data/tmp/dump_inprocess          # 可选：存 GT/pred .npy 供逐值对拍
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import yaml  # noqa: E402


class _Logger:
    """最小 logger：validator 只用 info_rank0 / info / warning。"""

    def info_rank0(self, msg, *a, **k):
        print(f"[eval] {msg}", flush=True)

    def info(self, msg, *a, **k):
        print(f"[eval] {msg}", flush=True)

    def warning(self, msg, *a, **k):
        print(f"[eval][WARN] {msg}", flush=True)


def _pick(d, *keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="hf_ckpt 目录（或 global_step_N 目录）")
    ap.add_argument("--train-ids", default=None, help="train-monitor 回合白名单 json")
    ap.add_argument("--val-ids", default=None, help="held-out val 回合白名单 json")
    ap.add_argument("--use-length", type=int, default=50)
    ap.add_argument("--use-bf16", action="store_true")
    ap.add_argument("--use-compile", action="store_true")
    ap.add_argument("--out-dir", default="/data/tmp/open_loop_inprocess")
    ap.add_argument("--dump-dir", default=None)
    a = ap.parse_args()

    ckpt = Path(a.ckpt).resolve()
    if not (ckpt / "config.json").exists():
        # 允许传 global_step_N，自动下钻到 hf_ckpt
        if (ckpt / "hf_ckpt" / "config.json").exists():
            ckpt = ckpt / "hf_ckpt"
        else:
            raise SystemExit(f"❌ {ckpt} 里没有 config.json（不是 hf_ckpt 目录）")

    cli = ckpt.parent.parent.parent / "lingbotvla_cli.yaml"
    if not cli.exists():
        raise SystemExit(f"❌ 找不到 {cli}（deploy 就是从这里读训练配置的）")
    raw = yaml.safe_load(cli.open())
    print(f"[eval] ckpt = {ckpt}\n[eval] cli  = {cli}", flush=True)

    # ---- 建模型：完全复用 deploy 的流程（eager + use_cache=True 都在里面）----
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

    server = LingbotVLAv2Server(
        path_to_pi_model=str(ckpt),
        robot_norm_path=None,
        use_length=a.use_length,
        chunk_ret=True,
        use_bf16=a.use_bf16,
        use_fp32=not a.use_bf16,
        use_compile=a.use_compile,
    )
    vla, processor = server.vla, server.processor
    cfg = vla.config
    print(f"[eval] config: attention_implementation={cfg.attention_implementation!r} "
          f"use_cache={cfg.use_cache} chunk_size={cfg.chunk_size} "
          f"n_action_steps={cfg.n_action_steps} max_action_dim={cfg.max_action_dim}", flush=True)

    # ---- 数据配置（用训练时写下来的 data 段）----
    data_cfg = dict(raw.get("data", {}))
    ns = types.SimpleNamespace(**data_cfg)
    if not hasattr(ns, "chunk_size"):
        ns.chunk_size = int(cfg.chunk_size)
    if not hasattr(ns, "num_episode"):
        ns.num_episode = None

    # 训练脚本里：`use_depth_align = True if args.train.align_params != {} else False`
    align_params = raw.get("train", {}).get("align_params") or {}
    uda = bool(align_params)
    print(f"[eval] use_depth_align={uda}（align_params 非空）"
          f" image_augment={getattr(ns, 'image_augment', None)}"
          f" use_future_image={getattr(ns, 'use_future_image', None)}", flush=True)

    class _NS:
        pass

    args = _NS()
    args.data = ns
    args.model = cfg
    args.train = _NS()
    args.train.output_dir = a.out_dir
    args.train.global_rank = 0
    args.train.use_bf16 = a.use_bf16

    from lingbotvla.utils.open_loop_validation import OpenLoopValidator, _load_episode_ids

    tr_ids = _load_episode_ids(a.train_ids) if a.train_ids else None
    va_ids = _load_episode_ids(a.val_ids) if a.val_ids else None

    validator = OpenLoopValidator(
        model=vla,
        args=args,
        processor=processor,
        use_depth_align=uda,
        writer=None,
        logger=_Logger(),
        train_monitor_ids=tr_ids,
        val_ids=va_ids,
        dump_dir=a.dump_dir,
    )

    # ---- 跑（与 validate() 里同样的准备，但不涉及训练状态）----
    vla.eval()
    torch.cuda.empty_cache()
    res = validator._run(0)

    print()
    print("=" * 78)
    for tag in ("train", "val"):
        r = res[tag]
        print(f"[{tag}] n={r['n']} chunk={r['n_chunks']} frames={r['frames']} dims={r['dims']} "
              f"mse={r['mse']:.6f} mae={r['mae']:.6f} "
              f"baseline={r['mean_baseline_mse']:.6f} r2={r['r2']:+.3f}")
        for i, m, f in zip(r["per_traj_ids"], r["per_traj_mse"], r["per_traj_frames"]):
            print(f"      ep{i}: mse={m:.6f} ({f} 帧)")
    print("=" * 78)
    if a.dump_dir:
        print(f"[eval] GT/pred 已 dump 到 {a.dump_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
