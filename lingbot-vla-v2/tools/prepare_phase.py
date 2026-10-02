#!/usr/bin/env python
"""生成 L1-L4 阶段课程的数据文件 + sentinel 评测清单。

产出 (默认 /data/train/phases/):
    datasets.txt                            数据集清单, 四个阶段共用同一份
    phase1_L1.episode_ids.json              回合号白名单, 每次阶段不同
    phase2_L1_L2.episode_ids.json
    phase3_L1_L2_L3.episode_ids.json
    phase4_all.episode_ids.json
    phase1_eval.txt                         sentinel 评测清单 (累积 4/8/12/16 个任务)
    phase2_eval.txt
    phase3_eval.txt
    phase4_eval.txt
    README.md                               逐个文件说明用途

三件事互相正交:
    datasets.txt            回答「数据在哪」   -> --data.train_path
    phase*.episode_ids.json 回答「训练读哪些回合」-> --data.episode_ids_file
    phase*_eval.txt         回答「评测跑哪些任务」-> launcher 的 --task_list_file

sentinel 任务名**不在本脚本里硬编码**, 唯一真源是课程配置 yaml 的
`evaluation.sentinel` 段 (configs/curriculum/robotwin_curriculum_v1.yaml)。
改 sentinel = 改 yaml, 然后重跑本脚本。

用法:
    python tools/prepare_phase.py --phase all
    python tools/prepare_phase.py --phase 1 --gbs 112
    python tools/prepare_phase.py --phase 1 --gbs 112 --step-time 3.31
    python tools/prepare_phase.py --phase all --check-csv <旧manifest.csv>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robotwin_curriculum import (  # noqa: E402
    derive_episode_table,
    load_curriculum,
    load_phase_defs,
    load_sentinel,
    load_skill_levels,
    resolve_phase_episodes,
    resolve_phase_sentinel,
    summarize,
)

DEFAULT_CURRICULUM = "/data/code/lingbot-vla-v2/configs/curriculum/robotwin_curriculum_v1.yaml"
DEFAULT_OUT_DIR = "/data/train/phases"
DEFAULT_STEP_TIME = 3.31   # 实测: 4xRTX PRO 6000 96G, micro=28, accum=1, gbs=112


def check_lerobot_episodes_support():
    """预检: 当前环境的 LeRobot 是否支持 LeRobotDataset(episodes=...)。"""
    try:
        import inspect

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        params = inspect.signature(LeRobotDataset.__init__).parameters
        if "episodes" not in params:
            print("  [警告] 当前 LeRobot 的 LeRobotDataset 不含 `episodes` 参数,")
            print("         训练时将无法按回合筛选。请确认 lerobot 版本。")
            return False
        return True
    except Exception as exc:  # pragma: no cover
        print(f"  [提示] 跳过 LeRobot 版本预检: {type(exc).__name__}: {exc}")
        return False


def write_if_changed(path: Path, content: str) -> bool:
    """只在内容变化时写盘, 避免 datasets.txt 的 mtime 无谓变动。"""
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def format_eval_list(phase_key: str, levels, tasks: list[str]) -> str:
    """渲染 `phaseN_eval.txt` 的内容。

    头部 `#` 注释说明来源; 正文每行一个任务名, 顺序即评测顺序。
    launcher 的 --task_list_file 会跳过空行与 `#` 开头的行。
    """
    header = [
        f"# RoboTwin sentinel 评测清单 —— {phase_key} ({'+'.join(levels)})",
        "#",
        "# 由 tools/prepare_phase.py 生成, 请勿手工编辑。",
        "# 唯一真源: configs/curriculum/robotwin_curriculum_v1.yaml -> evaluation.sentinel",
        "#",
        "# 用法:",
        "#   bash experiment/robotwin/start_robotwin_infer_and_eval.sh \\",
        f"#       --task_list_file <本文件> --task_config demo_clean ...",
        "#",
        f"# 任务数: {len(tasks)}",
    ]
    return "\n".join(header + tasks) + "\n"


def main():
    ap = argparse.ArgumentParser(description="生成 L1-L4 阶段课程数据文件")
    ap.add_argument("--phase", default="all",
                    help="1|2|3|4|all (默认 all)")
    ap.add_argument("--curriculum", default=DEFAULT_CURRICULUM)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--gbs", type=int, default=None,
                    help="可选; 仅用于打印 steps/epoch 与耗时预估, 不影响数据")
    ap.add_argument("--step-time", type=float, default=DEFAULT_STEP_TIME,
                    help=f"可选; 单步秒数, 用于耗时预估 (默认 {DEFAULT_STEP_TIME}s, "
                         f"来自 4xRTX PRO 6000 micro=28 实测)")
    ap.add_argument("--check-csv", default=None,
                    help="可选; 旧 curriculum_manifest_v1.csv 路径, 用于交叉验证")
    args = ap.parse_args()

    cfg = load_curriculum(args.curriculum)
    task_to_level = load_skill_levels(cfg)
    phase_defs = load_phase_defs(cfg)
    sentinel = load_sentinel(cfg, task_to_level)

    dataset_root = cfg["dataset"]["root"]

    print("=" * 78)
    print("RoboTwin L1-L4 阶段数据生成")
    print("=" * 78)
    print(f"  课程配置 : {args.curriculum}")
    print(f"  数据集   : {dataset_root}")
    print(f"  输出目录 : {args.out_dir}")
    print()

    check_lerobot_episodes_support()

    table = derive_episode_table(dataset_root, task_to_level)
    print(f"  回合表构建成功: {len(table)} 回合, "
          f"{table['task'].nunique()} 任务, {int(table['length'].sum())} 帧")
    print()

    # ---- 决定要生成哪些阶段 ----
    if args.phase == "all":
        targets = list(phase_defs.keys())
    else:
        key = f"P{args.phase}"
        if key not in phase_defs:
            raise SystemExit(f"未知阶段 {args.phase} (可选: 1 2 3 4 all)")
        targets = [key]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 数据集清单 (四阶段共用) ----
    # 第一列是 data_name, 决定 configs/robot_configs/<data_name>.yaml,
    # 所以取自 dataset.data_name (robotwin), 不是数据集本身的名字。
    data_name = cfg["dataset"].get("data_name")
    if not data_name:
        raise SystemExit(
            "课程配置的 dataset 段缺少 `data_name` (manifest 第一列, 应为 robotwin)"
        )

    # 护栏: data_name 必须能对上 configs/robot_configs/<data_name>.yaml,
    # 否则训练时 FeatureTransform 会找不到 robot 配置。
    repo_root = Path(args.curriculum).resolve().parents[2]
    robot_cfg = repo_root / "configs" / "robot_configs" / f"{data_name}.yaml"
    if not robot_cfg.exists():
        raise SystemExit(
            f"manifest 第一列 `{data_name}` 对应的机器人配置不存在: {robot_cfg}\n"
            f"  请把 dataset.data_name 改成 configs/robot_configs/ 下已有的名字"
            f" (现有: {sorted(p.name for p in robot_cfg.parent.glob('*.yaml'))})"
        )
    print(f"  [护栏] 机器人配置存在: {robot_cfg.relative_to(repo_root)}")

    manifest_line = f"{data_name} {dataset_root}\n"
    manifest_path = out_dir / "datasets.txt"
    changed = write_if_changed(manifest_path, manifest_line)
    print(f"  [清单] {manifest_path}  {'已更新' if changed else '内容未变'}")
    print(f"         {manifest_line.strip()}")
    print()

    # ---- 逐阶段生成白名单 + sentinel 评测清单 ----
    rows = []
    eval_rows = []
    print(f"  {'阶段':<10} {'等级':<14} {'任务':>4} {'回合':>6} {'帧数':>9}"
          f" {'steps/epoch':>12} {'预计耗时':>10}")
    print("  " + "-" * 74)

    for key in targets:
        info = phase_defs[key]
        levels = info["levels"]
        dir_name = info.get("dir_name", key.lower())
        eval_name = info.get("eval_name")
        if not eval_name:
            raise SystemExit(
                f"阶段 {key} 缺少 `eval_name` (sentinel 评测清单文件名); "
                f"请在课程配置的 phases 段补上"
            )

        ep_ids = resolve_phase_episodes(table, levels)
        s = summarize(table, levels)

        ep_path = out_dir / f"{dir_name}.episode_ids.json"
        write_if_changed(ep_path, json.dumps(ep_ids))

        # ---- sentinel 评测清单 (任务名来自 yaml, 不硬编码) ----
        eval_tasks = resolve_phase_sentinel(sentinel, levels)
        eval_path = out_dir / eval_name
        write_if_changed(eval_path, format_eval_list(key, levels, eval_tasks))
        eval_rows.append((key, levels, eval_path, eval_tasks))

        steps_txt = "-"
        time_txt = "-"
        if args.gbs:
            steps = s["frames"] / args.gbs
            steps_txt = f"{round(steps)}"
            secs = steps * args.step_time
            time_txt = (f"{secs/60:.0f}分" if secs < 3600 else f"{secs/3600:.1f}时")

        rows.append((key, dir_name, levels, s, ep_path, steps_txt, time_txt))
        print(f"  {key:<10} {'+'.join(levels):<14} {s['tasks']:>4} {s['episodes']:>6}"
              f" {s['frames']:>9} {steps_txt:>12} {time_txt:>10}")

    print()

    # ---- sentinel 评测清单概览 ----
    print("  Sentinel 评测清单 (累积, 传给 launcher 的 --task_list_file):")
    print(f"    {'阶段':<8} {'等级':<14} {'任务数':>6}  {'文件':<20} 任务")
    print("    " + "-" * 76)
    for key, levels, eval_path, eval_tasks in eval_rows:
        print(f"    {key:<8} {'+'.join(levels):<14} {len(eval_tasks):>6}  "
              f"{eval_path.name:<20} {' '.join(eval_tasks)}")
    print()

    # ---- 累积关系自检 (P1 必须是 P2 的前缀, 依此类推) ----
    for prev, cur in zip(eval_rows, eval_rows[1:]):
        prev_tasks, cur_tasks = prev[3], cur[3]
        if cur_tasks[:len(prev_tasks)] != prev_tasks:
            raise SystemExit(
                f"sentinel 累积关系被破坏: {prev[0]} 的清单不是 {cur[0]} 的前缀\n"
                f"  {prev[0]}: {prev_tasks}\n  {cur[0]}: {cur_tasks}"
            )
    print("  [自检] sentinel 累积关系正确 (P1 ⊂ P2 ⊂ P3 ⊂ P4, 且为前缀关系)")
    print()

    # ---- 交叉验证 (防回归护栏) ----
    if args.check_csv:
        import pandas as pd

        old = pd.read_csv(args.check_csv)
        print("  交叉验证 (对比旧 8 阶段 manifest 的偶数累积阶 C2/C4/C6/C8):")
        for key in targets:
            n = int(key[1])              # "P1" -> 1
            info = phase_defs[key]
            mine = set(resolve_phase_episodes(table, info["levels"]))
            theirs = set(int(x) for x in old.loc[
                old["first_stage_num"] <= 2 * n, "episode_index"
            ])
            ok = mine == theirs
            print(f"    {key} vs C{2*n}: 本脚本 {len(mine)} 回合, "
                  f"旧清单 {len(theirs)} 回合  [{'一致' if ok else '不一致!'}]")
            if not ok:
                raise SystemExit(f"交叉验证失败: {key} 与 C{2*n} 的回合集合不一致")
        print()

    # ---- README ----
    readme_lines = [
        "# L1-L4 阶段课程数据文件",
        "",
        "由 `tools/prepare_phase.py` 生成, 请勿手工编辑。",
        "",
        "## 文件用途",
        "",
        "| 文件 | 回答的问题 | 传给谁 | 是否随阶段变化 |",
        "|---|---|---|---|",
        "| `datasets.txt` | 数据**在哪** | 训练 `--data.train_path` | 否 (四阶段同一份) |",
        "| `phase*.episode_ids.json` | 训练读**哪些回合** | 训练 `--data.episode_ids_file` | 是 |",
        "| `phase*_eval.txt` | 评测跑**哪些任务** | launcher `--task_list_file` | 是 (累积) |",
        "",
        "`datasets.txt` 每行格式为 `名称 路径`; 白名单是 JSON 整数数组 (升序、去重)。",
        "",
        "`phase*_eval.txt` 每行一个任务名, `#` 开头为注释 (launcher 会跳过)。",
        "任务名不硬编码, 唯一真源是课程配置 yaml 的 `evaluation.sentinel` 段。",
        "",
        "## 各阶段",
        "",
        "| 阶段 | 等级 | 训练任务数 | 回合数 | 帧数 | 白名单文件 | sentinel 数 | 评测清单 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(phase_defs.keys()):
        info = phase_defs[key]
        s = summarize(table, info["levels"])
        ev = resolve_phase_sentinel(sentinel, info["levels"])
        readme_lines.append(
            f"| {key} | {'+'.join(info['levels'])} | {s['tasks']} | {s['episodes']} "
            f"| {s['frames']} | `{info.get('dir_name', key.lower())}.episode_ids.json` "
            f"| {len(ev)} | `{info.get('eval_name', '-')}` |"
        )
    readme_lines += [
        "",
        "## 启动示例",
        "",
        "```bash",
        "cd /data/code/lingbot-vla-v2",
        "export PATH=/data/miniconda3/envs/lingbotvla/bin:$PATH",
        "",
        "# 训练",
        "bash train.sh tasks/vla/train_lingbotvla.py \\",
        "  /data/train/configs/robotwin_official_paths.yaml \\",
        f"  --data.train_path        {out_dir}/datasets.txt \\",
        f"  --data.episode_ids_file  {out_dir}/phase1_L1.episode_ids.json \\",
        "  --train.output_dir       /data/outputs/phase1_L1 \\",
        "  --train.micro_batch_size 28 \\",
        "  --train.gradient_accumulation_steps 1 \\",
        "  --train.global_batch_size 112 \\",
        "  --train.train_expert_only true \\",
        "  --data.image_augment true",
        "",
        "# 评测 (Clean)",
        "bash experiment/robotwin/start_robotwin_infer_and_eval.sh \\",
        f"  --task_list_file {out_dir}/phase1_eval.txt \\",
        "  --task_config demo_clean \\",
        "  --model_path <hf_ckpt> --num_gpus 4 --num_per_gpu 1 \\",
        "  --eval_workdir /data/code/RoboTwin-lingbot \\",
        "  --output_base /data/eval_results/run1",
        "```",
        "",
        "切阶段只需改 `--data.episode_ids_file`、`--train.output_dir`、`--task_list_file` 三处。",
        "",
    ]
    write_if_changed(out_dir / "README.md", "\n".join(readme_lines))
    print(f"  [说明] {out_dir}/README.md")

    print()
    print("=" * 78)
    print("完成")
    print("=" * 78)


if __name__ == "__main__":
    main()
