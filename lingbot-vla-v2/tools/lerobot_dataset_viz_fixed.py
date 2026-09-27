import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import lerobot.scripts.lerobot_dataset_viz as viz


ROOT = Path(
    "/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30"
)

TASKS = [
    "adjust_bottle",
    "click_bell",
    "hanging_mug",
    "move_stapler_pad",
    "place_a2b_left",
    "place_can_basket",
    "place_fan",
    "place_phone_stand",
    "rotate_qrcode",
    "stack_blocks_two",
    "beat_block_hammer",
    "dump_bin_bigbin",
    "lift_pot",
    "open_laptop",
    "place_a2b_right",
    "place_cans_plasticbox",
    "place_mouse_pad",
    "place_shoe",
    "scan_object",
    "stack_bowls_three",
    "blocks_ranking_rgb",
    "grab_roller",
    "move_can_pot",
    "open_microwave",
    "place_bread_basket",
    "place_container_plate",
    "place_object_basket",
    "press_stapler",
    "shake_bottle",
    "stack_bowls_two",
    "blocks_ranking_size",
    "handover_block",
    "move_pillbottle_pad",
    "pick_diverse_bottles",
    "place_bread_skillet",
    "place_dual_shoes",
    "place_object_scale",
    "put_bottles_dustbin",
    "shake_bottle_horizontally",
    "stamp_seal",
    "click_alarmclock",
    "handover_mic",
    "move_playingcard_away",
    "pick_dual_bottles",
    "place_burger_fries",
    "place_empty_cup",
    "place_object_stand",
    "put_object_cabinet",
    "stack_blocks_three",
    "turn_switch",
]


def first_item(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()

    if isinstance(x, (list, tuple)):
        return x[0] if len(x) else ""

    if x is None:
        return ""

    return str(x)


def get_episode_arg():
    for i, arg in enumerate(sys.argv):
        if arg == "--episode-index" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])

        if arg.startswith("--episode-index="):
            return int(arg.split("=", 1)[1])

    return None


def print_episode_info(ep):
    files = sorted((ROOT / "meta" / "episodes").rglob("*.parquet"))

    meta = pd.concat(
        [pd.read_parquet(p) for p in files],
        ignore_index=True,
    )

    row = meta[meta["episode_index"] == ep]

    if len(row) == 0:
        print(f"WARNING: metadata for episode {ep} not found")
        return

    row = row.iloc[0]

    block = ep // 50
    round_id = ep % 50 + 1

    if 0 <= block < len(TASKS):
        task_name = TASKS[block]
    else:
        task_name = f"block_{block}"

    if "tasks" in row.index:
        instruction = first_item(row["tasks"])
    elif "task" in row.index:
        instruction = first_item(row["task"])
    else:
        instruction = "(no instruction found)"

    length = row["length"] if "length" in row.index else "?"

    print()
    print("=" * 88)
    print("ROBOTWIN EPISODE")
    print("=" * 88)
    print(f"Canonical task : {task_name}")
    print(f"Block          : {block}")
    print(f"Episode index  : {ep}")
    print(f"Task round     : {round_id} / 50")
    print(f"Frames         : {length}")
    print()
    print("Instruction:")
    print(instruction)
    print("=" * 88)
    print()


class FixedEpisodeSampler(torch.utils.data.Sampler):
    """
    Fix LeRobot 0.4.2 visualization bug for episode_index > 0.
    Dataset has already been filtered to one episode, so use local
    indices 0..len(dataset)-1 instead of global frame indices.
    """

    def __init__(self, dataset, episode_index):
        self.frame_ids = range(len(dataset))

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self):
        return len(self.frame_ids)


# ------------------------------------------------------------------
# Force PyAV only for this visualization helper.
# Do NOT modify installed LeRobot / torch / torchcodec.
# ------------------------------------------------------------------

OriginalLeRobotDataset = viz.LeRobotDataset


class FixedLeRobotDataset(OriginalLeRobotDataset):
    def __init__(self, *args, **kwargs):
        kwargs["video_backend"] = "pyav"
        super().__init__(*args, **kwargs)

        # Make the recording name easier to recognize in Rerun.
        episodes = kwargs.get("episodes")

        if episodes and len(episodes) == 1:
            ep = int(episodes[0])
            block = ep // 50
            round_id = ep % 50 + 1

            if 0 <= block < len(TASKS):
                task = TASKS[block]
                self.repo_id = (
                    f"{task}__ep{ep:04d}__round{round_id:02d}"
                )


viz.EpisodeSampler = FixedEpisodeSampler
viz.LeRobotDataset = FixedLeRobotDataset


if __name__ == "__main__":
    episode = get_episode_arg()

    if episode is not None:
        print_episode_info(episode)

    viz.main()
