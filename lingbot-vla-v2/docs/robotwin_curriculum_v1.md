# RoboTwin Curriculum V1

## 1. Goal

This curriculum is designed for LingBot-VLA-v2 training on the RoboTwin clean
LeRobot dataset.

The curriculum separates two different notions of difficulty:

1. Skill complexity:
   L1 -> L2 -> L3 -> L4

2. Temporal execution length within the same task:
   Short -> Long

Trajectory length is not treated as a direct measure of task difficulty.

---

## 2. Dataset

Dataset:

`RoboTwin_lerobot_v30`

Root:

`/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30`

Statistics:

- 50 canonical tasks
- 50 demonstrations per task
- 2500 total episodes
- 548893 total frames
- 15 FPS
- 14-D observation.state
- 14-D action
- 3 camera streams

Each canonical task occupies one consecutive block of 50 episodes.

---

## 3. Dataset QA conclusion

Automated structural QA:

- hard integrity problems: 0
- frame count mismatches: 0
- non-contiguous frame indices: 0
- non-monotonic timestamps: 0
- state NaN/Inf episodes: 0
- action NaN/Inf episodes: 0
- unreadable MP4 files: 0
- ffprobe failures: 0

150 / 2500 episodes were marked as statistical anomaly candidates.

Manual inspection showed that sampled high-score candidates were still valid,
successful demonstrations.

Therefore Curriculum V1 performs:

**no episode deletion**

and:

**no temporal trimming**

All 2500 episodes are retained.

### open_microwave special note

open_microwave contains unusually long trajectories and large idle ratios.
Manual inspection showed that the door still opens successfully, but the
interaction is slow and contains long pauses.

These demonstrations are retained in V1.

A future curriculum version may compare:

- raw trajectory length
- active-motion trajectory length
- idle-segment compression

but V1 intentionally keeps the original demonstrations unchanged.

---

## 4. Short / Long definition

Short and Long are defined separately inside every canonical task.

For each task:

1. collect its 50 episodes
2. sort by `(trajectory_length, episode_index)`
3. ranks 1-25 -> Short
4. ranks 26-50 -> Long

The episode index is used as a deterministic tie-breaker.

Important:

Short != easy

Long != hard

Short / Long only describes relative demonstration duration inside the same
task.

---

## 5. Skill complexity levels

### L1 - Basic atomic skills

Single dominant manipulation primitive with limited sequential dependency.

14 tasks:

- click_bell
- click_alarmclock
- press_stapler
- turn_switch
- grab_roller
- lift_pot
- move_playingcard_away
- place_phone_stand
- place_object_stand
- place_mouse_pad
- move_pillbottle_pad
- move_stapler_pad
- move_can_pot
- place_shoe

### L2 - Constrained basic skills

Basic primitive plus orientation, spatial relation, tool use, dynamic motion,
or simple coordination.

15 tasks:

- adjust_bottle
- place_a2b_left
- place_a2b_right
- rotate_qrcode
- place_fan
- place_container_plate
- place_bread_skillet
- place_empty_cup
- place_object_scale
- beat_block_hammer
- stamp_seal
- pick_dual_bottles
- pick_diverse_bottles
- shake_bottle
- shake_bottle_horizontally

### L3 - Composed skills

Multiple primitives, articulated manipulation, multi-object interaction, or
two-step coordination.

11 tasks:

- dump_bin_bigbin
- open_laptop
- open_microwave
- handover_mic
- place_bread_basket
- place_burger_fries
- place_object_basket
- place_can_basket
- place_cans_plasticbox
- stack_blocks_two
- stack_bowls_two

### L4 - Complex composed skills

Strong multi-stage dependency, regrasp, handover followed by additional
manipulation, ordering, or three-object manipulation.

10 tasks:

- scan_object
- place_dual_shoes
- handover_block
- put_object_cabinet
- hanging_mug
- blocks_ranking_rgb
- blocks_ranking_size
- stack_bowls_three
- stack_blocks_three
- put_bottles_dustbin

---

## 6. Eight-stage cumulative curriculum

The curriculum is cumulative.

Previous data is never removed when entering the next stage.

| Stage | New data introduced | Cumulative episodes |
|---|---|---:|
| C1 | L1 Short | 350 |
| C2 | L1 Long | 700 |
| C3 | L2 Short | 1075 |
| C4 | L2 Long | 1450 |
| C5 | L3 Short | 1725 |
| C6 | L3 Long | 2000 |
| C7 | L4 Short | 2250 |
| C8 | L4 Long | 2500 |

Therefore:

C8 = complete original 2500-episode dataset.

No manually tuned replay weights are used.

As new data is introduced, the relative fraction of earlier curriculum data
naturally decreases.

---

## 7. Training interpretation

C1 first teaches short demonstrations of basic atomic skills.

C2 extends those same skills with longer demonstrations.

C3 introduces short demonstrations of more constrained skills while retaining
all L1 demonstrations.

This pattern continues until C8 contains the complete dataset.

The intended progression is therefore:

basic -> constrained -> composed -> complex composed

and inside each skill level:

short execution -> long execution

---

## 8. Important interpretation rules

Trajectory length and skill complexity must not be conflated.

For example:

- a semantically simple articulated task may have a very long trajectory
- a semantically complex coordination task may have a relatively short trajectory

The skill level is determined by manipulation structure.

Short / Long is determined only by the within-task trajectory ranking.

---

## 9. Evaluation

The evaluation protocol is not yet frozen in V1.

The following principles are fixed:

- checkpoint comparisons must use a fixed evaluation set
- use fixed seeds when possible
- evaluate previously learned skills as well as newly introduced skills
- Short / Long is not an evaluation category
- final evaluation runs canonical tasks normally
- clean and randomized results must be reported separately

The exact representative tasks and seeds will be frozen before training.

---

## 10. Future ablations

At minimum, compare:

1. Curriculum V1
2. Full-data mixed baseline

Possible later ablations:

3. active-motion Short / Long split
4. idle-segment compression
5. alternative L1-L4 task boundaries

The curriculum should be treated as an experimental hypothesis, not assumed to
be superior without comparison against the mixed baseline.
