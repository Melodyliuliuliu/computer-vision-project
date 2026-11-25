# Goal Force × CausalCascade (RQ2 subset, forward / direct-force mode)

Goal Force inputs for the 21 matched factual/counterfactual pairs of four
CausalCascade scenes. `goal-force/` and `CausalCascade/` are not modified.

| Scene | delete | move | scale | mass | pairs |
|---|---:|---:|---:|---:|---:|
| `Domino_Series/l_turn_domino_obstacle_course` | 1 | 1 | 2 | 2 | 6 |
| `Domino_Series/straight_domino_line` | 1 | 1 | 0 | 4 | 6 |
| `Instability_Series/ball_drop_stack_collapse` | 1 | 1 | 0 | 2 | 4 |
| `Instability_Series/simple_platform_collapse` | 1 | 0 | 2 | 2 | 5 |

35 generations: 4 shared factual rows (delete/move/scale), 10 factual rows for the
mass pairs (they carry a mass blob), and 21 counterfactual rows.

## Files

- `inputs/*.csv`: one Goal Force CSV per generation (same format as
  `goal-force/datasets/examples`), with images in `inputs/images/`
- `inputs/manifest.csv`: pair to CSV mapping, reference videos, force settings,
  real and encoded masses
- `inputs/all_csvs.txt`: CSV paths relative to `goal-force/`
- `inputs/previews/`: each input drawn on its frame (red = force, blue/green = mass)
- `make_goalforce_csvs.py`: regenerates `inputs/` (needs `bpy`, `pandas`, `pyarrow`)
- `preview_inputs.py`: regenerates `inputs/previews/`
- `run_inference.sh`: runs all 35 CSVs

## Run

```bash
cd goal-force
bash ../experiments/goalforce_causalcascade/run_inference.sh
```

Outputs go to `goal-force/checkpoints/goal_force/step-3000-videos/`, named after the
image (for example `step-3000_straight_domino_line__mass__edit_009_obj1__cf__...mp4`).

## Evaluate (CausalCascade RQ2 Effect / Preserve IoU)

```bash
pip install opencv-python numpy pandas
python experiments/goalforce_causalcascade/eval_cf.py \
  --gen-dir /path/to/step-3000-videos \
  --out-dir experiments/goalforce_causalcascade/results
```

The script writes `results/per_pair.csv`, `results/summary_scene_macro.csv` and
`results/missing.json`. The metric code is CausalCascade's own
`metrics/counterfactual/cf_effect_metrics.py`, copied unmodified into `third_party/`
(MIT license). Their driver script expects all 225 pairs, so `eval_cf.py` does the
same pairing and the same scene-macro aggregation for our 21 pairs. A sanity check
that feeds the reference videos in as "generated" gives 1.000 on every metric.

## Object-level timing (supplement to the official metrics)

The official CF metrics pool motion over the whole clip, so they miss whether the
cascade reaches each object and in what order. `object_timing.py` takes the 2D box of
every dynamic object (`inputs/object_boxes.json`, projected from the `.blend` with the
edit applied) and measures, in all four videos of a pair, whether the object moves and
when it starts.

```bash
python experiments/goalforce_causalcascade/object_timing.py \
  --gen-dir /path/to/step-3000-videos \
  --out-dir experiments/goalforce_causalcascade/results/timing
```

Outputs:
- `per_object.csv`: moved / onset time for each object in each video
- `per_pair_timing.csv`, with these columns:
  - `factual_moved_acc`, `cf_moved_acc`: share of event-graph objects whose
    moved/not-moved status matches the reference
  - `flip_recall`: for objects whose status changes between the reference factual
    and counterfactual (the causal effect of the edit), whether the generated pair
    changes them the same way
  - `false_flips`: objects the generated pair changes but the reference does not
  - `factual_order_acc`, `cf_order_acc`: pairwise agreement of start order
- one plot of the motion curves per pair

Only event-graph participants count toward pair metrics by default. Static bystanders
(vases, cones) show motion whenever something falls in front of them. A box cannot
detect interpenetration, so that still needs eyeballing. Feeding the reference videos
in as "generated" gives 1.00 on every pair. `make_object_boxes.py` (needs `bpy`)
regenerates the boxes.

## How inputs were built

- **Coordinates.** Each `.blend` is opened with `bpy`, the intervention from `diff.json`
  is applied (delete / set location / set scale), and object bounding-box centres are
  projected through the scene camera (1920×1080 render, 50 mm lens) into Goal Force's
  832×480 space. Goal Force stretches the 854×480 frames to 832×480, so the CSV uses
  `width=832, height=480`, with `coordy` measured from the bottom. All previews checked.
- **Force.** A direct force is applied to the event-graph trigger (event `e01`), pointing
  toward the next participant (event `e02`). The angle is computed on the factual scene
  and kept the same for the counterfactual. Magnitude is 30, the weakest force used in
  training, meant as a nudge. `target_indirect_force = -1`.
- **Mass.** Only the mass pairs use the mass channel, with the blob on the edited object.
  CausalCascade's ratios (0.015× to 50×) do not fit Goal Force's [1, 4] range. Only the
  direction is encoded: heavier gives factual 1 → counterfactual 4, lighter gives 4 → 1.
- **Prompts.** These are the official matched Level 1 prompts from
  `tables/rq2_pairs.parquet`.

## Caveats

- None of these scenes has an external push. They start from a leaning domino, a
  free-falling ball, or an unbalanced plank, so the direct force here is a stand-in
  trigger. Goal Force cannot run with no force at all: both force fields set to -1
  breaks its control-signal code.
- Reference videos are 854×480 at 24 fps, 100–120 frames. Goal Force outputs 832×480
  at 15 fps, 81 frames. The counterfactual metrics resize everything to 256×256 and
  pool over the whole clip, so no alignment is needed. What can still differ is how
  much physical time each clip covers.
