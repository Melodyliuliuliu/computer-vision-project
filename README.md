# computer-vision-project

Counterfactual tests of whether Goal Force (action-conditioned video model) has learned
physical laws, on a 4-scene subset of the CausalCascade benchmark.

## Structure

```
computer-vision-project/
├── goal-force/                     # Goal Force code, unmodified copy
├── CausalCascade/                  # dataset subset, unmodified (Git LFS)
│   ├── annotations/                #   scene metadata, event graphs, interventions (diff.json)
│   ├── assets/                     #   first frames + reference videos (base / counterfactual)
│   ├── prompts/                    #   level1-3 prompts, rq2_matched prompts
│   ├── source_scenes/blender/      #   .blend scenes
│   ├── tables/                     #   parquet tables (rq2_pairs, ...)
│   └── causalcascade/, docs/       #   loader + official docs
└── experiments/goalforce_causalcascade/
    ├── inputs/                     # Goal Force CSVs, force 30 (default)
    │   ├── *.csv, images/          #   35 generations = 21 factual/counterfactual pairs
    │   ├── manifest.csv            #   pair -> CSVs -> reference videos
    │   ├── all_csvs.txt            #   run list
    │   ├── object_boxes.json       #   2D boxes of dynamic objects (for timing)
    │   └── previews/               #   force / mass overlays
    ├── inputs_f100/, inputs_f200/  # same samples, force 100 / 200
    ├── make_goalforce_csvs.py      # build inputs (needs bpy)
    ├── make_object_boxes.py        # build object_boxes.json (needs bpy)
    ├── preview_inputs.py           # draw previews
    ├── run_inference.sh            # run Goal Force on a run list
    ├── eval_cf.py                  # CausalCascade RQ2 Effect / Preserve IoU
    ├── object_timing.py            # per-object moved / onset / order analysis
    └── third_party/                # CausalCascade metric code (MIT, unmodified)
```

Subset: `l_turn_domino_obstacle_course`, `straight_domino_line`,
`ball_drop_stack_collapse`, `simple_platform_collapse` (21 RQ2 pairs).

## Run

**1. Inference (GPU).** Goal Force checkpoint and Wan models go under `goal-force/` as in
its README.

```bash
cd goal-force
bash ../experiments/goalforce_causalcascade/run_inference.sh                       # force 30
INPUTS=inputs_f100 bash ../experiments/goalforce_causalcascade/run_inference.sh    # force 100
LIST=/path/to/list.txt bash ../experiments/goalforce_causalcascade/run_inference.sh  # custom list
# optional: MODEL_CKPT_PATH=..., SEED=5, DEVICE_ID / WORLD_SIZE for multi-GPU
```

Outputs: `goal-force/checkpoints/goal_force/step-3000-videos/`

**2. Evaluation (CPU).** Needs the CausalCascade reference videos: `git lfs pull`.

```bash
pip install opencv-python numpy pandas matplotlib
E=experiments/goalforce_causalcascade
python $E/eval_cf.py       --gen-dir <videos> --out-dir $E/results/cf
python $E/object_timing.py --gen-dir <videos> --out-dir $E/results/timing
# other force setting: add --manifest $E/inputs_f100/manifest.csv
```

**3. Rebuild inputs (optional).**

```bash
pip install bpy pandas pyarrow pillow
python $E/make_goalforce_csvs.py [--force 100]
python $E/make_object_boxes.py
python $E/preview_inputs.py [--inputs inputs_f100]
```

## Notes

- CSV coordinates are in Goal Force's 832×480 space, with y measured from the bottom.
  Input frames are saved as RGB, because CausalCascade PNGs use a color palette.
- Mode is direct force on the event-graph trigger. The force is identical within a pair.
  Mass edits keep only their direction, encoded in the [1, 4] range.
- Details: `experiments/goalforce_causalcascade/README.md`.
