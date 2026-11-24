---
license: cc-by-4.0
pretty_name: CausalCascade
task_categories:
  - image-to-video
tags:
  - video
  - image-to-video
  - physics
  - causal-reasoning
  - counterfactual
  - datasets
configs:
  - config_name: base_scenes
    default: true
    data_files:
      - split: test
        path: tables/base_scenes.parquet
  - config_name: counterfactuals
    data_files:
      - split: test
        path: tables/counterfactual_variants.parquet
  - config_name: rq2_pairs
    data_files:
      - split: test
        path: tables/rq2_pairs.parquet
  - config_name: diagnostic_qa
    data_files:
      - split: test
        path: tables/diagnostic_questions.parquet
---

# CausalCascade

[![Paper](https://img.shields.io/badge/Paper-arXiv_2610.15562-red)](https://arxiv.org/abs/2610.15562)
[![Paper](https://img.shields.io/badge/Paper-Hugging%20Face-yellow)](https://huggingface.co/papers/2610.15562)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/deepdeltalab/CausalCascade)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/SJTU-DeepDeltaLab/CausalCascade)

**CausalCascade** is a simulator-grounded benchmark for evaluating *causal cascade generation* in image-to-video models: generating a complete multi-stage physical process in which initiating events propagate through causally dependent interactions to a terminal outcome. The task tests faithful realization of a causal process, not merely a plausible event or final state.

The release contains **50 event-rich Blender scenes**, structured causal event graphs, **872 graph-grounded diagnostic questions**, three levels of future-event specification, and **2,832 controlled counterfactual variants**. A subset of **225 matched factual-counterfactual pairs** supports evaluation of selective responsiveness to physical interventions.

## Dataset Overview

### Scene Diversity

![Overview](assets/showcase/scene_overview.png)

### Three Levels of Future-Event Specification

The same initial frame and reference rollout are paired with three textual conditions: **state-only** (describes the current scene state and relevant physical conditions), **trigger-aware** (additionally specifies the initiating event), and **full event-chain** (verbalizes the major event chain). 

<details>
<summary>View example: Y-shaped domino cascade</summary>

![Case preview](assets/showcase/reference_cascade.gif)

<details>
<summary>Show the three prompt levels</summary>

**Level 1: State-only**

```text
Generate a physically realistic continuation video by predicting the subsequent physical evolution from the given fixed-camera first frame.

Important instructions:
- Preserve the fixed camera view.
- The first frame is authoritative for all visible geometry, object sizes, initial poses, contact relations, spatial layout, and camera view.
- Treat only the listed dynamic active objects and dynamic active object groups as freely moving rigid bodies.
- Treat anchored static colliders as fixed to the world.

Global rigid-body convention:
- All rigid bodies use friction = 0.50, restitution = 0.10, linear damping = 0.04, angular damping = 0.10.
- For every rigid body, the geometric center is also its center of mass.

Dynamic active objects:
- upper branch target sphere: the large target sphere at the upper/right end of the fork. Mass = 1.0 kg.
- lower branch target sphere: the large target sphere at the lower/right end of the fork. Mass = 1.0 kg.

Dynamic active object groups:
- central stem domino row: the five upright dominoes forming the straight stem before the fork splits. This group contains 5 dominoes. Each object has mass = 1.0 kg.
- upper fork domino branch: the upper row of upright dominoes after the split. This group contains 9 dominoes. Each object has mass = 1.0 kg.
- lower fork domino branch: the lower row of upright dominoes after the split. This group contains 9 dominoes. Each object has mass = 1.0 kg.

Anchored static objects:
- wooden tabletop surface: the broad surface beneath the forked domino structure.

Decorative context:
- The remaining objects are anchored static decorators and should remain visually unchanged.
```

**Level 2: Trigger-aware**

```text
Generate a physically realistic continuation video by predicting the subsequent physical evolution from the given fixed-camera first frame.

Trigger event: The first tilted blue domino at the left start of the fork falls into the next domino.

Important instructions:
- Preserve the fixed camera view.
- The first frame is authoritative for all visible geometry, object sizes, initial poses, contact relations, spatial layout, and camera view.
- Treat only the listed dynamic active objects and dynamic active object groups as freely moving rigid bodies.
- Treat anchored static colliders as fixed to the world.

Global rigid-body convention:
- All rigid bodies use friction = 0.50, restitution = 0.10, linear damping = 0.04, angular damping = 0.10.
- For every rigid body, the geometric center is also its center of mass.

Dynamic active objects:
- upper branch target sphere: the large target sphere at the upper/right end of the fork. Mass = 1.0 kg.
- lower branch target sphere: the large target sphere at the lower/right end of the fork. Mass = 1.0 kg.

Dynamic active object groups:
- central stem domino row: the five upright dominoes forming the straight stem before the fork splits. This group contains 5 dominoes. Each object has mass = 1.0 kg.
- upper fork domino branch: the upper row of upright dominoes after the split. This group contains 9 dominoes. Each object has mass = 1.0 kg.
- lower fork domino branch: the lower row of upright dominoes after the split. This group contains 9 dominoes. Each object has mass = 1.0 kg.

Anchored static objects:
- wooden tabletop surface: the broad surface beneath the forked domino structure.

Decorative context:
- The remaining objects are anchored static decorators and should remain visually unchanged.
```

**Level 3: Full event-chain**

```text
A curved, slightly branching line of brightly colored domino blocks stands upright across the middle of a wooden tabletop, with a small purple cone at the far left, a small red-and-teal rounded piece behind the dominoes, several little green and brown peg-like toys in front, a large pink ball near the back-right end, and a large light-blue ball near the front-right end. The motion starts at the left end when the first domino tips and hits the next, launching a chain reaction that travels steadily from left to right through the whole arrangement. Each block knocks its neighbor over, and the collapse passes through the center and then fans through the angled right-side branches, so the dominoes slap down in sequence and spread into overlapping lines on the table. As the final rightmost blocks fall, they nudge the two large balls, making the pink ball roll slightly farther back and right and the blue ball roll rightward away from the domino pile. By the end, all of the dominoes lie flat in a scattered fan-shaped heap across the center, the two balls have shifted to the right, and the smaller cone and peg-like pieces remain standing around the wreckage. Keep the camera fixed.
```

</details>

</details>

### Factual–Counterfactual Pairs

Each edit applies a single-object **delete**, **move**, **scale**, or **mass** intervention and provides separately simulated factual and counterfactual references with matched Level 1 prompts.

<details>
<summary>View delete intervention case</summary>

![Fig3_Delete](assets/showcase/delete_pair.gif)

**Description:** Remove the third peg from the left in the top row of the Galton board.

</details>

<details>
<summary>View move intervention case</summary>

![Fig4_Move](assets/showcase/move_pair.gif)

**Description:** Move the first pink domino on the upper branch of the Y-shaped chain toward the upper left.

</details>

<details>
<summary>View scale intervention case</summary>

![Fig5_Scale](assets/showcase/scale_pair.gif)

**Description:** Reduce the height of the rightmost yellow domino on the middle step.

</details>

<details>
<summary>View mass intervention case</summary>

![Fig6_Mass](assets/showcase/mass_pair.gif)

**Description:** Increase the mass of the barrel.

</details>

## Results

### Causal Cascade Fidelity

![RQ1Result](assets/showcase/causal_fidelity.png)

**Left**: Performance of current I2V models under the Level 1 prompt.
**Right**: Model-average performance as future-event specification becomes more explicit.

### Selective Causal Responsiveness

![RQ2Result](assets/showcase/counterfactual_response.png)

**Counterfactual motion response across intervention types. Each marker denotes one I2V model.** Higher values on both axes indicate stronger selective causal responsiveness.

For evaluation code and further usage, see the [GitHub repository](https://github.com/SJTU-DeepDeltaLab/CausalCascade).

## Choose a Configuration

| Configuration | Rows | Use it to access |
|---|---:|---|
| `base_scenes` | 50 | Initial frames, reference videos, three prompt levels, event graphs, QA, and scene metadata. |
| `counterfactuals` | 2,832 | All retained single-object edits, targets, instructions, and JSON deltas. Media paths are populated only for the 225 RQ2 variants. |
| `rq2_pairs` | 225 | Ready-matched factual/counterfactual frames, videos, Level 1 prompts, and edit deltas: 50 delete, 50 move, 50 scale, 75 mass. |
| `diagnostic_qa` | 872 | Individual questions, answer options, reference answers, and graph references across five diagnostic dimensions. |

All configurations use one public benchmark split, `test`. Notice that only the 225 RQ2 variants have edited initial frames and reference videos in this release. Other variants provide edit metadata, which can be used to produce counterfactual media.

### Read Metadata

```bash
pip install "datasets>=4,<5"
```

```python
from datasets import load_dataset

repo = "deepdeltalab/CausalCascade"
base_table = load_dataset(repo, "base_scenes", split="test")
pairs_table = load_dataset(repo, "rq2_pairs", split="test")
edits_table = load_dataset(repo, "counterfactuals", split="test")
qa_table = load_dataset(repo, "diagnostic_qa", split="test")
print(len(base_table), len(pairs_table), len(edits_table), len(qa_table))
# 50 225 2832 872

# Select metadata columns before accessing rows to avoid native video decoding.
row = base_table.select_columns(["scene_id", "reference_video_path"])[0]
print(row)
```

The Hub viewer displays images/videos for `base_scenes` and `rq2_pairs`. The separate `*_path` fields are repository-relative strings. Use the loader below to fetch and decode the assets without manually matching files.

## Load Media and Annotations

Use Python 3.10+. Download just the loader files, then install locally; this does not download the dataset's videos or Blender files:

```bash
pip install -U huggingface_hub
hf download deepdeltalab/CausalCascade --repo-type=dataset --include="causalcascade/*.py" --include="pyproject.toml" --local-dir=CausalCascade-loader
pip install -e "./CausalCascade-loader[video]"
```

The bundled loader resolves assets on access and caches Hub downloads. It uses OpenCV for videos; native `datasets.Video` decoding and TorchCodec are not required for this route.

### One Base Scene

```python
from causalcascade import CausalCascadeDataset

base = CausalCascadeDataset(
    repo_id="deepdeltalab/CausalCascade",
    subset="base_scenes",
    decode_video=False,
)
sample = base[0]
print(sample["scene_id"])
print(sample["initial_frame"].shape)  # RGB uint8: (3, H, W)
print(sample["prompt_level1"])       # State-only
# prompt_level2: trigger-aware; prompt_level3: full event-chain
print(len(sample["event_graph"]["events"]))
print(len(sample["event_graph"]["causal_edges"]))

question = sample["diagnostic_qa"]["questions"][0]
print(question["question"])
print(question["answer_options"], question["ground_truth_answer"])
# sample["scene_metadata"] contains the Blender-derived metadata.
```

With `decode_video=False`, `reference_video_path` remains a relative string and no video is downloaded. Set `decode_video=True` to also obtain `sample["reference_video"]`.

### One Factual–Counterfactual Pair

```python
from causalcascade import CausalCascadeDataset

pairs = CausalCascadeDataset(
    repo_id="deepdeltalab/CausalCascade",
    subset="rq2_pairs",
    decode_video=True,
    video_size=(240, 426),    # Preview size: (height, width)
    video_frame_stride=4,
    video_max_frames=16,     # At most 16 sampled frames per video
)
pair = pairs[0]
print(pair["scene_id"], pair["variant_id"], pair["edit_type"])
print(pair["factual_initial_frame"].shape)
print(pair["counterfactual_initial_frame"].shape)
print(pair["factual_reference_video"].shape)         # (T, 3, 240, 426)
print(pair["counterfactual_reference_video"].shape)
print(pair["factual_prompt_level1"])
print(pair["counterfactual_prompt_level1"])
print(pair["delta"])                                # Parsed intervention JSON
```

Use the two supplied Level 1 prompts as generation condition.`rq2_pairs` provides matched media, prompts, and deltas; base graphs/QA are accessed through `base_scenes` using the same `scene_id`. Counterfactual graphs/QA are not part of these four tables.

### PyTorch DataLoader

Continuing with `pairs` above:

```python
from torch.utils.data import DataLoader
from causalcascade import collate_fn

loader = DataLoader(pairs, batch_size=1, num_workers=0, collate_fn=collate_fn)
batch = next(iter(loader))
print(batch["factual_reference_video"].shape)  # (1, T, 3, H, W)
print(batch["scene_id"])                     # List of scene IDs
```

Images/videos are RGB `uint8` tensors in `C x H x W` / `T x C x H x W` order, with values 0–255. Equal-shaped tensors are stacked; unequal shapes remain lists. Start with `num_workers=0`, especially in Windows/notebooks. See [the loading guide](docs/LOADING.md) for batching, local/offline use, raw file downloads, filtering edits, and joining annotations; see [the schema](docs/SCHEMA.md) for all fields.

## Evaluation

Generate factual rollouts under state-only (L1), trigger-aware (L2), or full event-chain (L3) future-event specification. For counterfactual evaluation, use both sides of the 225 official pairs with their matched Level 1 prompts. Event graphs describe observable events and direct causal dependencies; diagnostic questions assess five complementary dimensions.

See [the evaluation protocol](docs/EVALUATION.md) for metrics and aggregation, and the [GitHub usage guide](https://github.com/SJTU-DeepDeltaLab/CausalCascade/blob/main/USAGE.md) for runnable evaluation commands. 

## License

CausalCascade's original dataset content is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Third-party assets retain their original licenses.

Some `.blend` source scenes and rendered media incorporate textures, HDRIs, and 3D models from [Poly Haven](https://polyhaven.com), released under [CC0 1.0](https://polyhaven.com/license). See [ATTRIBUTION.md](ATTRIBUTION.md).


## Citation

If you use the CausalCascade dataset in your research, please cite:

```bibtex
@misc{causalcascade,
  title         = {A Plausible Future Is Not a Causal Future: Can Video Models Generate How the World Unfolds?},
  author        = {Jialiang Li and Zhengning Xu and Zongyang Ma and Shaobo Cui},
  year          = {2026},
  eprint        = {2610.15562},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2610.15562}
}
```
