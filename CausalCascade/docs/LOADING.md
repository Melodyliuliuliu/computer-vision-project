# Loading Guide

Start with the [README](../README.md#load-media-and-annotations) for loader installation and single-scene/pair examples. This guide covers metadata-only access, raw files, annotations, and batching. All examples use the public `test` split.

## What Gets Loaded

| Route | Behavior |
|---|---|
| `load_dataset()` | Loads the Parquet table. Native image/video columns may decode on row access; `*_path` strings are not automatically resolved. |
| `CausalCascadeDataset` | Loads the table, removes native viewer columns, then resolves requested files inside `__getitem__`. |
| `hf_hub_download()` | Downloads one named repository file and returns its local cached filename. |

The wrapper fetches whole MP4 files before decoding; it does not stream video frames over HTTP. Files are cached, so repeated access normally reuses the local download. Images embedded in the Parquet viewer columns are still part of the initial table download.

With the wrapper, images decode by default, videos do not, and JSON annotations load by default. Set `decode_images=False`, `decode_video=False`, and `load_annotations=False` for metadata-only samples. Disabling video decoding leaves the relative `*_path` fields; it does not produce a new local video path.

## Filter and Join Metadata

### Select Edits

```python
from datasets import load_dataset

repo = "deepdeltalab/CausalCascade"
edits = load_dataset(repo, "counterfactuals", split="test")
move_edits = edits.filter(lambda row: row["edit_type"] == "move")
official = edits.filter(lambda row: row["in_rq2_subset"])
assert len(official) == 225
row = move_edits[0]
print(row["variant_id"], row["target_object_id"], row["edit_instruction"])
print(row["counterfactual_reference_video_path"])  # None outside RQ2
```

`counterfactuals` has no native media preview columns, so filtering its metadata does not invoke a video decoder. Filter on `edit_type` values `delete`, `move`, `scale`, or `mass`; variant IDs use directory prefixes such as `move_type/`.

### Get Questions for a Scene

```python
from datasets import load_dataset

repo = "deepdeltalab/CausalCascade"
scene_id = "Domino_Series/ball_trigger_step_domino_chain"
questions = load_dataset(repo, "diagnostic_qa", split="test")
scene_questions = questions.filter(lambda row: row["scene_id"] == scene_id)
q = scene_questions[0]
print((q["scene_id"], q["question_id"]))
print(q["dimension"], q["question"])
print(q["answer_options"], q["reference_answer"])
print(q["related_event_ids"], q["related_edge_ids"])
```

Use `(scene_id, question_id)` as the unique key; question IDs can repeat across scenes. These are reference-video answers, not predictions for model-generated videos. Raw QA JSON uses `ground_truth_answer` and `source_event_ids` / `source_edge_ids`; the flattened table uses `reference_answer` and `related_event_ids` / `related_edge_ids`.

### Attach Base Graph/QA to a Pair

```python
from causalcascade import CausalCascadeDataset

repo = "deepdeltalab/CausalCascade"
base = CausalCascadeDataset(
    repo_id=repo, subset="base_scenes", decode_images=False, decode_video=False,
)
pairs = CausalCascadeDataset(
    repo_id=repo, subset="rq2_pairs", decode_images=False, decode_video=False,
)
index = {scene: i for i, scene in enumerate(base.table["scene_id"])}
pair = pairs[0]
factual = base[index[pair["scene_id"]]]
graph = factual["event_graph"]
qa = factual["diagnostic_qa"]["questions"]
delta = pair["delta"]
print(pair["variant_id"], len(graph["events"]), len(qa))
```

Graphs describe observable events and direct causal dependencies, with temporal stages on nodes. The pair's `delta` describes the intervention. This join gives the **factual** graph and QA; it does not construct counterfactual graphs or answers.

## Download Raw Assets Without the Wrapper

To get an actual local filename, resolve a `*_path` with `hf_hub_download`. This works for PNG, MP4, JSON, prompt text, and Blender files:

```python
import json
from datasets import load_dataset
from huggingface_hub import hf_hub_download

repo = "deepdeltalab/CausalCascade"
base = load_dataset(repo, "base_scenes", split="test").select_columns([
    "scene_id", "initial_frame_path", "reference_video_path", "event_graph_path",
])
row = base[0]  # No native viewer columns remain, so no TorchCodec is needed.

def fetch(path):
    return hf_hub_download(repo_id=repo, repo_type="dataset", filename=path)

image_path = fetch(row["initial_frame_path"])
video_path = fetch(row["reference_video_path"])
graph_path = fetch(row["event_graph_path"])
with open(graph_path, encoding="utf-8-sig") as stream:
    graph = json.load(stream)
print(image_path, video_path, len(graph["events"]))
```

Native `datasets.Video` decoding needs compatible TorchCodec dependencies. For metadata, select columns before indexing/filtering rows. For decoded video tensors without TorchCodec, use the OpenCV wrapper.

## Tensor Format and Batching

The loader returns the original table fields plus requested decoded media/annotations:

| Output | Format |
|---|---|
| `initial_frame` or factual/counterfactual equivalents | RGB `torch.uint8`, `(3, H, W)`, values 0–255. |
| `reference_video` or factual/counterfactual equivalents | RGB `torch.uint8`, `(T, 3, H, W)`. |
| `event_graph`, `diagnostic_qa`, `scene_metadata` | Parsed per-scene JSON dictionaries for `base_scenes`. |
| `delta` | Parsed edit JSON for `counterfactuals` and `rq2_pairs`. |
| `*_path` | Original repository-relative strings, retained after decoding. |

No float conversion or normalization is applied. If a model expects floats in `[0, 1]`, convert explicitly: `video.float() / 255`.

```python
from torch.utils.data import DataLoader
from causalcascade import CausalCascadeDataset, collate_fn

def main():
    dataset = CausalCascadeDataset(
        repo_id="deepdeltalab/CausalCascade", subset="rq2_pairs",
        decode_video=True, video_size=(240, 426),
        video_frame_stride=4, video_max_frames=16,
        load_annotations=False,
    )
    loader = DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(batch["scene_id"])
    video = batch["factual_reference_video"]
    print(video.shape if hasattr(video, "shape") else [v.shape for v in video])

if __name__ == "__main__":
    main()
```

`collate_fn` recursively batches dictionaries. Equal-shaped tensors stack into `(B, ...)`; unequal-shaped tensors and string/list values remain lists. JSON dictionaries are recursively collated, not kept as a list of original dictionaries; inspect annotations per sample if you need their unbatched structure.

`video_size=(height, width)` applies only to videos, not initial images. `video_frame_stride` samples every nth frame; `video_max_frames` caps the number of retained frames. These controls are for inspection or downstream loading, **not official scoring**. Start with `num_workers=0`; when increasing workers in a Windows Python script, retain the `if __name__ == "__main__"` guard.

## Local Data and Reproducibility

For a downloaded HF-format repository containing the tables and referenced files:

```python
from causalcascade import CausalCascadeDataset

dataset = CausalCascadeDataset(
    local_root="/path/to/CausalCascade", subset="base_scenes", decode_video=True,
)
sample = dataset[0]
```

Provide exactly one of `local_root` or `repo_id`. A loader-only installation is not a local dataset checkout: it lacks `tables/` and media. Local loading preserves the Parquet feature schema while bypassing remote file resolution.

For reproducible Hub loading, pass the dataset commit hash as `revision` and optionally set `cache_dir`. This fixes which table and asset versions are downloaded:

```python
dataset = CausalCascadeDataset(
    repo_id="deepdeltalab/CausalCascade", subset="base_scenes",
    revision="<dataset-commit-hash>", cache_dir="./hf-cache",
)
```

For official evaluation, use complete stored MP4s and the companion metric scripts; preview sampling is not the benchmark protocol. See [Evaluation](EVALUATION.md) and the [GitHub usage guide](https://github.com/SJTU-DeepDeltaLab/CausalCascade/blob/main/USAGE.md).
