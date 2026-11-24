# Dataset schema

The four configurations expose factual scenes, controlled intervention variants, the designated RQ2 subset, and graph-grounded diagnostic questions. All `*_path` columns are relative to the repository root. Lists are stored as native Parquet list columns. Heterogeneous intervention deltas are preserved in `edit_params_json` and as raw JSON files.

Native media columns supplement the relative `*_path` columns: images contain
embedded bytes and videos contain `hf://datasets/deepdeltalab/CausalCascade@main/`
repository URIs accepted by the Hub viewer. `base_scenes` has
`initial_frame` and `reference_video`; `rq2_pairs` has factual and counterfactual
image/video columns. `counterfactuals` contains metadata and nullable media
paths, with previews available in `rq2_pairs`. These columns enable previews
in the Hub viewer without duplicating video bytes in Parquet.

## `base_scenes` (50 rows)

| Field | Type | Description |
|---|---|---|
| `scene_id` | string | Stable `<series>/<scene>` identifier. |
| `series`, `scene_name` | string | Scene family and local name. |
| `initial_frame_path` | string | Base first frame. |
| `reference_video_path` | string | Base ground-truth rollout. |
| `event_graph_path` | string | Reference causal event graph of observable events and direct causal dependencies. |
| `qa_path` | string | Graph-grounded, closed-ended diagnostic VQA JSON. |
| `scene_metadata_path` | string | Blender-derived scene metadata. |
| `prompt_level1`, `prompt_level2`, `prompt_level3` | string | State-only, trigger-aware, and full event-chain future-event specification. |
| `num_rigid_bodies` | int | Objects with rigid-body metadata. |
| `num_objects` | int | Total objects recorded in the scene metadata. |
| `num_events`, `num_causal_edges`, `temporal_depth` | int | Event-graph statistics. |
| `blender_file_path` | string | Source Blender scene, downloadable separately from media. |

Event graph edges represent direct causal dependencies, not mere temporal precedence. Relative timing is recorded by node temporal stages. The prompt levels vary future-event specification; they need not be literal extensions of one another.

## `counterfactuals` (2,832 rows)

| Field | Type | Description |
|---|---|---|
| `variant_id` | string | Stable `<edit_type>/<series>/<scene>/<edit>` identifier. |
| `scene_id` | string | Corresponding base scene. |
| `edit_type` | string | One of `delete`, `mass`, `move`, or `scale`. |
| `target_object_id` | string | Metadata object key affected by the edit. |
| `target_object_description` | string or null | Semantic alias when explicitly stored by the edit. |
| `edit_params_json` | string | Complete normalized JSON serialization of the intervention. |
| `delta_json_path` | string | Raw intervention JSON. |
| `edit_instruction_path`, `edit_instruction` | string | Natural-language edit instruction. |
| `in_rq2_subset` | bool | Whether the variant belongs to the official RQ2 subset. |
| `counterfactual_initial_frame_path` | string or null | Present only for the 225 RQ2 variants. |
| `counterfactual_reference_video_path` | string or null | Present only for the 225 RQ2 variants. |

Each variant applies one intervention to one selected object. Delete, move, and scale change the initial observation; mass changes the physical mass without changing the initial frame. The retained set includes downstream event-structure changes and finer-grained motion or timing changes.

## `rq2_pairs` (225 rows)

Each row joins one base scene to one selected counterfactual variant. The model-blind subset comprises 50 delete, 50 move, 50 scale, and 75 mass pairs. Factual media are deduplicated under `assets/base`; counterfactual media are deduplicated under `assets/counterfactual`.

The `factual_prompt_level1` and `counterfactual_prompt_level1` fields provide matched Level 1 conditions, with only intervention-affected attributes updated. Factual and counterfactual reference videos come from separate simulations, not manual video edits.

| Field | Description |
|---|---|
| `pair_id`, `scene_id`, `variant_id`, `edit_type` | Pair identity, base-scene join key, variant identity, and intervention type. |
| `factual_initial_frame_path`, `counterfactual_initial_frame_path` | Repository-relative PNG paths. |
| `factual_reference_video_path`, `counterfactual_reference_video_path` | Repository-relative MP4 paths. |
| `factual_prompt_level1`, `counterfactual_prompt_level1` | Full generation prompts for the two initial states. |
| `counterfactual_prompt_path` | Raw counterfactual prompt text file. |
| `delta_json_path` | Raw intervention JSON. |

The loader adds decoded media fields without the `_path` suffix when requested, and parses `delta_json_path` into `delta`. It does not attach a base graph or QA to pair rows automatically: join `base_scenes` on `scene_id`. Counterfactual event graphs and QA are not supplied in these configurations.

## `diagnostic_qa` (872 rows)

Each row contains a closed-ended diagnostic question, its reference answer, and native list-valued `related_event_ids` and `related_edge_ids`. These references are validated against the corresponding event graph. The five dimensions are event occurrence, causal pathway, temporal ordering, outcome and termination, and scene consistency. Questions are traceable to events, dependencies, temporal stages, or participants; this table is not a set of human verdicts on model-generated videos.

Use `(scene_id, question_id)` as the unique question key; `question_id` alone can repeat across scenes. The flattened table uses `reference_answer`, `dimension`, and `related_event_ids` / `related_edge_ids`. Raw per-scene QA JSON uses `ground_truth_answer`, `diagnostic_dimension`, and `source_event_ids` / `source_edge_ids` for the same information. The loader preserves the raw JSON schema under `diagnostic_qa`.
