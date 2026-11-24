# Evaluation protocol

For loading reference samples, see the [README](../README.md) and [loading guide](LOADING.md). This document defines the evaluation protocol; runnable generation and scoring commands are in the [GitHub usage guide](https://github.com/SJTU-DeepDeltaLab/CausalCascade/blob/main/USAGE.md).

## RQ1: causal cascade fidelity

Use `base_scenes` for all 50 factual initial frames and reference rollouts. Generate the complete physical process from the initial frame and one textual condition; the target is a coherent causal cascade, not only an isolated event or final outcome.

### Future-event specification

| Condition | Specification |
|---|---|
| Level 1: state-only | Current scene state and relevant physical conditions, without specifying a future event. |
| Level 2: trigger-aware | The initiating event is additionally specified; subsequent evolution remains unspecified. |
| Level 3: full event-chain | The major intended event chain is verbalized from initiation through intermediate interactions to the downstream outcome. |


### Motion-level evaluation

Following Physics-IQ, compare spatially normalized motion maps from generated and reference rollouts:

- **Spatial IoU:** overlap of regions exhibiting motion anywhere in the rollout.
- **Strict ST-IoU:** motion agreement under fixed frame-level correspondence on the reference physical-time grid.
- **Weighted Spatial IoU:** agreement of spatial motion activity, weighted by the fraction of frames in which each location is active.

Average per-scene scores across the 50 scenes. Global affine alignment and constrained DTW are post-hoc timing diagnostics within RQ1, not separate research questions or replacements for Strict ST-IoU. Affine alignment permits a global shift and rate change; constrained DTW additionally permits local stretching or compression while preserving order.

### Event-level evaluation

`diagnostic_qa` contains closed-ended questions derived from the reference graph's observable events, direct causal dependencies, temporal stages, and participants. Each question retains its source event and edge references where applicable.

| Diagnostic dimension | Evaluation target | Questions |
|---|---|---:|
| Event occurrence | Whether expected trigger, mediator, or outcome events occur. | 152 |
| Causal pathway | Whether the direct dependencies connecting events are realized. | 226 |
| Temporal ordering | Whether the stated events occur in the correct relative order. | 179 |
| Outcome and termination | Whether the cascade reaches the reference outcome and terminal state. | 115 |
| Scene consistency | Whether entities, identity/geometry, trajectories, and the static scene remain coherent. | 200 |
| Total | | 872 |

The paper uses a tool-augmented video-QA evaluator combining full-rollout inspection, temporally aligned views, and object-centric evidence. Reference answers define scoring targets; they are not instructions for generated videos.

Score each question by exact answer agreement. First average within each scene and dimension, then average across scenes for each dimension. **VQA Overall** is the mean of the five dimension scores. Do not pool all questions into one accuracy, which would give more weight to scenes with more questions. Use the released answer options, including absent-event and unreached-state options, without forcing a relation when its event endpoints are missing.

## RQ2: selective causal responsiveness

Use exactly the 225 rows in `rq2_pairs`: 50 delete, 50 move, 50 scale, and 75 mass interventions. Each row binds a factual scene to a separately re-simulated counterfactual reference. The model-blind subset was selected without inspecting model generations; do not substitute other variants when reporting the official RQ2 result.

Each intervention modifies one selected object. Delete removes it; move changes its initial position; scale changes its size or proportions; mass changes only its mass and leaves the initial RGB frame unchanged.

Use the supplied **matched Level 1 prompts**, updating only intervention-affected attributes between the factual and counterfactual conditions. Do not replace them with Level 2/3 descriptions of the reference future. Generate both factual and counterfactual rollouts: evaluating only the edited rollout does not test selective response.

The paired criteria compare generated changes against reference changes:

- **Effect:** whether motion is suppressed or induced in the correct regions and direction.
- **Preserve:** whether motion shared by the factual and counterfactual references remains shared in the generated pair.

Report **CF Spatial Effect IoU**, **CF Weighted Effect IoU**, **CF Preserve Spatial IoU**, and **CF Preserve Weighted IoU**, separately for each intervention type. Effect combines suppression and induction as two directional channels; Preserve compares shared-motion maps. Weighted maps encode motion-active frame fractions rather than speed or force. These criteria test selective causal responsiveness, not simply whether two generations differ.

Average edits within each represented base scene and intervention type, then average those scene scores. Retained variants need not all change the final event structure; some affect only finer-grained motion or timing.

## RQ3: failure localization

Dimension-level VQA scores summarize weaknesses but do not by themselves identify the first failure in a rollout. The paper separately audits 90 rollouts with matched scene-prompt coverage across nine models and complements this with qualitative case analysis.

**Foundational failures** comprise scene consistency and event occurrence; **process failures** comprise causal pathway, temporal ordering, and outcome or termination. The audit localizes the earliest primary deviation rather than treating all low dimension scores as independent initiating failures. Its human audit records are separate from the 872 reference questions; the four released Parquet tables do not encode those audit annotations.
