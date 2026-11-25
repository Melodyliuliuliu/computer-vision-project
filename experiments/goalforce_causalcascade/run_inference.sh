#!/usr/bin/env bash
# Run Goal Force on all 35 CausalCascade inputs.
# Run from the goal-force/ directory (same place as its own inference_goal_force.sh):
#   cd goal-force
#   bash ../experiments/goalforce_causalcascade/run_inference.sh
#
# Multi-GPU: start one process per GPU with the same WORLD_SIZE, e.g.
#   DEVICE_ID=0 WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0 bash ../experiments/.../run_inference.sh &
#   DEVICE_ID=1 WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=1 bash ../experiments/.../run_inference.sh &
set -euo pipefail

DEVICE_ID=${DEVICE_ID:-0}
WORLD_SIZE=${WORLD_SIZE:-1}
SEED=${SEED:-5}
MODEL_CKPT_PATH=${MODEL_CKPT_PATH:-"checkpoints/goal_force/step-3000.safetensors"}
# Another force setting: INPUTS=inputs_f100 bash ../experiments/goalforce_causalcascade/run_inference.sh
INPUTS=${INPUTS:-inputs}
LIST=${LIST:-"../experiments/goalforce_causalcascade/${INPUTS}/all_csvs.txt"}

mapfile -t EXAMPLE_PATHS < <(grep -v '^$' "$LIST")
echo "Running ${#EXAMPLE_PATHS[@]} CSVs (device ${DEVICE_ID}/${WORLD_SIZE}, seed ${SEED})"

python scripts/inference/inference_goal_force.py \
  --device_id "${DEVICE_ID}" \
  --world_size "${WORLD_SIZE}" \
  --seed "${SEED}" \
  --control_signal_type goal_force \
  --model_ckpt_path "${MODEL_CKPT_PATH}" \
  --example_paths "${EXAMPLE_PATHS[@]}"
