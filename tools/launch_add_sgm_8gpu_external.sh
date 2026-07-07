#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ISAACGYM_SETUP="${ISAACGYM_SETUP:-/home/ubuntu/FAR/holosoma/scripts/source_isaacgym_setup.sh}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [[ -f "$ISAACGYM_SETUP" ]]; then
  # shellcheck source=/dev/null
  source "$ISAACGYM_SETUP"
fi

ARG_FILE="${ARG_FILE:-args/add_smpl_generalsit_heightmap_args.txt}"
OUT_DIR="${OUT_DIR:-output/add_sgm_8gpu_debug}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_SAMPLES="${MAX_SAMPLES:-327680}"
MASTER_PORT="${MASTER_PORT:-6132}"
LOGGER="${LOGGER:-wandb}"
SAVE_INT_MODELS="${SAVE_INT_MODELS:-false}"
MODEL_FILE="${MODEL_FILE:-}"

export WANDB_ENTITY="${WANDB_ENTITY:-zihanw22}"
export WANDB_PROJECT="${WANDB_PROJECT:-sgm}"
export WANDB_NAME="${WANDB_NAME:-add-sgm-8gpu-debug}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR"

pids=()
for rank in $(seq 0 "$((NUM_GPUS - 1))"); do
  model_args=()
  if [[ -n "$MODEL_FILE" ]]; then
    model_args=(--model_file "$MODEL_FILE")
  fi

  CUDA_VISIBLE_DEVICES="$rank" python mimickit/run.py \
    --arg_file "$ARG_FILE" \
    --mode train \
    --visualize false \
    --logger "$LOGGER" \
    "${model_args[@]}" \
    --devices cuda:0 \
    --proc_rank "$rank" \
    --num_procs "$NUM_GPUS" \
    --master_port "$MASTER_PORT" \
    --num_envs "$NUM_ENVS" \
    --max_samples "$MAX_SAMPLES" \
    --save_int_models "$SAVE_INT_MODELS" \
    --out_dir "$OUT_DIR" \
    > "$OUT_DIR/rank_${rank}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
