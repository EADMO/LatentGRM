#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE_MODEL=${BASE_MODEL:-models/Qwen3-8B}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/interpreter/$(basename "$BASE_MODEL")/model}
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" --module latentgrm.interpreter.train \
  --paths configs/interpreter.json --split-file outputs/interpreter/split.json \
  --base-model "$BASE_MODEL" --output-dir "$OUTPUT_DIR" --seed 42 \
  --input-mode latent --latent-order forward --soft-mode weighted \
  --soft-temperature 1 --full-frontier-probability 1 \
  --train-views 1 --eval-views 1 --max-eval-records 512 \
  --num-train-epochs 2 --learning-rate 1e-5 --lora-rank 32 \
  --per-device-train-batch-size 1 --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 4 --checkpoint-strategy epoch \
  --logging-steps 10 --dataloader-num-workers 0 "$@"
