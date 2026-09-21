#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec torchrun --standalone --nproc_per_node=8 --module latentgrm.interpreter.train \
  --paths configs/interpreter.json --split-file outputs/interpreter/split.json \
  --output-dir outputs/interpreter/model --seed 42 \
  --input-mode latent --latent-order forward --soft-mode weighted \
  --soft-temperature 1 --full-frontier-probability 1 \
  --train-views 1 --eval-views 1 --max-eval-records 512 \
  --num-train-epochs 2 --learning-rate 1e-5 --lora-rank 32 \
  --per-device-train-batch-size 1 --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 4 --checkpoint-strategy epoch \
  --logging-steps 10 --dataloader-num-workers 0 "$@"
