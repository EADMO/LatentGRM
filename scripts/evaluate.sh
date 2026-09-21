#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for benchmark in rewardbench rewardbench2; do
  for vote in 1 5; do
    python evaluate.py --benchmark "$benchmark" --vote "$vote" "$@"
  done
done
