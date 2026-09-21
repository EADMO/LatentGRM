#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for benchmark in rewardbench rewardbench2 ppe-ifeval ifbench rm-bench helpsteer3; do
  python generate_rubrics.py --benchmark "$benchmark" "$@"
done
