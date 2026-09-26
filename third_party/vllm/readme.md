# vLLM overlay

LatentGRM inference requires the included overlay for **vLLM 0.26.0**.

## Installation

Run from the repository root:

```bash
pip install -r requirements-inference.txt
python -m latentgrm.vllm_support
```

This creates `.runtime/vllm` and applies `third_party/vllm/overlay`. `evaluate.py` activates it automatically.

## Run LatentGRM inference

Run one benchmark:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python evaluate.py \
  --backend vllm \
  --model outputs/LatentGRM-8B/stage2/hf \
  --benchmark rewardbench \
  --vote 5 \
  --tensor-parallel-size 8
```

LatentGRM uses custom engine and sampling parameters that are not exposed by the OpenAI-compatible request schema, so use `evaluate.py` rather than stock `vllm serve`.

Upstream configuration documentation: [vLLM 0.26.0](https://docs.vllm.ai/en/v0.26.0/cli/serve/). Upstream files and modifications are distributed under [Apache-2.0](LICENSE).
