# Optimized vLLM overlay

This directory contains 13 changed or additional Python modules for **vLLM 0.26.0**. Unmodified Python files and compiled libraries come from the installed wheel.

The selected implementation combines:

- tensor-parallel local top-k candidates instead of full-vocabulary communication;
- cached embeddings for decode inputs;
- GPU latent projection, state updates, and Gumbel sampling;
- compatibility with the model runner's default CUDA graph and asynchronous execution paths.

`latentgrm/evaluation/continuous_native_n.py` adds a continuous native multi-vote queue whose admission accounts for child rollouts. The default target is 80 active rollouts.

## Installation

Create a separate inference environment from the repository root:

```bash
conda create -n latentgrm-infer python=3.12 -y
conda activate latentgrm-infer
pip install -r requirements-inference.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

The overlay requires exactly `vllm==0.26.0`. Build an isolated runtime without modifying the installed wheel:

```bash
python -m latentgrm.vllm_support
```

This copies the installed vLLM package to `.runtime/vllm/vllm`, preserves its compiled libraries, and then copies `third_party/vllm/overlay/vllm` over it. Re-run the command after changing an overlay file; existing runtime files are updated in place.

`evaluate.py` activates this runtime automatically. For another Python entry point, activate it before importing vLLM:

```python
from latentgrm.vllm_support import activate

activate()

from vllm import LLM, SamplingParams, TokensPrompt
```

For shell entry points, export the runtime explicitly:

```bash
export PYTHONPATH="$PWD/.runtime/vllm:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python -c 'import vllm; print(vllm.__file__)'
```

The printed path should start with the repository's `.runtime/vllm/vllm` directory.

To apply the files directly to the active environment instead, locate its package directory and copy the overlay into it:

```bash
VLLM_PACKAGE=$(python -c 'import importlib.metadata as m; print(m.distribution("vllm").locate_file("vllm"))')
cp -a third_party/vllm/overlay/vllm/. "$VLLM_PACKAGE/"
```

The isolated runtime is the default used by this repository and does not alter the environment's vLLM installation.

## Run LatentGRM inference

Run one benchmark with the patched runtime:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python evaluate.py \
  --backend vllm \
  --model outputs/Qwen3-8B/stage2/hf \
  --benchmark rewardbench \
  --vote 5 \
  --tensor-parallel-size 8
```

Run vote@1 and vote@5 on all configured benchmarks:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/evaluate.sh
```

For direct engine integration, latent mode is enabled when the engine is created, while the end token and Gumbel settings are supplied for each request:

```python
from transformers import AutoTokenizer
from latentgrm.vllm_support import activate

activate()
from vllm import LLM, SamplingParams, TokensPrompt

model = "outputs/Qwen3-8B/stage2/hf"
tokenizer = AutoTokenizer.from_pretrained(model)
latent_end = tokenizer("</think>", add_special_tokens=False)["input_ids"][-1]

llm = LLM(
    model=model,
    enable_latent=True,
    latent_end_token_id=latent_end,
    latent_topk=10,
    tensor_parallel_size=8,
    max_model_len=6144,
    gpu_memory_utilization=0.9,
)
sampling = SamplingParams(
    n=5,
    max_tokens=264,
    temperature=0.0,
    latent_end_token_id=latent_end,
    add_noise_gumbel_softmax=True,
    gumbel_softmax_temperature=1.0,
    noise_scale=1.0,
    seed=42,
)
```

The repository uses vLLM's Python engine because latent decoding adds engine and request parameters that are not part of vLLM's OpenAI-compatible request schema. A stock `vllm serve` command therefore does not reproduce LatentGRM inference. `evaluate.py` is the complete runnable entry point and can be used as the reference when wrapping the engine in an application service.

Use the evaluator's command-line options to set the context length, latent budget, vote count, tensor-parallel size, and number of active rollouts. Stock vLLM cannot execute the latent projection interface used by this evaluator.

Upstream configuration documentation: [vLLM 0.26.0](https://docs.vllm.ai/en/v0.26.0/cli/serve/). Upstream files and modifications are distributed under [Apache-2.0](LICENSE).
