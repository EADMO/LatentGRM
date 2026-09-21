# LatentGRM

Implementation of **LatentGRM**, a generative reward model that evaluates response pairs through rubric-guided latent reasoning.

## Overview

LatentGRM compresses explicit evaluation traces into latent chains and learns to generate these chains before predicting a preference. Each latent state is a weighted combination of vocabulary embeddings.

- **Semantic Chunking** partitions reasoning traces using rubric structure and linguistic boundaries.
- **Two-stage training** learns an encoder-decoder system, then trains the reward model with latent supervision and preference labels.
- **Latent inference** supports single judgments and multi-vote evaluation with an optimized vLLM backend.
- **Latent interpretation** reconstructs explicit reasoning from latent states using an optional decoder.

## Installation

Run the following commands from the repository root on Linux:

```bash
conda create -n latentgrm-train python=3.12 -y
conda activate latentgrm-train
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash-attn==2.8.3 --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Training uses PyTorch, Transformers, PEFT, and DeepSpeed. See [requirements.md](requirements.md) for CUDA and inference dependencies.

## Model and Data Preparation

```bash
export HF_ENDPOINT=https://hf-mirror.com
python download.py
python prepare_data.py train
```

`download.py` downloads Qwen3-8B, OpenRubrics, the rubric generator, RewardBench, RewardBench 2, and the spaCy English parser. Model and dataset versions are listed in [configs/assets.json](configs/assets.json). `HF_ENDPOINT` is optional when downloading directly from Hugging Face.

Training data is saved to `data/train.jsonl`. Preparation converts OpenRubrics into the judge format and removes examples with empty candidate responses, yielding **35,612 examples**.

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train.sh
```

The launcher runs Semantic Chunking, encoder training, decoder training, joint training, latent target export, decoder merging, and Stage 2 training in sequence. Training uses Qwen3-8B, seed 42, a compression rate of 8, top-10 latent interpolation, and LoRA rank 64. Each Stage 1 phase runs for 3 epochs, and **Stage 2 runs for 9 epochs**.

Hyperparameters are set in [configs/qwen3_8b.json](configs/qwen3_8b.json). See [Training](docs/training.md) for individual stages, Semantic Chunking, and checkpoint usage.

The final model is exported to `outputs/Qwen3-8B/stage2/hf/`. Rerunning the training command resumes an interrupted run and skips completed stages.

## Evaluation

### Install the inference environment

```bash
conda create -n latentgrm-infer python=3.12 -y
conda activate latentgrm-infer
pip install -r requirements-inference.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m latentgrm.vllm_support
```

The [vLLM extension](third_party/vllm/readme.md) supports latent sampling, tensor-parallel top-k projection, cached decode embeddings, and continuous multi-vote generation.

### Prepare benchmarks and generate rubrics

```bash
python prepare_data.py benchmarks
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_rubrics.py --benchmark rewardbench
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_rubrics.py --benchmark rewardbench2
```

| Benchmark | Subsets | Directional pairs |
| --- | --- | ---: |
| RewardBench | Chat, Chat Hard | 1,628 |
| RewardBench 2 | Precise IF, Focus | 3,930 |

Preparation includes both candidate orders. Rubrics are generated with `OpenRubrics/RubricRM-8B-Rubric-v2` using greedy decoding and up to 1,024 output tokens. Each distinct request receives one rubric, shared across its response pairs. Generation resumes automatically when rerun.

The resulting evaluation inputs are `data/eval/rewardbench.jsonl` and `data/eval/rewardbench2.jsonl`. Use `--tensor-parallel-size` to set the GPU count or `--backend transformers` for rubric generation with Transformers.

### Evaluate LatentGRM

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/evaluate.sh
```

This runs single-vote and five-vote evaluation on both benchmarks. To run an individual evaluation:

```bash
python evaluate.py --benchmark rewardbench --vote 5
python evaluate.py --benchmark rewardbench2 --vote 5
```

Evaluation loads `outputs/Qwen3-8B/stage2/hf/` by default. Use `--model` to select another model, `--tensor-parallel-size` to set the GPU count, or `--backend hf` to use Transformers. The default decoding settings use seed 42, top-10 interpolation, Gumbel temperature and noise scale 1, a context length of 6,144, up to 256 latent steps, and 8 answer tokens.

Results and metric summaries are written to `outputs/evaluation/<benchmark>/<checkpoint>/vote<N>/`. Rerunning the command continues from saved results. Use `--output` to select a different output file.

## Optional Interpreter

The interpreter learns to reconstruct explicit reasoning from the joint encoder's latent states. After training the main model, run:

```bash
conda activate latentgrm-train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_interpreter.sh
```

See [Interpreter](docs/interpreter.md) for training settings, reconstruction, latent controls, and fidelity metrics.

## Acknowledgments

The training implementation builds on [Latent-SFT](https://github.com/DJC-GO-SOLO/Latent-SFT). We use OpenRubrics for training and RewardBench and RewardBench 2 for evaluation. See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream licenses.
