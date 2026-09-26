# LatentGRM

Implementation of **LatentGRM**, a generative reward model that evaluates response pairs through rubric-guided latent reasoning.

## Overview

LatentGRM compresses explicit evaluation traces into latent chains and learns to generate these chains before predicting a preference. Each latent state is a weighted combination of vocabulary embeddings.

- **Semantic Chunking** partitions reasoning traces using rubric structure and linguistic boundaries.
- **Two-stage training** learns an encoder-decoder system, then trains the reward model with latent supervision and preference labels.
- **Latent inference** supports single judgments and multi-vote evaluation with an optimized vLLM backend.
- **Latent interpretation** initializes an interpreter from the Stage 1 decoder to reconstruct explicit reasoning from latent states.

## Installation

Run the following commands from the repository root on Linux:

```bash
conda create -n latentgrm-train python=3.12 -y
conda activate latentgrm-train
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

Training uses PyTorch, Transformers, PEFT, and DeepSpeed on Linux with Python 3.12 and CUDA 13. Install FlashAttention after the other packages because its build imports PyTorch.

## Model and Data Preparation

```bash
python download.py
python prepare_data.py train
```

`download.py` downloads Qwen3-4B, Qwen3-8B, OpenRubrics, the rubric generator, RewardBench, RewardBench 2, PPE-IFEval, IFBench, RM-Bench, HelpSteer3, and the spaCy English parser. Model and dataset versions are listed in [configs/assets.json](configs/assets.json).

Training data is saved to `data/train.jsonl`. Preparation converts OpenRubrics directly into LatentGRM training records and skips records with empty candidate responses.

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train.sh
```

The launcher runs Semantic Chunking, encoder training, decoder training, joint training, latent target export, decoder merging, and Stage 2 training in sequence. Training uses Qwen3-8B, seed 42, a compression rate of 8, top-10 latent interpolation, and LoRA rank 64. Each Stage 1 phase runs for 3 epochs, and **Stage 2 runs for 9 epochs**.

Hyperparameters are set in [configs/qwen3_8b.json](configs/qwen3_8b.json). See [Training](docs/training.md) for individual stages, Semantic Chunking, and checkpoint usage.

The 4B model uses the same recipe:

```bash
python train.py --config configs/qwen3_4b.json
```

The final model is exported to `outputs/LatentGRM-8B/stage2/hf/`, or `outputs/LatentGRM-4B/stage2/hf/` with the 4B configuration. Rerunning the training command resumes an interrupted run and skips completed stages.

## Evaluation

### Install the inference environment

```bash
conda create -n latentgrm-infer python=3.12 -y
conda activate latentgrm-infer
pip install -r requirements-inference.txt
python -m latentgrm.vllm_support
```

The [vLLM extension](third_party/vllm/readme.md) requires vLLM 0.26.0 and supports latent sampling, tensor-parallel top-k projection, cached decode embeddings, and continuous multi-vote generation.

### Prepare benchmarks and generate rubrics

```bash
python prepare_data.py benchmarks
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/generate_rubrics.sh
```

| Benchmark | Evaluation subsets | Evaluated directional pairs |
| --- | --- | ---: |
| RewardBench | Chat, Chat Hard | 1,628 |
| RewardBench 2 | Precise IF, Focus | 3,930 |
| PPE-IFEval | Sampled conflict pairs | 5,120 |
| IFBench | All released preference pairs | 888 |
| RM-Bench | Chat | 2,322 |
| HelpSteer3 | Non-tie preference validation pairs | 3,834 |

Data preparation and rubric generation preserve the complete source ordering and both candidate orders; evaluation selects the subsets listed above. Rubrics are generated with `OpenRubrics/RubricRM-8B-Rubric-v2` using greedy decoding and up to 1,024 output tokens. Each source prompt receives one rubric, shared across its response pairs and both candidate orders. Generation resumes automatically when rerun.

Evaluation inputs are saved to `data/eval/<benchmark>.jsonl`. To prepare or generate rubrics for one benchmark, use `python prepare_data.py benchmarks --benchmark rm-bench` or `python generate_rubrics.py --benchmark rm-bench`. Use `--tensor-parallel-size` to set the GPU count or `--backend transformers` for rubric generation with Transformers.

### Evaluate LatentGRM

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/evaluate.sh
```

This runs single-vote and five-vote evaluation on all six benchmarks using the subsets listed above. To run an individual evaluation:

```bash
python evaluate.py --benchmark rewardbench --vote 5
python evaluate.py --benchmark rewardbench2 --vote 5
```

Evaluation loads `outputs/LatentGRM-8B/stage2/hf/` by default. Use `--model` to select another model, `--tensor-parallel-size` to set the GPU count, or `--backend hf` to use Transformers. The default decoding settings use seed 42, top-10 interpolation, Gumbel temperature and noise scale 1, a context length of 6,144, up to 256 latent steps, and 8 answer tokens.

Scoring follows each benchmark's aggregation: weighted section accuracy for RewardBench; prompt-level success for RewardBench 2; conflict-pair accuracy for PPE-IFEval; pairwise accuracy for IFBench and HelpSteer3; and the mean of Easy, Normal, and Hard accuracies for RM-Bench Chat. Scores are reported for each candidate order and their average.

Results and metric summaries are written to `outputs/evaluation/<benchmark>/LatentGRM-8B/vote<N>/`, or `LatentGRM-4B` when evaluating the 4B model. Rerunning the command continues from saved results. Use `--output` to select a different output file.

## Optional Interpreter

The optional interpreter is initialized from the Stage 1 decoder at `outputs/LatentGRM-8B/decoder/hf/` and learns to reconstruct explicit reasoning from the joint encoder's latent states. After completing Stage 1 and exporting its latent targets, run:

```bash
conda activate latentgrm-train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_interpreter.sh
```

The launcher uses the Stage 1 decoder for both model weights and tokenization. Set `DECODER_MODEL` only when the decoder was exported to another directory. Outputs are saved under `outputs/interpreter/stage1-decoder/model/`.

See [Interpreter](docs/interpreter.md) for training settings, reconstruction, latent controls, and fidelity metrics.

## Acknowledgments

The training implementation builds on [Latent-SFT](https://github.com/DJC-GO-SOLO/Latent-SFT). We use OpenRubrics for training and the six datasets listed above for evaluation. See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream licenses.
