# Training

The default configuration trains Qwen3-8B. Use `--config configs/qwen3_4b.json` for the Qwen3-4B model reported in the paper.

`python train.py` runs the full pipeline. The `--stage` argument selects an individual stage.

## Stage 1: Latent Compression

```bash
python train.py --stage cache
python train.py --stage encoder
python train.py --stage decoder
python train.py --stage joint
```

Semantic Chunking tokenizes each explicit reasoning trace and stores its chunk boundaries. The supervision frontier is sampled during training.

The encoder starts from Qwen3-8B and learns to compress explicit reasoning. Decoder training uses the trained encoder and a decoder initialized from Qwen3-8B. Joint training starts from both trained components and updates them together. Each phase runs for 3 epochs.

## Stage 2: Latent Reasoning

```bash
python train.py --stage targets
python train.py --stage merge
python train.py --stage stage2
```

The joint encoder exports sparse top-10 distributions as latent targets. The joint decoder adapter is merged into the trained decoder to initialize Stage 2. The reward model trains for **9 epochs**, with equally weighted latent KL and answer cross-entropy losses.

## Hyperparameters

| Setting | Stage 1, per phase | Stage 2 |
| --- | ---: | ---: |
| Epochs | 3 | 9 |
| GPUs | 8 | 8 |
| Batch size per GPU | 1 | 2 |
| Gradient accumulation | 8 | 16 |
| Global batch size | 64 | 256 |
| Learning rate | `3e-5` | `1e-5` |
| Warmup | 5% of training steps | 70 steps |
| LR scheduler | `WarmupDecayLR` | `WarmupDecayLR` |
| LR decay horizon | Training steps | 1,400 steps |
| Attention | SDPA | FlashAttention 2 |

Both stages use BF16, DeepSpeed ZeRO-2, seed 42, and LoRA with rank 64, alpha 128, and dropout 0.1. Stage 2 uses logarithmic warmup followed by linear learning-rate decay, with the decay horizon configured independently of the training duration. Latent generation uses top-10 interpolation with Gumbel temperature and noise scale set to 1.

Edit [configs/qwen3_8b.json](../configs/qwen3_8b.json) to change the training settings. `python train.py --dry-run` prints the stage commands.

## Semantic Chunking

A trace with `T` tokens receives a latent budget of `ceil(T / 8)`. The method allocates this budget to rubric-structured segments and forms balanced initial chunks. Boundaries are then refined using punctuation, word boundaries, dependency structure, and protected rubric markers.

A constrained dynamic program preserves the segment's latent budget, allowing chunk lengths of 4–16 tokens with a preference for 6–10 tokens. It searches within four tokens of each initial boundary. A five-token search is used when it removes additional within-word cuts. The implementation is in [latentgrm/semantic_chunking](../latentgrm/semantic_chunking), with boundary costs in [refinement/config.py](../latentgrm/semantic_chunking/refinement/config.py).

## Outputs and Resuming

Training outputs are saved under `outputs/Qwen3-8B/`:

| Directory | Contents |
| --- | --- |
| `cache/` | Tokenized traces and chunk boundaries |
| `encoder/`, `decoder/`, `joint/` | Stage 1 models and checkpoints |
| `targets/` | Sparse latent targets |
| `joint_decoder/` | Merged decoder for Stage 2 initialization |
| `stage2/` | Reward model and checkpoints |
| `stage2/hf/` | Final model for evaluation |

The launcher resumes the latest checkpoint automatically. To select one explicitly:

```bash
python train.py --stage stage2 --resume outputs/Qwen3-8B/stage2/checkpoint-1120
```

Checkpoints are saved every epoch with optimizer, scheduler, and random states. Use `--output outputs/new-run` for a separate training run. The run's `code/` directory stores its source and environment settings.
