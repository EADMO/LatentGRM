# Latent Interpreter

The optional interpreter is initialized from pretrained Qwen3-4B or Qwen3-8B and learns to reconstruct explicit reasoning from the joint encoder's weighted top-10 latent targets. Each latent is represented as a weighted sum of token embeddings from the selected interpreter.

## Training

After exporting the Stage 1 cache and latent targets, run:

```bash
conda activate latentgrm-train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_interpreter.sh
```

This initializes Qwen3-8B. To initialize Qwen3-4B instead:

```bash
python download.py --assets qwen3-4b
BASE_MODEL=models/Qwen3-4B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_interpreter.sh
```

Outputs are saved under `outputs/interpreter/Qwen3-8B/model/` or `outputs/interpreter/Qwen3-4B/model/`. Set `OUTPUT_DIR` to choose another output directory and `NPROC_PER_NODE` to select the GPU count. The Python entry point also accepts `--base-model`. Both sizes use the Qwen3 tokenizer and the main model's cache and latent targets specified in `configs/interpreter.json`.

Training uses 2 epochs, LoRA rank 32, learning rate `1e-5`, and full latent prefixes. Examples sharing a prompt are assigned to the same train, validation, or test split. Paths are in [configs/interpreter.json](../configs/interpreter.json), and training settings are in [scripts/train_interpreter.sh](../scripts/train_interpreter.sh).

Training resumes automatically; use `--resume <checkpoint-directory>` with `python -m latentgrm.interpreter.train` to select a checkpoint.

## Reconstruction

```bash
CUDA_VISIBLE_DEVICES=0 python -m latentgrm.interpreter.evaluate \
  --paths configs/interpreter.json \
  --split-file outputs/interpreter/split.json \
  --adapter outputs/interpreter/Qwen3-8B/model/explainer_lora \
  --output outputs/interpreter/Qwen3-8B/evaluation.json \
  --split test --max-records 256 --views 1 --free-generation-records 16
```

For the 4B interpreter, use its adapter and output directories. Evaluation loads the pretrained base recorded in the adapter; `--base-model` overrides its location if the model directory has moved.

Teacher-forced reconstruction compares correct, absent, shuffled, reversed, uniform, and zero latent inputs. Free generation uses correct, absent, shuffled, and zero inputs by default; set `--free-generation-modes` to choose other controls.

The default input is cached joint-encoder targets. Use `--dynamic-latents` to supply a `.pt` file containing generated latent trajectories.

## Fidelity Metrics

```bash
pip install -r requirements-interpreter.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python download.py --assets bertscore
python -m latentgrm.interpreter.score \
  --evaluation outputs/interpreter/Qwen3-8B/evaluation.json \
  --gold-answers data/train.jsonl --output outputs/interpreter/Qwen3-8B/fidelity.json
```

Metrics cover criterion statuses, final judgments, and justification similarity. The semantic metric uses RoBERTa-large with BERTScore. Use `--skip-semantic` to compute only criterion-status and judgment metrics.
