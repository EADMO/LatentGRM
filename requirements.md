# Environment requirements

The software stack uses Linux, Python 3.12, and CUDA 13. The training launcher uses eight GPUs. Training depends on PyTorch 2.11.0, Transformers 5.5.3, PEFT 0.18.1, Accelerate 1.11.0, DeepSpeed 0.19.4, and FlashAttention 2.8.3. Semantic preprocessing uses spaCy 3.8.15 with `en_core_web_sm` 3.8.0. Exact Python package constraints are in `requirements.txt`; inference dependencies are in `requirements-inference.txt`.

## Training

```bash
conda create -n latentgrm-train python=3.12 -y
conda activate latentgrm-train
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install flash-attn==2.8.3 --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Install a CUDA toolkit and a compatible NVIDIA driver before building CUDA extensions. FlashAttention is installed after PyTorch because its build imports PyTorch. A source build also requires a C++ compiler, `ninja`, and sufficient host memory. Training saves merged models, optimizer checkpoints, and sparse latent targets.

## Optimized inference

```bash
conda create -n latentgrm-infer python=3.12 -y
conda activate latentgrm-infer
python -m pip install -r requirements-inference.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m latentgrm.vllm_support
```

The overlay requires **vLLM 0.26.0** and its compatible compiled wheel. Do not substitute another vLLM version without porting the overlay. The installer copies Python modules into `.runtime/vllm` and links the wheel's unchanged shared libraries; it does not modify `site-packages`. Rubric generation uses ordinary vLLM, while `evaluate.py` activates the latent extension before importing vLLM.

## Downloads

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
python download.py
```

`HF_ENDPOINT` is optional outside mirror-based environments. Repository revisions are pinned in `configs/assets.json`. Authentication, if needed for an asset, is supplied through `HF_TOKEN`; credentials are never stored in the repository. The spaCy model is downloaded from its official release. No pre-generated evaluation rubric files are required.

If the training host cannot reach GitHub, download the [official spaCy wheel](https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl) on a connected machine, transfer it, and run `python download.py --assets parser --parser-wheel <wheel-file>`. The installer extracts the parser into `models/en_core_web_sm/`.
