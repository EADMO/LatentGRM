from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .paths import resolve_repo_path


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    # Qwen3 changes its assistant prefix when thinking is enabled. Keep this
    # task-level so rubric generation uses the requested chat template.
    enable_thinking: bool = False

def infer_base_model_dir(adapter_dir: str | Path) -> str | None:
    adapter_config = Path(adapter_dir) / "adapter_config.json"
    if not adapter_config.exists():
        return None
    data = json.loads(adapter_config.read_text(encoding="utf-8"))
    base_model = data.get("base_model_name_or_path")
    return str(base_model) if base_model else None

def resolve_model_checkpoint_dir(
    model_dir: str | Path,
) -> Path:
    resolved = resolve_repo_path(model_dir)
    nested_hf = resolved / "hf"
    if (
        not (resolved / "config.json").is_file()
        and not (resolved / "adapter_config.json").is_file()
        and (nested_hf / "config.json").is_file()
    ):
        return nested_hf
    return resolved

class TransformersGenerator:
    def __init__(self, model_dir: str | Path, base_model_dir: str | Path | None = None):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for inference.") from exc

        self.torch = torch
        load_model_dir, adapter_dir = resolve_load_model_dir(model_dir, base_model_dir)
        tokenizer_dir = adapter_dir or load_model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        base_model = AutoModelForCausalLM.from_pretrained(
            str(load_model_dir),
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        if adapter_dir is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("peft is required to load LoRA adapter checkpoints.") from exc
            self.model = PeftModel.from_pretrained(
                base_model,
                str(adapter_dir),
                is_trainable=False,
                local_files_only=True,
            )
        else:
            self.model = base_model
        self.model.eval()

    def render_chat_prompt(self, prompt: str, *, enable_thinking: bool = False) -> str:
        if self.tokenizer.chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return prompt

    def encode_prompts(self, prompts: list[str]):
        rendered = [self.render_chat_prompt(prompt) for prompt in prompts]
        return self.tokenizer(rendered, return_tensors="pt", padding=True)

    def encode_prompt(self, prompt: str):
        return self.encode_prompts([prompt])

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        inputs = self.tokenizer(
            self.render_chat_prompt(prompt, enable_thinking=config.enable_thinking),
            return_tensors="pt",
        ).to(self.model.device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if config.temperature > 0:
            kwargs["temperature"] = config.temperature
        if config.temperature > 0 and config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.temperature > 0 and config.top_k is not None and config.top_k > 0:
            kwargs["top_k"] = config.top_k
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        with self.torch.no_grad():
            output = self.model.generate(**inputs, **kwargs)
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_batch(self, prompts: list[str], config: GenerationConfig) -> list[str]:
        if not prompts:
            return []
        rendered = [
            self.render_chat_prompt(prompt, enable_thinking=config.enable_thinking)
            for prompt in prompts
        ]
        inputs = self.tokenizer(rendered, return_tensors="pt", padding=True).to(self.model.device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if config.temperature > 0:
            kwargs["temperature"] = config.temperature
        if config.temperature > 0 and config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.temperature > 0 and config.top_k is not None and config.top_k > 0:
            kwargs["top_k"] = config.top_k
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        with self.torch.no_grad():
            output = self.model.generate(**inputs, **kwargs)
        prompt_len = inputs["input_ids"].shape[-1]
        generated = output[:, prompt_len:]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

def resolve_load_model_dir(
    model_dir: str | Path,
    base_model_dir: str | Path | None = None,
) -> tuple[Path, Path | None]:
    model_dir = resolve_model_checkpoint_dir(model_dir)
    adapter_dir = model_dir if (model_dir / "adapter_config.json").exists() else None
    load_model_dir = model_dir
    if adapter_dir is not None:
        if base_model_dir is not None:
            load_model_dir = resolve_model_checkpoint_dir(
                base_model_dir
            )
        else:
            inferred = infer_base_model_dir(adapter_dir)
            if inferred:
                load_model_dir = resolve_repo_path(inferred)
    return load_model_dir, adapter_dir

class VLLMAdapterGenerator:
    def __init__(self, llm: Any, tokenizer: Any, adapter_dir: Path | None, adapter_name: str, adapter_id: int):
        self.llm = llm
        self.tokenizer = tokenizer
        self.lora_request = None
        if adapter_dir is not None:
            try:
                from vllm.lora.request import LoRARequest
            except ImportError as exc:
                raise RuntimeError("vllm with LoRA support is required for --backend vllm.") from exc
            self.lora_request = LoRARequest(adapter_name, adapter_id, str(adapter_dir))

    def render_chat_prompt(self, prompt: str, *, enable_thinking: bool = False) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return prompt

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        return self.generate_batch([prompt], config)[0]

    def generate_batch(self, prompts: list[str], config: GenerationConfig) -> list[str]:
        if not prompts:
            return []
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise RuntimeError("vllm is required for --backend vllm.") from exc

        kwargs: dict[str, Any] = {
            "max_tokens": config.max_new_tokens,
            "temperature": config.temperature,
        }
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.top_k is not None:
            kwargs["top_k"] = config.top_k
        sampling_params = SamplingParams(**kwargs)
        rendered = [
            self.render_chat_prompt(prompt, enable_thinking=config.enable_thinking)
            for prompt in prompts
        ]
        # Callers provide their own task-level progress bar. vLLM's per-call
        # bars otherwise redraw "Rendering/Processed prompts" for every batch.
        outputs = self.llm.generate(
            rendered,
            sampling_params,
            lora_request=self.lora_request,
            use_tqdm=False,
        )
        return [output.outputs[0].text.strip() for output in outputs]

def load_single_generator(
    backend: str,
    model_dir: str | Path,
    base_model_dir: str | Path | None = None,
    vllm_tensor_parallel_size: int = 1,
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_max_lora_rank: int = 64,
    adapter_name: str = "rubric",
    tokenizer_dir: str | Path | None = None,
) -> Any:
    if backend == "transformers":
        return TransformersGenerator(model_dir, base_model_dir)
    if backend != "vllm":
        raise ValueError(f"unsupported backend: {backend}")

    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError("vllm is required for --backend vllm. Install requirements-inference.txt.") from exc

    load_model_dir, adapter_dir = resolve_load_model_dir(model_dir, base_model_dir)
    resolved_tokenizer_dir = (
        resolve_repo_path(tokenizer_dir)
        if tokenizer_dir is not None
        else (adapter_dir or load_model_dir)
    )
    enable_lora = adapter_dir is not None
    print(f"[model] requested checkpoint: {Path(model_dir).resolve()}", flush=True)
    print(f"[model] resolved vLLM model: {load_model_dir}", flush=True)
    print(
        f"[model] resolved LoRA adapter: {adapter_dir or 'none'}",
        flush=True,
    )
    print(f"[model] resolved tokenizer: {resolved_tokenizer_dir}", flush=True)
    llm = LLM(
        model=str(load_model_dir),
        tokenizer=str(resolved_tokenizer_dir),
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=vllm_tensor_parallel_size,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
        enable_lora=enable_lora,
        max_loras=1,
        max_lora_rank=vllm_max_lora_rank,
    )
    tokenizer = llm.get_tokenizer()
    return VLLMAdapterGenerator(llm, tokenizer, adapter_dir, adapter_name, 1)
