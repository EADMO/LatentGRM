#!/usr/bin/env python
"""Initialize the latent interpreter from the Stage 1 decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from filelock import FileLock
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unpack(payload: dict[str, Any], field: str, row: int) -> list[int]:
    offsets = payload[f"{field}_offsets"]
    start, end = int(offsets[row]), int(offsets[row + 1])
    return payload[f"{field}_values"][start:end].long().tolist()


class ShardedLatentStore:
    """Small per-process LRU over cache/soft-label shards."""

    def __init__(self, cache_dir: str, label_dir: str, max_cached_shards: int = 2):
        self.cache_dir = Path(cache_dir)
        self.label_dir = Path(label_dir)
        self.manifest = load_json(self.cache_dir / "manifest.json")
        self.label_manifest = load_json(self.label_dir / "manifest.json")
        self.shards = self.manifest["shards"]
        self.num_records = int(self.manifest["num_records"])
        if int(self.label_manifest["num_records"]) != self.num_records:
            raise ValueError("cache and soft-label manifests disagree on num_records")
        for field in ("compression_rate", "use"):
            if self.manifest.get(field) != self.label_manifest.get(field):
                raise ValueError(
                    f"cache and soft-label manifests disagree on {field}: "
                    f"{self.manifest.get(field)!r} != "
                    f"{self.label_manifest.get(field)!r}"
                )
        if bool(self.label_manifest.get("full_vocab", False)):
            raise ValueError("full-vocabulary soft labels are not supported")
        self.max_cached_shards = max_cached_shards
        self._lru: OrderedDict[str, tuple[dict[str, Any], Any]] = OrderedDict()

    def _locate(self, index: int) -> tuple[dict[str, Any], int]:
        if not 0 <= index < self.num_records:
            raise IndexError(index)
        # Shards are sorted and currently uniform; the loop also supports a
        # shorter final shard and future non-uniform manifests.
        for shard in self.shards:
            if int(shard["start"]) <= index < int(shard["end"]):
                return shard, index - int(shard["start"])
        raise RuntimeError(f"No shard contains record {index}")

    def get(self, index: int) -> dict[str, Any]:
        shard, row = self._locate(index)
        filename = shard["file"]
        if filename not in self._lru:
            cache = torch.load(
                self.cache_dir / filename, map_location="cpu", weights_only=False
            )
            labels = torch.load(
                self.label_dir / filename, map_location="cpu", weights_only=False
            )
            self._lru[filename] = (cache, labels)
            self._lru.move_to_end(filename)
            while len(self._lru) > self.max_cached_shards:
                self._lru.popitem(last=False)
        else:
            self._lru.move_to_end(filename)
        cache, labels = self._lru[filename]
        probs, token_ids = labels[row]
        record = {
            "record_index": index,
            "prefix_ids": unpack(cache, "prefix_ids", row),
            "explicit_ids": unpack(cache, "explicit_ids", row),
            "frontier_ends": unpack(cache, "frontier_ends", row),
            "latent_probs": probs.float(),
            "latent_token_ids": token_ids.long(),
        }
        if (
            len(record["frontier_ends"]) != record["latent_probs"].shape[0]
            or record["latent_probs"].shape != record["latent_token_ids"].shape
        ):
            raise ValueError(
                f"Record {index} cache/label mismatch: "
                f"{len(record['frontier_ends'])}, "
                f"{tuple(record['latent_probs'].shape)}, "
                f"{tuple(record['latent_token_ids'].shape)}"
            )
        frontiers = record["frontier_ends"]
        if (
            not frontiers
            or any(left >= right for left, right in zip([0] + frontiers, frontiers))
            or frontiers[-1] != len(record["explicit_ids"])
        ):
            raise ValueError(
                f"Record {index} has invalid frontiers for "
                f"{len(record['explicit_ids'])} explicit tokens"
            )
        if not torch.isfinite(record["latent_probs"]).all():
            raise ValueError(f"Record {index} has non-finite latent probabilities")
        if (record["latent_probs"] < 0).any() or (
            record["latent_probs"].sum(dim=-1) <= 0
        ).any():
            raise ValueError(f"Record {index} has invalid latent probabilities")
        return record


class DynamicLatentStore:
    """Overlay autoregressively generated Top-K traces on immutable targets."""

    def __init__(self, static_store: ShardedLatentStore, trace_path: str):
        self.static = static_store
        payload = torch.load(trace_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "latent-trajectory":
            raise ValueError(f"unexpected dynamic trace schema: {payload.get('schema')}")
        self.records = {int(key): value for key, value in payload["records"].items()}
        self.num_records = static_store.num_records
        self.manifest = dict(static_store.manifest)
        self.label_manifest = dict(static_store.label_manifest)

    def get(self, index: int) -> dict[str, Any]:
        base = self.static.get(index)
        trace = self.records.get(int(index))
        if trace is None:
            raise KeyError(f"dynamic trace does not contain record {index}")
        probs = trace["probs"].float()
        token_ids = trace["token_ids"].long()
        if probs.ndim != 2 or probs.shape != token_ids.shape or not len(probs):
            raise ValueError(f"invalid/empty dynamic trace for record {index}: {tuple(probs.shape)}")
        target_len = len(base["explicit_ids"])
        count = len(probs)
        # Full training always consumes the entire chain and complete explicit
        # target. Intermediate quantiles only preserve dataset invariants and
        # auxiliary boundary bookkeeping.
        frontiers = [max(1, math.ceil(target_len * (i + 1) / count)) for i in range(count)]
        frontiers[-1] = target_len
        base.update({
            "frontier_ends": frontiers,
            "latent_probs": probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12),
            "latent_token_ids": token_ids,
        })
        return base


def deterministic_frontier(count: int, record_index: int, view: int, seed: int) -> int:
    if count <= 0:
        raise ValueError("latent count must be positive")
    digest = hashlib.blake2b(
        f"{seed}:{record_index}:{view}".encode(), digest_size=8
    ).digest()
    return 1 + int.from_bytes(digest, "little") % count


class BackwardPrefixDataset(Dataset):
    def __init__(
        self,
        store: ShardedLatentStore,
        indices: list[int],
        tokenizer,
        views_per_record: int,
        seed: int,
        max_length: int,
        validation_quantiles: bool = False,
        include_shuffled_control: bool = False,
        stochastic_frontiers: bool = False,
        latent_order: str = "forward",
        full_frontier_probability: float = 0.0,
        soft_mode: str = "weighted",
        soft_temperature: float = 1.0,
        input_mode: str = "latent",
    ):
        self.store = store
        self.indices = indices
        self.tokenizer = tokenizer
        self.views_per_record = views_per_record
        self.seed = seed
        self.max_length = max_length
        self.validation_quantiles = validation_quantiles
        self.include_shuffled_control = include_shuffled_control
        self.stochastic_frontiers = stochastic_frontiers
        if not 0.0 <= full_frontier_probability <= 1.0:
            raise ValueError("full_frontier_probability must be in [0, 1]")
        self.full_frontier_probability = full_frontier_probability
        if soft_mode not in {"weighted", "top1", "uniform", "temperature"}:
            raise ValueError(f"unsupported soft_mode={soft_mode}")
        if soft_temperature <= 0:
            raise ValueError("soft_temperature must be positive")
        self.soft_mode = soft_mode
        self.soft_temperature = soft_temperature
        if input_mode not in {"latent", "prompt_only"}:
            raise ValueError(f"unsupported input_mode={input_mode}")
        self.input_mode = input_mode
        if latent_order not in {"forward", "reverse"}:
            raise ValueError(f"Unsupported latent_order={latent_order}")
        self.latent_order = latent_order
        self.left_ids = tokenizer("<think>", add_special_tokens=False)["input_ids"]
        self.right_ids = tokenizer("</think>", add_special_tokens=False)["input_ids"]
        if len(self.left_ids) != 1 or len(self.right_ids) != 1:
            raise ValueError((self.left_ids, self.right_ids))
        expected_left = self.store.manifest.get("latent_left_ids")
        expected_right = self.store.manifest.get("latent_right_ids")
        if expected_left is not None and list(expected_left) != self.left_ids:
            raise ValueError(
                f"tokenizer/cache latent-left mismatch: {self.left_ids} != {expected_left}"
            )
        if expected_right is not None and list(expected_right) != self.right_ids:
            raise ValueError(
                f"tokenizer/cache latent-right mismatch: {self.right_ids} != {expected_right}"
            )

    def __len__(self) -> int:
        return len(self.indices) * self.views_per_record

    def __getitem__(self, item: int) -> dict[str, Any]:
        record_index = self.indices[item // self.views_per_record]
        view = item % self.views_per_record
        record = self.store.get(record_index)
        count = len(record["frontier_ends"])
        if self.validation_quantiles:
            fraction = (view + 1) / self.views_per_record
            frontier = max(1, min(count, math.ceil(count * fraction)))
        elif self.stochastic_frontiers:
            # Mix complete-CoT reconstruction with random prefix reconstruction.
            # The complete view directly optimizes the user's primary objective;
            # random frontiers retain temporal localization and faithfulness.
            if float(torch.rand(1)) < self.full_frontier_probability:
                frontier = count
            else:
                frontier = int(torch.randint(1, count + 1, (1,)).item())
        else:
            frontier = deterministic_frontier(count, record_index, view, self.seed)
        explicit_end = int(record["frontier_ends"][frontier - 1])
        target_ids = record["explicit_ids"][:explicit_end] + self.right_ids
        prefix_ids = record["prefix_ids"] + self.left_ids
        latent_start = len(prefix_ids)
        if self.input_mode == "prompt_only":
            # A fair prompt-only baseline is trained on the physically compacted
            # prompt followed immediately by the explicit CoT target.  It does
            # not retain latent-length positional hints or off-manifold zeros.
            input_ids = prefix_ids + target_ids
            target_start = latent_start
        else:
            input_ids = (
                prefix_ids
                + [self.tokenizer.pad_token_id] * frontier
                + target_ids
            )
            target_start = latent_start + frontier
        labels = [-100] * target_start + target_ids
        if len(input_ids) > self.max_length:
            # This should not occur because the original encoder sequence
            # contains all explicit tokens plus all latent placeholders and is
            # already capped at 6144. Fail rather than silently changing the
            # retrospective target.
            raise ValueError(
                f"Record {record_index} length {len(input_ids)} exceeds "
                f"max_length={self.max_length} at frontier={frontier}/{count}"
            )
        probs = record["latent_probs"][:frontier]
        token_ids = record["latent_token_ids"][:frontier]
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if self.soft_mode == "top1":
            chosen = probs.argmax(dim=-1, keepdim=True)
            probs = torch.zeros_like(probs).scatter(-1, chosen, 1.0)
        elif self.soft_mode == "uniform":
            probs = torch.full_like(probs, 1.0 / probs.shape[-1])
        elif self.soft_mode == "temperature":
            probs = probs.clamp_min(1e-12).pow(1.0 / self.soft_temperature)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        if self.latent_order == "reverse":
            probs = torch.flip(probs, dims=(0,))
            token_ids = torch.flip(token_ids, dims=(0,))
        latent_positions = list(range(latent_start, latent_start + frontier))
        if self.input_mode == "prompt_only":
            latent_positions = []
            probs = probs[:0]
            token_ids = token_ids[:0]
        result = {
            "record_index": record_index,
            "frontier": frontier,
            "latent_count": count,
            "input_ids": input_ids,
            "labels": labels,
            "latent_positions": latent_positions,
            "latent_token_ids": token_ids,
            "latent_probs": probs,
            "target_length": len(target_ids),
            # Starts (relative to the target) of the explicit compression
            # chunks.  Evaluation uses these to report boundary-token NLL in
            # addition to the usual, heavily teacher-forced corpus NLL.
            "target_chunk_starts": [0] + record["frontier_ends"][: frontier - 1],
        }
        if self.include_shuffled_control:
            if len(self.indices) < 2:
                raise ValueError("shuffled control requires at least two records")
            record_position = item // self.views_per_record
            # A fixed derangement at record level: every view of a target uses
            # the next held-out record, never itself and never a training row.
            donor_index = self.indices[(record_position + 1) % len(self.indices)]
            donor = self.store.get(donor_index)
            donor_count = len(donor["frontier_ends"])
            if self.validation_quantiles:
                donor_frontier = max(
                    1, min(donor_count, math.ceil(donor_count * (view + 1) / self.views_per_record))
                )
            else:
                donor_frontier = deterministic_frontier(
                    donor_count, donor_index, view, self.seed
                )
            # Match the target's number of latent slots by sampling the donor
            # prefix at equal relative-progress quantiles.  This prevents
            # sequence length/position from identifying the control.
            donor_positions = torch.linspace(
                0, donor_frontier - 1, steps=frontier
            ).round().long()
            donor_probs = donor["latent_probs"][donor_positions]
            donor_token_ids = donor["latent_token_ids"][donor_positions]
            donor_probs = donor_probs / donor_probs.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            if self.soft_mode == "top1":
                chosen = donor_probs.argmax(dim=-1, keepdim=True)
                donor_probs = torch.zeros_like(donor_probs).scatter(-1, chosen, 1.0)
            elif self.soft_mode == "uniform":
                donor_probs = torch.full_like(donor_probs, 1.0 / donor_probs.shape[-1])
            elif self.soft_mode == "temperature":
                donor_probs = donor_probs.clamp_min(1e-12).pow(1.0 / self.soft_temperature)
                donor_probs = donor_probs / donor_probs.sum(dim=-1, keepdim=True)
            if self.latent_order == "reverse":
                donor_probs = torch.flip(donor_probs, dims=(0,))
                donor_token_ids = torch.flip(donor_token_ids, dims=(0,))
            result.update(
                {
                    "shuffled_record_index": donor_index,
                    "shuffled_latent_token_ids": donor_token_ids,
                    "shuffled_latent_probs": donor_probs,
                }
            )
        return result


@dataclass
class BackwardCollator:
    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_size = len(examples)
        seq_len = max(len(x["input_ids"]) for x in examples)
        seq_len = math.ceil(seq_len / self.pad_to_multiple_of) * self.pad_to_multiple_of
        latent_len = max(len(x["latent_positions"]) for x in examples)
        top_k = int(examples[0]["latent_token_ids"].shape[-1])
        input_ids = torch.full(
            (batch_size, seq_len), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
        position_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
        latent_positions = torch.full(
            (batch_size, latent_len), -1, dtype=torch.long
        )
        latent_token_ids = torch.zeros(
            (batch_size, latent_len, top_k), dtype=torch.long
        )
        latent_probs = torch.zeros(
            (batch_size, latent_len, top_k), dtype=torch.float32
        )
        target_chunk_len = max(len(x["target_chunk_starts"]) for x in examples)
        target_chunk_starts = torch.full(
            (batch_size, target_chunk_len), -1, dtype=torch.long
        )
        has_shuffled = "shuffled_latent_token_ids" in examples[0]
        shuffled_latent_token_ids = (
            torch.zeros_like(latent_token_ids) if has_shuffled else None
        )
        shuffled_latent_probs = torch.zeros_like(latent_probs) if has_shuffled else None
        for row, example in enumerate(examples):
            n = len(example["input_ids"])
            m = len(example["latent_positions"])
            input_ids[row, :n] = torch.tensor(example["input_ids"], dtype=torch.long)
            labels[row, :n] = torch.tensor(example["labels"], dtype=torch.long)
            attention_mask[row, :n] = 1
            position_ids[row, :n] = torch.arange(n, dtype=torch.long)
            latent_positions[row, :m] = torch.tensor(
                example["latent_positions"], dtype=torch.long
            )
            latent_token_ids[row, :m] = example["latent_token_ids"]
            latent_probs[row, :m] = example["latent_probs"]
            starts = example["target_chunk_starts"]
            target_chunk_starts[row, : len(starts)] = torch.tensor(
                starts, dtype=torch.long
            )
            if has_shuffled:
                shuffled_latent_token_ids[row, :m] = example[
                    "shuffled_latent_token_ids"
                ]
                shuffled_latent_probs[row, :m] = example["shuffled_latent_probs"]
        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "latent_positions": latent_positions,
            "latent_token_ids": latent_token_ids,
            "latent_probs": latent_probs,
            "target_chunk_starts": target_chunk_starts,
        }
        if has_shuffled:
            result["shuffled_latent_token_ids"] = shuffled_latent_token_ids
            result["shuffled_latent_probs"] = shuffled_latent_probs
        return result


class BackwardDecoder(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.config = decoder.config

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.decoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.decoder.gradient_checkpointing_disable()

    def prepare_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        latent_positions: torch.Tensor,
        latent_token_ids: torch.Tensor,
        latent_probs: torch.Tensor,
        latent_mode: str = "correct",
        shuffled_latent_token_ids: torch.Tensor | None = None,
        shuffled_latent_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed = self.decoder.get_input_embeddings()
        inputs_embeds = embed(input_ids)
        probs = latent_probs
        token_ids = latent_token_ids
        if latent_mode == "reverse":
            probs = probs.clone()
            token_ids = token_ids.clone()
            for row in range(probs.shape[0]):
                count = int(latent_positions[row].ge(0).sum())
                probs[row, :count] = torch.flip(probs[row, :count], dims=(0,))
                token_ids[row, :count] = torch.flip(
                    token_ids[row, :count], dims=(0,)
                )
        elif latent_mode == "uniform_topk":
            valid = probs.sum(dim=-1, keepdim=True).gt(0)
            probs = torch.where(
                valid, torch.full_like(probs, 1.0 / probs.shape[-1]), probs
            )
        elif latent_mode == "shuffled":
            if shuffled_latent_token_ids is None or shuffled_latent_probs is None:
                raise ValueError("shuffled mode requires matched donor latents")
            token_ids = shuffled_latent_token_ids
            probs = shuffled_latent_probs
        elif latent_mode == "prompt_only":
            raise ValueError(
                "prompt_only requires physically compacting away latent slots; "
                "use evaluate_backward_decoder.compact_prompt_only"
            )
        elif latent_mode not in {"correct", "zero"}:
            raise ValueError(f"Unsupported latent_mode={latent_mode}")
        mixed = (
            F.embedding(token_ids, embed.weight)
            * probs.to(embed.weight.dtype).unsqueeze(-1)
        ).sum(dim=-2)
        if latent_mode == "zero":
            mixed = torch.zeros_like(mixed)
        rows = []
        for row in range(inputs_embeds.shape[0]):
            valid = latent_positions[row].ge(0)
            positions = latent_positions[row, valid]
            replacement = mixed[row, valid].to(inputs_embeds.dtype)
            if positions.numel():
                index = positions.unsqueeze(-1).expand(-1, inputs_embeds.shape[-1])
                rows.append(inputs_embeds[row].scatter(0, index, replacement))
            else:
                rows.append(inputs_embeds[row])
        inputs_embeds = torch.stack(rows, dim=0)
        return inputs_embeds, attention_mask, position_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        latent_positions: torch.Tensor,
        latent_token_ids: torch.Tensor,
        latent_probs: torch.Tensor,
        latent_mode: str = "correct",
        shuffled_latent_token_ids: torch.Tensor | None = None,
        shuffled_latent_probs: torch.Tensor | None = None,
        **_: Any,
    ):
        inputs_embeds, attention_mask, position_ids = self.prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            latent_positions=latent_positions,
            latent_token_ids=latent_token_ids,
            latent_probs=latent_probs,
            latent_mode=latent_mode,
            shuffled_latent_token_ids=shuffled_latent_token_ids,
            shuffled_latent_probs=shuffled_latent_probs,
        )
        # Required by gradient checkpointing when all base embeddings are
        # frozen and only internal LoRA parameters are trainable.
        if self.training and not inputs_embeds.requires_grad:
            inputs_embeds.requires_grad_(True)
        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=False,
        )


class AdapterOnlyTrainer(Trainer):
    def __init__(self, *args, tokenizer_for_save=None, metadata=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer_for_save = tokenizer_for_save
        self.metadata = metadata or {}
        self.model_accepts_loss_kwargs = False

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights
        wrapped = model or self.model
        adapter = Path(resume_from_checkpoint) / "explainer_lora"
        state = load_peft_weights(str(adapter), device="cpu")
        set_peft_model_state_dict(wrapped.decoder, state, adapter_name="default")

    def _save(self, output_dir=None, state_dict=None):
        del state_dict
        output_dir = output_dir or self.args.output_dir
        if not self.is_world_process_zero():
            return
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        if not isinstance(model.decoder, PeftModel):
            raise TypeError("expected the explainer decoder to be a PeftModel")
        model.decoder.save_pretrained(output / "explainer_lora")
        if self.tokenizer_for_save is not None:
            self.tokenizer_for_save.save_pretrained(output / "tokenizer")
        (output / "experiment_metadata.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        torch.save(self.args, output / "training_args.bin")


def build_split_indices(
    store: ShardedLatentStore, split_file: Path, seed: int
) -> dict[str, list[int]]:
    split_file.parent.mkdir(parents=True, exist_ok=True)
    # torchrun executes this code independently on every rank before Trainer
    # initializes DDP. Serialize split creation so eight ranks cannot truncate
    # and rewrite the same JSON concurrently.
    with FileLock(str(split_file) + ".lock"):
        if split_file.exists():
            payload = load_json(split_file)
            if int(payload["num_records"]) != store.num_records:
                raise ValueError("split manifest record count does not match cache")
            return {
                name: list(map(int, payload[name]))
                for name in ("train", "validation", "test")
            }

        result = {"train": [], "validation": [], "test": []}
        # Hash prefix ids so exact duplicate prompts cannot cross splits.
        for index in range(store.num_records):
            prefix_ids = store.get(index)["prefix_ids"]
            key = hashlib.sha256(
                torch.tensor(prefix_ids, dtype=torch.int32).numpy().tobytes()
            ).digest()
            bucket = int.from_bytes(key[:8], "little") % 1000
            name = "train" if bucket < 800 else "validation" if bucket < 900 else "test"
            result[name].append(index)
        payload = {
            "schema": "interpreter-split",
            "seed": seed,
            "grouping": "sha256(prefix_token_ids)",
            "num_records": store.num_records,
            **result,
        }
        split_file.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return result


def limited(indices: list[int], maximum: int | None, seed: int) -> list[int]:
    if maximum is None or maximum < 0 or len(indices) <= maximum:
        return indices
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(indices), generator=generator)[:maximum].tolist()
    return [indices[i] for i in order]


def load_interpreter(paths: dict[str, str], tokenizer, args, adapter=None):
    """Load the Stage 1 decoder with a new or trained interpreter LoRA."""
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        paths["base_model"],
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        use_cache=False,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    embedding_count = base.get_input_embeddings().num_embeddings
    if embedding_count < len(tokenizer):
        raise ValueError(
            f"Stage 1 decoder tokenizer exceeds model embeddings: "
            f"embeddings={embedding_count}, tokenizer={len(tokenizer)}"
        )
    base.config.use_cache = False
    if adapter is not None:
        return BackwardDecoder(
            PeftModel.from_pretrained(base, adapter, is_trainable=False)
        )
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    decoder = get_peft_model(base, config)
    decoder.print_trainable_parameters()
    return BackwardDecoder(decoder)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(ROOT / "configs/interpreter.json"))
    parser.add_argument(
        "--base-model",
        help="Stage 1 decoder checkpoint directory; defaults to configs/interpreter.json.",
    )
    parser.add_argument(
        "--split-file", default=str(ROOT / "outputs/backward_splits.json")
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default="auto", help="auto, none, or a full checkpoint directory")
    parser.add_argument(
        "--dynamic-latents",
        help="Autoregressively generated Top-K trace used for train/validation inputs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-eval-records", type=int, default=256)
    parser.add_argument("--train-views", type=int, default=1)
    parser.add_argument("--eval-views", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument(
        "--checkpoint-strategy",
        choices=["steps", "epoch"],
        default="steps",
        help="Save and evaluate either every configured number of steps or once per epoch.",
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--latent-order", choices=["forward", "reverse"], default="forward"
    )
    parser.add_argument("--full-frontier-probability", type=float, default=0.5)
    parser.add_argument("--soft-mode", choices=["weighted", "top1", "uniform", "temperature"], default="weighted")
    parser.add_argument("--soft-temperature", type=float, default=1.0)
    parser.add_argument(
        "--input-mode", choices=["latent", "prompt_only"], default="latent",
        help="Training input. prompt_only physically removes every latent slot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Bind each torchrun worker before any helper (including set_seed) can
    # initialize a CUDA context on the default GPU 0.
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    set_seed(args.seed)
    paths = load_json(args.paths)
    if args.base_model:
        paths["base_model"] = args.base_model
        paths["tokenizer"] = args.base_model
    tokenizer = AutoTokenizer.from_pretrained(
        paths["tokenizer"], local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    static_store = ShardedLatentStore(paths["stage1_cache"], paths["latent_soft_labels"])
    store = (
        DynamicLatentStore(static_store, args.dynamic_latents)
        if args.dynamic_latents else static_store
    )
    splits = build_split_indices(store, Path(args.split_file), args.seed)
    train_indices = limited(splits["train"], args.max_train_records, args.seed)
    eval_indices = limited(splits["validation"], args.max_eval_records, args.seed + 1)
    train_dataset = BackwardPrefixDataset(
        store,
        train_indices,
        tokenizer,
        views_per_record=args.train_views,
        seed=args.seed,
        max_length=args.max_length,
        stochastic_frontiers=True,
        latent_order=args.latent_order,
        full_frontier_probability=args.full_frontier_probability,
        soft_mode=args.soft_mode,
        soft_temperature=args.soft_temperature,
        input_mode=args.input_mode,
    )
    eval_dataset = BackwardPrefixDataset(
        store,
        eval_indices,
        tokenizer,
        views_per_record=args.eval_views,
        seed=args.seed + 1,
        max_length=args.max_length,
        validation_quantiles=True,
        latent_order=args.latent_order,
        soft_mode=args.soft_mode,
        soft_temperature=args.soft_temperature,
        input_mode=args.input_mode,
    )
    model = load_interpreter(paths, tokenizer, args)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=args.bf16,
        tf32=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy=args.checkpoint_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.checkpoint_strategy,
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    metadata = {
        "schema": "interpreter-training",
        "interpreter_initialization": paths["base_model"],
        "objective": "latent prefix z_1..z_t -> explicit CoT prefix x_1..x_t",
        "paths": paths,
        "split_file": args.split_file,
        "train_records": len(train_indices),
        "eval_records": len(eval_indices),
        "train_views": args.train_views,
        "eval_views": args.eval_views,
        "seed": args.seed,
        "lora_rank": args.lora_rank,
        "latent_order": args.latent_order,
        "full_frontier_probability": args.full_frontier_probability,
        "soft_mode": args.soft_mode,
        "soft_temperature": args.soft_temperature,
        "input_mode": args.input_mode,
        "checkpoint_strategy": args.checkpoint_strategy,
        "dynamic_latents": args.dynamic_latents,
    }
    trainer = AdapterOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=BackwardCollator(tokenizer.pad_token_id),
        tokenizer_for_save=tokenizer,
        metadata=metadata,
    )
    resume = None
    if args.resume != "none":
        candidates = ([Path(args.resume)] if args.resume != "auto" else
                      sorted(Path(args.output_dir).glob("checkpoint-*"),
                             key=lambda p: int(p.name.split("-")[-1]), reverse=True))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        for candidate in candidates:
            required = [candidate / name for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt",
                        "explainer_lora/adapter_config.json", "explainer_lora/adapter_model.safetensors")]
            required += [candidate / ("rng_state.pth" if world == 1 else f"rng_state_{rank}.pth") for rank in range(world)]
            if all(path.is_file() and path.stat().st_size for path in required):
                resume = str(candidate); break
        if args.resume != "auto" and resume is None:
            raise ValueError("The requested interpreter checkpoint lacks complete training state")
    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    trainer.save_state()
    metrics = dict(train_result.metrics)
    metrics.update(trainer.evaluate())
    metrics.update(
        {
            "train_records": len(train_indices),
            "eval_records": len(eval_indices),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        }
    )
    if trainer.is_world_process_zero():
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
