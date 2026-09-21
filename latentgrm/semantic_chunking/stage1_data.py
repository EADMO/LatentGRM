from __future__ import annotations

import torch

from torch.utils.data import Dataset

from latentgrm.semantic_chunking.structure import CompressionPlan, normalize_use

from latentgrm.prompting import (
    LATENT_SFT_MAX_LENGTH,
    build_qwen_generation_prefix,
    head_tail_truncate,
)

from latentgrm.training.stage1.data import read_jsonl

class Stage1Dataset(Dataset):
    def __init__(self, path, args, model):
        self.args = args
        self.model = model
        self.data = None
        self.cache = None
        if self.args.stage1_cache_path:
            # Import lazily to avoid a module cycle: the cache builder uses the
            # deterministic preparation helpers defined below.
            from latentgrm.semantic_chunking.stage1_cache import Stage1Cache

            self.cache = Stage1Cache(
                self.args.stage1_cache_path,
                data_path=path,
                compression_rate=self.args.compression_rate,
                use=self.args.use,
                model=model,
            )
            self.total_len = len(self.cache)
        else:
            self.data = read_jsonl(path)
            self.total_len = len(self.data)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if self.cache is not None:
            record = self.cache[idx]
        else:
            record = prepare_stage1_record(
                self.data[idx],
                self.model,
                self.args.compression_rate,
                self.args.use,
            )
        # Keep frontier sampling dynamic.  Only deterministic tokenization and
        # semantic splitting are cached, so every epoch retains the original
        # unbiased random-frontier objective.
        return materialize_stage1_record(record, self.model)

def _render_prefix_and_suffix(example, model):
    prefix = build_qwen_generation_prefix(model.tokenizer, example)
    suffix = example["cot_answer"] + model.tokenizer.eos_token
    return prefix, suffix

def prepare_stage1_record(
    example,
    model,
    compression_rate,
    use,
):
    """Tokenize and split one example deterministically.

    This is the expensive part of Stage-1 and is safe to persist.  The random
    supervision frontier is deliberately not selected here.
    """
    use = normalize_use(use)
    prefix_text, suffix_text = _render_prefix_and_suffix(example, model)
    prefix_ids = model.tokenizer(
        prefix_text,
        truncation=False,
        padding=False,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    suffix_ids = model.tokenizer(
        suffix_text,
        truncation=False,
        padding=False,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    metadata = None
    if use == "semantic":
        from .refinement.compression import build_semantic_plan
        explicit_ids, plan, metadata = build_semantic_plan(
            example["cot"], model.tokenizer, model.compress_token_id, compression_rate
        )
        # The expensive frontier plan is computed once. Only prompt tokens may
        # be truncated; never retokenize or alter the canonical CoT.
        fixed = len(explicit_ids) + plan.latent_count + sum(map(len, model.latent_token_ids))
        if fixed > LATENT_SFT_MAX_LENGTH:
            raise ValueError("Canonical CoT exceeds Stage-1 limit")
        prefix_ids = head_tail_truncate(prefix_ids, LATENT_SFT_MAX_LENGTH - fixed)
    else:
        raise ValueError("Only semantic chunking is supported")
    if plan.latent_count <= 0:
        raise AssertionError("No latent positions")

    record = {
        "prefix_ids": list(prefix_ids),
        "suffix_ids": list(suffix_ids),
        "explicit_ids": explicit_ids,
        "frontier_ends": list(plan.frontier_ends),
        "semantic_segment_count": len(plan.segment_latent_counts),
        "plan_metadata": metadata or {"segment_latent_counts": plan.segment_latent_counts},
    }
    return record

def materialize_encoder_cot(record, model):
    """Reconstruct encoder input exactly from a cached deterministic record."""
    explicit_ids = list(record["explicit_ids"])
    frontier_ends = list(record["frontier_ends"])
    inserted_ids = []
    start = 0
    for end in frontier_ends:
        if not start < end <= len(explicit_ids):
            raise ValueError((start, end, len(explicit_ids)))
        inserted_ids.extend(explicit_ids[start:end])
        inserted_ids.append(model.compress_token_id)
        start = end
    if start != len(explicit_ids):
        raise ValueError((start, len(explicit_ids)))
    return (
        list(record["prefix_ids"])
        + model.latent_token_ids[0]
        + inserted_ids
        + model.latent_token_ids[1]
    )

def materialize_stage1_record(record, model, supervision_frontier=None):
    """Apply the dynamic random frontier to cached deterministic content."""
    prefix_ids = list(record["prefix_ids"])
    suffix_ids = list(record["suffix_ids"])
    explicit_ids = list(record["explicit_ids"])
    frontier_ends = list(record["frontier_ends"])
    count = len(frontier_ends)
    if count <= 0:
        raise AssertionError("No latent positions")

    if supervision_frontier is None:
        supervision_frontier = int(torch.randint(1, count + 1, (1,)).item())
    if not 1 <= supervision_frontier <= count:
        raise ValueError(
            f"supervision_frontier must be in [1, {count}], got {supervision_frontier}"
        )

    # The explicit suffix starts at the real variable-size chunk boundary.
    explicit_suffix_start = frontier_ends[supervision_frontier - 1]
    explicit_suffix_ids = explicit_ids[explicit_suffix_start:]
    decoder_latent_ids = [-100] * supervision_frontier

    cot_ids = materialize_encoder_cot(record, model)
    decoder_prefix_ids = (
        prefix_ids + model.latent_token_ids[0] + decoder_latent_ids
    )
    explicit_budget = (
        LATENT_SFT_MAX_LENGTH
        - len(decoder_prefix_ids)
        - len(model.latent_token_ids[1])
        - len(suffix_ids)
    )
    if explicit_budget < 0:
        raise ValueError("Prompt, latent prefix and answer exceed the length limit")
    explicit_suffix_ids = head_tail_truncate(explicit_suffix_ids, explicit_budget)
    decoder_target_ids = explicit_suffix_ids + model.latent_token_ids[1] + suffix_ids

    input_ids = decoder_prefix_ids + decoder_target_ids
    return {
        "input_ids": input_ids,
        "labels": [-100] * len(decoder_prefix_ids) + decoder_target_ids,
        "cot_ids": cot_ids,
        "position_ids": list(range(len(input_ids))),
        "supervision_frontier": supervision_frontier,
        "latent_count": count,
    }

def pretrain_tokenize_function(
    example,
    model,
    compression_rate,
    use,
    supervision_frontier=None,
):
    record = prepare_stage1_record(example, model, compression_rate, use)
    return materialize_stage1_record(record, model, supervision_frontier)

def prepare_encoder_cot(example, model, encoder_model_path, compression_rate, use):
    """Shared soft-label preparation; returns encoder ids and the exact plan."""
    record = prepare_stage1_record(example, model, compression_rate, use)
    cot_ids = materialize_encoder_cot(record, model)
    # Reconstruct the return contract from the SAME plan, never run the parser
    # or splitter a second time for soft-label export.
    counts = record["plan_metadata"]["segment_latent_counts"]
    ends = record["frontier_ends"]
    sizes = [b - a for a, b in zip([0] + ends[:-1], ends)]
    grouped, cursor = [], 0
    for count in counts:
        grouped.append(sizes[cursor:cursor + count])
        cursor += count
    inserted_start = len(record["prefix_ids"]) + len(model.latent_token_ids[0])
    inserted = cot_ids[inserted_start:inserted_start + len(record["explicit_ids"]) + len(ends)]
    plan = CompressionPlan(inserted, ends, counts, grouped)
    return cot_ids, plan

__all__ = [
    "Stage1Dataset",
    "materialize_encoder_cot",
    "materialize_stage1_record",
    "pretrain_tokenize_function",
    "prepare_stage1_record",
    "prepare_encoder_cot",
]
