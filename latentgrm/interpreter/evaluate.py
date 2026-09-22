#!/usr/bin/env python
"""Strict evaluation for the retrospective decoder.

Full teacher-forced NLL is retained for comparability, but it is not treated as
the main result: after a few gold target tokens the decoder can lean on target
history. We therefore also report early-token and compression-boundary NLL,
paired information gain over real prompt-only and held-out shuffled-latent
controls, and short free generations with no target tokens supplied.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
from peft import PeftConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .train import (
    ROOT,
    BackwardCollator,
    BackwardPrefixDataset,
    ShardedLatentStore,
    build_split_indices,
    limited,
    load_json,
    load_interpreter,
)


MODES = ("correct", "prompt_only", "shuffled", "reverse", "uniform_topk", "zero")
SCOPES = ("first_1", "first_8", "first_32", "first_64", "chunk_boundaries", "all")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(ROOT / "configs/interpreter.json"))
    parser.add_argument("--base-model", help="Override the pretrained base directory saved with the adapter.")
    parser.add_argument("--split-file", default=str(ROOT / "outputs/interpreter/split.json"))
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dynamic-latents",
        help="Merged checkpoint-generated Top-K trace (.pt); use it instead of cached teacher labels.",
    )
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument(
        "--input-mode", choices=("latent", "prompt_only"), default="latent",
        help="Use real latent slots or a physically compact prompt-only sequence.",
    )
    parser.add_argument("--max-records", type=int, default=256)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--free-generation-records", type=int, default=16)
    parser.add_argument(
        "--teacher-forced-modes", nargs="+", choices=MODES, default=MODES,
        help="Run only the requested teacher-forced controls.",
    )
    parser.add_argument(
        "--free-generation-modes", nargs="+",
        choices=MODES,
        default=("correct", "prompt_only", "shuffled", "zero"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--generation-limit-policy", choices=("fixed", "reference"), default="reference"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--latent-order", choices=["forward", "reverse"], default="forward"
    )
    parser.add_argument("--soft-mode", choices=["weighted", "top1", "uniform", "temperature"], default="weighted")
    parser.add_argument("--soft-temperature", type=float, default=1.0)
    return parser.parse_args()


class DynamicLatentStore:
    """Overlay generated latent traces on the immutable Stage-1 target cache."""

    def __init__(self, static_store, trace_path):
        self.static = static_store
        payload = torch.load(trace_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "latent-trajectory":
            raise ValueError(f"unexpected dynamic trace schema: {payload.get('schema')}")
        self.records = {int(k): v for k, v in payload["records"].items()}
        self.num_records = static_store.num_records
        self.manifest = dict(static_store.manifest)

    def get(self, index):
        base = self.static.get(index)
        trace = self.records.get(int(index))
        if trace is None:
            raise KeyError(f"dynamic trace does not contain record {index}")
        probs = trace["probs"].float()
        token_ids = trace["token_ids"].long()
        if probs.ndim != 2 or probs.shape != token_ids.shape or not len(probs):
            raise ValueError(f"invalid/empty dynamic trace for record {index}: {tuple(probs.shape)}")
        # Dynamic latent steps do not have gold criterion boundaries.  Evaluation
        # uses the complete generated chain (views=1), so map its last step to the
        # complete explicit target and only use intermediate quantiles for the
        # auxiliary boundary-NLL bookkeeping.
        target_len = len(base["explicit_ids"])
        count = len(probs)
        frontiers = [max(1, math.ceil(target_len * (i + 1) / count)) for i in range(count)]
        frontiers[-1] = target_len
        base.update({
            "frontier_ends": frontiers,
            "latent_probs": probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12),
            "latent_token_ids": token_ids,
        })
        return base


def empty_scope():
    return {name: {"loss_sum": 0.0, "tokens": 0} for name in SCOPES}


def finish_scope(values):
    result = {}
    for name, item in values.items():
        nll = item["loss_sum"] / max(item["tokens"], 1)
        result[name] = {
            "nll": nll,
            "perplexity": math.exp(min(nll, 20.0)),
            "tokens": item["tokens"],
        }
    return result


def compact_prompt_only(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Physically remove latent slots to create a true prompt-only sequence.

    Masking latent keys alone is not enough: with causal-LM label shifting, the
    final masked placeholder would still predict the first target token.
    """
    keep_rows = []
    lengths = []
    for row in range(batch["input_ids"].shape[0]):
        keep = batch["attention_mask"][row].bool().clone()
        positions = batch["latent_positions"][row]
        positions = positions[positions.ge(0)]
        keep[positions] = False
        indices = torch.nonzero(keep, as_tuple=False).flatten()
        keep_rows.append(indices)
        lengths.append(int(indices.numel()))

    width = max(lengths)
    compact = dict(batch)
    compact["input_ids"] = batch["input_ids"].new_zeros(
        (batch["input_ids"].shape[0], width)
    )
    compact["labels"] = batch["labels"].new_full(
        (batch["labels"].shape[0], width), -100
    )
    compact["attention_mask"] = batch["attention_mask"].new_zeros(
        (batch["attention_mask"].shape[0], width)
    )
    compact["position_ids"] = batch["position_ids"].new_zeros(
        (batch["position_ids"].shape[0], width)
    )
    compact["latent_positions"] = batch["latent_positions"].new_full(
        batch["latent_positions"].shape, -1
    )
    for row, (indices, length) in enumerate(zip(keep_rows, lengths)):
        compact["input_ids"][row, :length] = batch["input_ids"][row, indices]
        compact["labels"][row, :length] = batch["labels"][row, indices]
        compact["attention_mask"][row, :length] = 1
        compact["position_ids"][row, :length] = torch.arange(
            length, device=batch["position_ids"].device
        )
    return compact


@torch.inference_mode()
def evaluate_mode(model, loader, device, mode):
    totals = empty_scope()
    example_losses = []
    for cpu_batch in loader:
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        effective_mode = mode
        if mode == "prompt_only":
            batch = compact_prompt_only(batch)
            effective_mode = "correct"
        labels = batch["labels"]
        outputs = model(**batch, latent_mode=effective_mode)
        logits = outputs.logits[:, :-1].float()
        targets = labels[:, 1:]
        losses = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), targets, ignore_index=-100, reduction="none"
        )
        valid = targets.ne(-100)
        for row in range(labels.shape[0]):
            ordered = torch.nonzero(valid[row], as_tuple=False).flatten()
            row_losses = losses[row, ordered]
            example_losses.append(float(row_losses.mean()))
            for width in (1, 8, 32, 64):
                selected = row_losses[:width]
                totals[f"first_{width}"]["loss_sum"] += float(selected.sum())
                totals[f"first_{width}"]["tokens"] += selected.numel()
            totals["all"]["loss_sum"] += float(row_losses.sum())
            totals["all"]["tokens"] += row_losses.numel()

            target_start = int(torch.nonzero(labels[row].ne(-100), as_tuple=False)[0])
            starts = batch["target_chunk_starts"][row]
            starts = starts[starts.ge(0)]
            boundary_losses = losses[row, target_start - 1 + starts]
            totals["chunk_boundaries"]["loss_sum"] += float(boundary_losses.sum())
            totals["chunk_boundaries"]["tokens"] += boundary_losses.numel()
    return {
        "scopes": finish_scope(totals),
        "examples": len(example_losses),
        "example_nll_mean": sum(example_losses) / max(len(example_losses), 1),
    }, example_losses


def lcs_length(a: list[int], b: list[int]) -> int:
    previous = [0] * (len(b) + 1)
    for left in a:
        current = [0]
        for j, right in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if left == right else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def token_f1(prediction: list[int], target: list[int]) -> float:
    overlap = sum((Counter(prediction) & Counter(target)).values())
    if not prediction or not target:
        return float(prediction == target)
    precision = overlap / len(prediction)
    recall = overlap / len(target)
    return 2 * precision * recall / max(precision + recall, 1e-12)


@torch.inference_mode()
def free_generate(
    model, dataset, collator, tokenizer, device, modes, maximum,
    max_new_tokens, generation_limit_policy,
):
    aggregate = {
        mode: {
            "prefix_match": 0,
            "aligned_correct": 0,
            "aligned_total": 0,
            "token_f1": 0.0,
            "rouge_l_f1": 0.0,
            "exact": 0,
            "examples": 0,
            "generated_target_tokens": 0,
            "full_target_tokens": 0,
            "actual_generated_tokens": 0,
        }
        for mode in modes
    }
    samples = []
    for item_index in range(min(maximum, len(dataset))):
        cpu_batch = collator([dataset[item_index]])
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        target_start = int(torch.nonzero(batch["labels"][0].ne(-100), as_tuple=False)[0])
        # The collator may leave ignored/padded label positions after the first
        # supervised token.  Tokenizers cannot decode the -100 ignore index, so
        # retain only real vocabulary ids for generation metrics.
        raw_reference = batch["labels"][0, target_start:]
        valid_reference = raw_reference[
            raw_reference.ge(0) & raw_reference.lt(len(tokenizer))
        ]
        full_reference = valid_reference.tolist()
        reference = full_reference[:max_new_tokens]
        cap = (
            max_new_tokens
            if generation_limit_policy == "fixed"
            else min(max_new_tokens, len(full_reference))
        )
        sample = {
            "dataset_item": item_index,
            "record_index": int(dataset.indices[item_index // dataset.views_per_record]),
            "view": item_index % dataset.views_per_record,
            "target_tokens": len(full_reference),
            "evaluated_reference_tokens": len(reference),
            "target": tokenizer.decode(reference),
        }
        for mode in modes:
            mode_batch = compact_prompt_only(batch) if mode == "prompt_only" else batch
            effective_mode = "correct" if mode == "prompt_only" else mode
            mode_target_start = int(
                torch.nonzero(
                    mode_batch["labels"][0].ne(-100), as_tuple=False
                )[0]
            )
            inputs_embeds, attention_mask, position_ids = model.prepare_inputs(
                input_ids=mode_batch["input_ids"][:, :mode_target_start],
                attention_mask=mode_batch["attention_mask"][:, :mode_target_start],
                position_ids=mode_batch["position_ids"][:, :mode_target_start],
                latent_positions=mode_batch["latent_positions"],
                latent_token_ids=mode_batch["latent_token_ids"],
                latent_probs=mode_batch["latent_probs"],
                latent_mode=effective_mode,
                shuffled_latent_token_ids=mode_batch.get("shuffled_latent_token_ids"),
                shuffled_latent_probs=mode_batch.get("shuffled_latent_probs"),
            )
            generated = model.decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                do_sample=False,
                max_new_tokens=cap,
                eos_token_id=dataset.right_ids[0],
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )[0].tolist()
            prediction = generated[-cap:]
            prefix_match = 0
            for predicted, gold in zip(prediction, reference):
                if predicted != gold:
                    break
                prefix_match += 1
            aligned = sum(a == b for a, b in zip(prediction, reference))
            lcs = lcs_length(prediction, reference)
            rouge_l = 2 * lcs / max(len(prediction) + len(reference), 1)
            target = aggregate[mode]
            target["prefix_match"] += prefix_match
            target["aligned_correct"] += aligned
            target["aligned_total"] += len(reference)
            target["token_f1"] += token_f1(prediction, reference)
            target["rouge_l_f1"] += rouge_l
            target["exact"] += int(prediction == reference)
            target["examples"] += 1
            target["generated_target_tokens"] += min(cap, len(full_reference))
            target["full_target_tokens"] += len(full_reference)
            target["actual_generated_tokens"] += len(prediction)
            sample[mode] = tokenizer.decode(prediction)
            sample[f"{mode}_tokens"] = len(prediction)
        samples.append(sample)
    for values in aggregate.values():
        count = max(values["examples"], 1)
        values["mean_exact_prefix_tokens"] = values.pop("prefix_match") / count
        values["aligned_token_accuracy"] = values.pop("aligned_correct") / max(values.pop("aligned_total"), 1)
        values["token_f1"] /= count
        values["rouge_l_f1"] /= count
        values["capped_exact_match"] = values.pop("exact") / count
        values["mean_generation_length_over_reference"] = values.pop(
            "actual_generated_tokens"
        ) / max(values["full_target_tokens"], 1)
        values["mean_target_coverage"] = values.pop("generated_target_tokens") / max(
            values.pop("full_target_tokens"), 1
        )
    return aggregate, samples


def main():
    args = parse_args()
    paths = load_json(args.paths)
    paths["base_model"] = (
        args.base_model or PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(paths["tokenizer"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    static_store = ShardedLatentStore(paths["stage1_cache"], paths["latent_soft_labels"])
    store = (
        DynamicLatentStore(static_store, args.dynamic_latents)
        if args.dynamic_latents else static_store
    )
    splits = build_split_indices(store, Path(args.split_file), args.seed)
    indices = limited(splits[args.split], args.max_records, args.seed + 17)
    dataset = BackwardPrefixDataset(
        store,
        indices,
        tokenizer,
        views_per_record=args.views,
        seed=args.seed,
        max_length=6144,
        validation_quantiles=True,
        # Prompt-only has zero latent slots; constructing a shuffled latent
        # donor is both meaningless and shape-incompatible with its collator.
        include_shuffled_control=args.input_mode == "latent",
        latent_order=args.latent_order,
        soft_mode=args.soft_mode,
        soft_temperature=args.soft_temperature,
        input_mode=args.input_mode,
    )
    model = load_interpreter(paths, tokenizer, args, adapter=args.adapter)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    collator = BackwardCollator(tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
    )
    results = {
        "schema": "interpreter-evaluation",
        "split": args.split,
        "records": len(indices),
        "views": args.views,
        "adapter": args.adapter,
        "dynamic_latents": args.dynamic_latents,
        "input_mode": args.input_mode,
        "latent_order": args.latent_order,
        "teacher_forced": {},
    }
    losses = {}
    for mode in args.teacher_forced_modes:
        metrics, mode_losses = evaluate_mode(model, loader, device, mode)
        results["teacher_forced"][mode] = metrics
        losses[mode] = mode_losses
        print(mode, json.dumps(metrics), flush=True)
    if "correct" in results["teacher_forced"]:
        correct = results["teacher_forced"]["correct"]["scopes"]
        control_modes = [
            mode for mode in args.teacher_forced_modes if mode != "correct"
        ]
        results["information_gain_over_correct"] = {
            mode: {
                scope: results["teacher_forced"][mode]["scopes"][scope]["nll"]
                - correct[scope]["nll"]
                for scope in SCOPES
            }
            for mode in control_modes
        }
        results["paired_fraction_control_worse"] = {
            mode: sum(
                control > good
                for good, control in zip(losses["correct"], losses[mode])
            ) / max(len(losses["correct"]), 1)
            for mode in control_modes
        }
    # Persist the expensive teacher-forced controls before free generation so
    # a generation-only failure never discards the completed measurements.
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    free_metrics, samples = free_generate(
        model,
        dataset,
        collator,
        tokenizer,
        device,
        tuple(args.free_generation_modes),
        args.free_generation_records,
        args.max_new_tokens,
        args.generation_limit_policy,
    )
    results["free_generation"] = free_metrics
    results["free_generation_samples"] = samples
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
