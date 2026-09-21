#!/usr/bin/env python
"""Evaluate the Stage-2 latent OpenRubric judge.

The evaluator accepts either Latent-SFT JSONL rows (problem/cot_answer) or an
OpenRubric fixed-rubric RewardBench cache.  Under torchrun, every process owns
one GPU and evaluates a deterministic strided shard.  Per-rank progress files
make interrupted runs resumable and are merged atomically by rank zero.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

try:
    from .efficiency_metrics import (
        official_subset_name,
        request_latency_seconds,
        summarize_efficiency,
    )
except ImportError:  # Direct script execution.
    from efficiency_metrics import (
        official_subset_name,
        request_latency_seconds,
        summarize_efficiency,
    )

# RewardBench v1 leaderboard aggregation constants.  These intentionally match
# allenai/reward-bench's EXAMPLE_COUNTS and SUBSET_MAPPING.  math-prm is
# upweighted from its 447 prompts to 984 so that math and code have equal weight
# inside the Reasoning section.
REWARDBENCH_EXAMPLE_COUNTS = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 984,
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 154,
    "xstest-should-respond": 250,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}
# Actual rows in each direction. Keep this separate from leaderboard weights:
# math-prm has 447 prompts but is intentionally weighted as 984 above.
REWARDBENCH_SAMPLE_COUNTS = {
    **REWARDBENCH_EXAMPLE_COUNTS,
    "math-prm": 447,
}
REWARDBENCH_SUBSET_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}
SPECIALIZED_SUMMARY_SUBSETS = {"Precise IF", "Focus", "ppe-ifeval"}
SPECIALIZED_SUMMARY_PREFIXES = ("rm-bench-", "helpsteer3-")


EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from latentgrm.modeling.modeling_stage2 import LatentSFTStage2SoftEmbedding
from latentgrm.openrubric_utils import (
    OPENRUBRIC_JUDGE_PREFIX,
    OPENRUBRIC_TASK_TYPE,
    TASK_MARKER,
    parse_binary_label,
)
from latentgrm.prompting import (
    LATENT_SFT_MAX_LENGTH,
    build_qwen_generation_prefix,
    head_tail_truncate,
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise TypeError(
                    f"Evaluation row at {path}:{line_number} must be an object"
                )
            rows.append(row)
    return rows


def normalize_external_label(value: object) -> str | None:
    """Normalize labels used by fixed-rubric RewardBench caches."""
    if not isinstance(value, str):
        return None
    aliases = {
        "Response A": "Response A",
        "Response B": "Response B",
        "response_a": "Response A",
        "response_b": "Response B",
    }
    return aliases.get(value.strip())


def normalize_eval_record(record: dict, eval_index: int) -> dict:
    """Accept either Latent-SFT rows or OpenRubric fixed-rubric cache rows."""
    if not isinstance(record, dict):
        raise TypeError(f"Evaluation row {eval_index} must be a JSON object")

    if "problem" in record or "cot_answer" in record:
        problem = record.get("problem")
        expected = parse_binary_label(record.get("cot_answer", ""))
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"Evaluation row {eval_index} has an invalid problem")
        if expected is None:
            raise ValueError(f"Evaluation row {eval_index} has an invalid cot_answer")
    else:
        required = ("instruction", "rubric", "response_a", "response_b", "label")
        missing = [field for field in required if record.get(field) in (None, "")]
        if missing:
            raise ValueError(
                f"Evaluation row {eval_index} is missing fields: {missing}"
            )
        expected = normalize_external_label(record["label"])
        if expected is None:
            raise ValueError(
                f"Evaluation row {eval_index} has an invalid label: {record['label']!r}"
            )
        task = (
            f"Instruction:\n{str(record['instruction']).strip()}\n"
            f"Rubric:\n{str(record['rubric']).strip()}\n"
            f"Response A:\n{str(record['response_a']).strip()}\n"
            f"Response B:\n{str(record['response_b']).strip()}"
        )
        problem = f"{OPENRUBRIC_JUDGE_PREFIX}\n\n{TASK_MARKER}\n{task}"

    return {
        "eval_index": eval_index,
        "source_index": record.get("source_index", record.get("dataset_index")),
        "problem": problem.strip(),
        "cot_answer": expected,
        "task_type": OPENRUBRIC_TASK_TYPE,
        "subset": str(record.get("subset") or "unknown"),
        "official_subset": official_subset_name(record),
        "exchange": bool(record.get("exchange", False)),
    }


def _latent_token_count(row: dict) -> int | None:
    latent = row.get("latent_tokens")
    answer = row.get("answer_tokens")
    if not isinstance(latent, (int, float)) or not isinstance(answer, (int, float)):
        return None
    return int(latent) + int(answer)


def _bucket_metrics(rows: list[dict], field: str) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        value = (
            row.get("official_subset") or official_subset_name(row)
            if field == "subset"
            else row.get(field, "unknown")
        )
        key = str(value).lower() if isinstance(value, bool) else str(value)
        buckets[key].append(row)
    result = {}
    for key, values in sorted(buckets.items()):
        metrics = {
            "samples": len(values),
            "correct": sum(bool(value["correct"]) for value in values),
            "accuracy": (sum(bool(value["correct"]) for value in values) / len(values)),
            "invalid_outputs": sum(value.get("prediction") is None for value in values),
        }
        metrics.update(summarize_efficiency(values, _latent_token_count))
        result[key] = metrics
    return result


def _rewardbench_scores(rows: list[dict]) -> dict:
    """Aggregate RewardBench v1 subsets with the official leaderboard weights."""
    subset_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        subset = str(row.get("subset", "unknown"))
        if subset in REWARDBENCH_EXAMPLE_COUNTS:
            subset_rows[subset].append(row)

    subset_scores = {
        subset: sum(bool(row["correct"]) for row in values) / len(values)
        for subset, values in sorted(subset_rows.items())
    }
    sections = {}
    for section, expected_subsets in REWARDBENCH_SUBSET_MAPPING.items():
        present = [subset for subset in expected_subsets if subset in subset_scores]
        if not present:
            continue
        official_weight = sum(REWARDBENCH_EXAMPLE_COUNTS[subset] for subset in present)
        weighted_correct = sum(
            subset_scores[subset] * REWARDBENCH_EXAMPLE_COUNTS[subset]
            for subset in present
        )
        score = weighted_correct / official_weight if official_weight else 0.0
        sections[section] = {
            "score": score,
            "score_percent": score * 100,
            "evaluated_samples": sum(len(subset_rows[subset]) for subset in present),
            "official_weight": official_weight,
            "missing_subsets": [
                subset for subset in expected_subsets if subset not in subset_scores
            ],
        }

    section_values = [section["score"] for section in sections.values()]
    expected = set(REWARDBENCH_EXAMPLE_COUNTS)
    complete = set(subset_scores) == expected and all(
        len(subset_rows[subset]) == REWARDBENCH_SAMPLE_COUNTS[subset]
        for subset in expected
    )
    return {
        "samples": sum(len(values) for values in subset_rows.values()),
        "complete": complete,
        "subsets": {
            subset: {
                "samples": len(subset_rows[subset]),
                "accuracy": score,
                "accuracy_percent": score * 100,
            }
            for subset, score in subset_scores.items()
        },
        "sections": sections,
        "score": sum(section_values) / len(section_values),
        "score_percent": 100 * sum(section_values) / len(section_values),
    }


def _rewardbench_summary(rows: list[dict]) -> dict | None:
    known_rows = [
        row
        for row in rows
        if str(row.get("subset", "unknown")) in REWARDBENCH_EXAMPLE_COUNTS
    ]
    if not known_rows:
        return None

    forward_rows = [row for row in known_rows if not bool(row.get("exchange", False))]
    reverse_rows = [row for row in known_rows if bool(row.get("exchange", False))]
    forward = _rewardbench_scores(forward_rows) if forward_rows else None
    reverse = _rewardbench_scores(reverse_rows) if reverse_rows else None

    def compact(scores: dict | None) -> dict | None:
        if scores is None:
            return None
        result = {
            section: {
                "score_percent": scores["sections"][section]["score_percent"],
            }
            for section in REWARDBENCH_SUBSET_MAPPING
            if section in scores["sections"]
        }
        result["Score"] = {
            "score_percent": scores["score_percent"],
        }
        return result

    if forward is not None and reverse is not None:
        present_sections = [
            section
            for section in REWARDBENCH_SUBSET_MAPPING
            if section in forward["sections"] or section in reverse["sections"]
        ]
        average = {
            "sections": {
                section: {
                    "score": sum(
                        scores["sections"][section]["score"]
                        for scores in (forward, reverse)
                        if section in scores["sections"]
                    )
                    / sum(
                        section in scores["sections"]
                        for scores in (forward, reverse)
                    ),
                }
                for section in present_sections
            },
        }
        for section in average["sections"].values():
            section["score_percent"] = section["score"] * 100
        average["score"] = sum(
            section["score"] for section in average["sections"].values()
        ) / len(average["sections"])
        average["score_percent"] = average["score"] * 100
    else:
        average = forward or reverse

    # Keep this order: the position average first, then original and reversed.
    return {
        "average": compact(average),
        "forward": compact(forward),
        "reverse": compact(reverse),
    }


def summarize_results(
    rows: list[dict],
    elapsed_seconds: float | None = None,
    generation_only_seconds: float | None = None,
) -> dict:
    if not rows:
        raise ValueError("Cannot summarize empty evaluation results")
    correct = sum(bool(row["correct"]) for row in rows)
    micro_accuracy = correct / len(rows)
    invalid = sum(row.get("prediction") is None for row in rows)
    valid = len(rows) - invalid
    latent_lengths = [int(row["latent_tokens"]) for row in rows]
    answer_lengths = [int(row["answer_tokens"]) for row in rows]
    latent_end_count = sum(bool(row["reached_latent_end"]) for row in rows)
    rewardbench = _rewardbench_summary(rows)
    summary = {"rewardbench": rewardbench} if rewardbench is not None else {}
    summary.update(
        {
            "samples": len(rows),
            "correct": correct,
            "accuracy": micro_accuracy,
            "invalid_outputs": invalid,
            "invalid_output_rate": invalid / len(rows),
            "valid_output_accuracy": correct / valid if valid else 0.0,
            "latent_end_reached": latent_end_count,
            "latent_end_reached_rate": latent_end_count / len(rows),
            "average_latent_tokens": sum(latent_lengths) / len(latent_lengths),
            "maximum_latent_tokens": max(latent_lengths),
            "average_answer_tokens": sum(answer_lengths) / len(answer_lengths),
            "labels": _bucket_metrics(rows, "expected"),
            "subsets": _bucket_metrics(rows, "subset"),
            "exchange": _bucket_metrics(rows, "exchange"),
        }
    )
    summary.update(summarize_efficiency(rows, _latent_token_count))
    specialized_summary_expected = any(
        str(row.get("subset", "")) in SPECIALIZED_SUMMARY_SUBSETS
        or str(row.get("subset", "")).startswith(SPECIALIZED_SUMMARY_PREFIXES)
        for row in rows
    )
    if rewardbench is not None or specialized_summary_expected:
        # Every supported benchmark has a benchmark-specific headline metric.
        # Do not expose the all-row micro accuracy as a competing top-level
        # score, including in the intermediate summary printed before an
        # external specialized summarizer runs.
        summary.pop("accuracy")
    if elapsed_seconds is not None:
        summary["elapsed_seconds"] = elapsed_seconds
        summary["average_elapsed_seconds_per_sample"] = (
            elapsed_seconds / len(rows)
        )
        summary["samples_per_second"] = (
            len(rows) / elapsed_seconds if elapsed_seconds > 0 else 0.0
        )
    if generation_only_seconds is not None:
        summary["timing_mode"] = (
            "parallel_vote_requests_llm_generate_wall_excluding_model_load"
        )
        summary["generation_only_seconds"] = generation_only_seconds
        summary["generation_only_seconds_per_sample"] = (
            generation_only_seconds / len(rows)
        )
        summary["generation_only_samples_per_second"] = (
            len(rows) / generation_only_seconds
            if generation_only_seconds > 0 else 0.0
        )
    rollouts = [rollout for row in rows for rollout in row.get("vote_rollouts", [])]
    if rollouts:
        rollout_total_tokens = [int(rollout["total_tokens"]) for rollout in rollouts]
        rollout_latent_tokens = [int(rollout["latent_tokens"]) for rollout in rollouts]
        rollout_answer_tokens = [int(rollout["answer_tokens"]) for rollout in rollouts]
        per_sample_total_tokens = [
            sum(int(rollout["total_tokens"]) for rollout in row["vote_rollouts"])
            for row in rows
        ]
        summary["output_token_count"] = {
            "per_rollout_total": sum(rollout_total_tokens),
            "per_rollout_mean": sum(rollout_total_tokens) / len(rollout_total_tokens),
            "per_rollout_max": max(rollout_total_tokens),
            "per_rollout_min": min(rollout_total_tokens),
            "per_sample_total": sum(per_sample_total_tokens),
            "per_sample_mean": sum(per_sample_total_tokens) / len(per_sample_total_tokens),
            "latent_per_rollout_total": sum(rollout_latent_tokens),
            "latent_per_rollout_mean": sum(rollout_latent_tokens) / len(rollout_latent_tokens),
            "answer_per_rollout_total": sum(rollout_answer_tokens),
            "answer_per_rollout_mean": sum(rollout_answer_tokens) / len(rollout_answer_tokens),
        }
        rollout_correct = sum(bool(rollout["correct"]) for rollout in rollouts)
        rollout_valid = sum(
            rollout.get("prediction") is not None for rollout in rollouts
        )
        all_valid_unanimous = sum(
            row.get("vote_valid_count") == row.get("vote_num")
            and max(row.get("vote_counts", {}).values(), default=0)
            == row.get("vote_num")
            for row in rows
        )
        mean_rollout_accuracy = rollout_correct / len(rollouts)
        summary["vote"] = {
            "num_votes": rows[0].get("vote_num", 1),
            "rollout_samples": len(rollouts),
            "mean_single_rollout_accuracy": mean_rollout_accuracy,
            "mean_single_rollout_valid_accuracy": (
                rollout_correct / rollout_valid if rollout_valid else 0.0
            ),
            "single_rollout_invalid_output_rate": (1 - rollout_valid / len(rollouts)),
            "single_rollout_latent_end_reached_rate": (
                sum(bool(rollout["reached_latent_end"]) for rollout in rollouts)
                / len(rollouts)
            ),
            "all_valid_unanimous_rate": all_valid_unanimous / len(rows),
            "tie_rate": sum(bool(row.get("vote_tied")) for row in rows) / len(rows),
            "accuracy_gain_vs_mean_single_rollout": (
                micro_accuracy - mean_rollout_accuracy
            ),
        }
    return summary


def evaluation_signature(args: argparse.Namespace, data_path: Path) -> str:
    stat = data_path.stat()
    payload = {
        "data_path": str(data_path.resolve()),
        "data_size": stat.st_size,
        "data_mtime_ns": stat.st_mtime_ns,
        "model_path": str(Path(args.model_path).resolve()),
        "lora_path": (str(Path(args.lora_path).resolve()) if args.lora_path else None),
        "max_samples": args.max_samples,
        "include_subsets": sorted(
            parse_subset_filter(getattr(args, "include_subsets", None)) or ()
        ),
        "max_total_length": args.max_total_length,
        "max_latent_tokens": args.max_latent_tokens,
        "max_answer_tokens": args.max_answer_tokens,
        "topk_interpolation": args.topk_interpolation,
        "add_gumbel_noise": args.add_gumbel_noise,
        "gumbel_temperature": args.gumbel_temperature,
        "noise_scale": args.noise_scale,
        "gumbel_seed": args.gumbel_seed,
        "vote": args.vote,
        "backend": args.backend,
        "tensor_parallel_size": args.tensor_parallel_size,
        "active_rollouts": args.vllm_target_active_rollouts,
        "model_files": [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                        for p in sorted(Path(args.model_path).glob("*"))
                        if p.is_file() and p.suffix in {".json", ".bin", ".safetensors"}],
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_completed(paths: list[Path], signature: str) -> dict[int, dict]:
    completed: dict[int, dict] = {}
    for path in paths:
        if not path.is_file():
            continue
        from latentgrm.evaluation.progress import read_complete_rows
        for row in read_complete_rows(path):
            if row.get("evaluation_id") != signature:
                raise ValueError(
                    f"Existing result belongs to another evaluation: {path}"
                )
            index = row.get("eval_index")
            if not isinstance(index, int):
                raise TypeError(f"Result has invalid eval_index in {path}")
            previous = completed.get(index)
            if previous is not None and previous != row:
                raise ValueError(f"Conflicting resumed result for eval_index={index}")
            completed[index] = row
    return completed


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _all_part_paths(output_path: Path) -> list[Path]:
    return sorted(output_path.parent.glob(output_path.name + ".rank*.part.jsonl"))


def distributed_runtime(device_arg: str | None) -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed evaluation requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, torch.device(f"cuda:{local_rank}")
    device_name = device_arg or ("cuda:0" if torch.cuda.is_available() else "cpu")
    return 0, 1, torch.device(device_name)


def prepare_inputs(
    example: dict,
    model,
    device: torch.device,
    max_total_length: int,
    max_latent_tokens: int,
    max_answer_tokens: int,
) -> dict[str, torch.Tensor]:
    prefix = build_qwen_generation_prefix(model.tokenizer, example)
    prefix_ids = model.tokenizer(
        prefix,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    reserved = (
        len(model.latent_token_ids[0])
        + max_latent_tokens
        + len(model.latent_token_ids[1])
        + max_answer_tokens
    )
    prefix_budget = max_total_length - reserved
    if prefix_budget < 0:
        raise ValueError(
            "Generation limits exceed --max_total_length: "
            f"reserved={reserved}, max_total_length={max_total_length}"
        )
    prefix_ids = head_tail_truncate(prefix_ids, prefix_budget)
    input_ids = prefix_ids + model.latent_token_ids[0]
    tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    return {
        "input_ids": tensor,
        "attention_mask": torch.ones_like(tensor),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an OpenRubric Latent-SFT Stage-2 model."
    )
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument(
        "--model_path",
        "--base_model_path",
        dest="model_path",
        required=True,
        help="Merged Stage-2 HF model, or its base model when --lora_path is used.",
    )
    parser.add_argument("--lora_path")
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument(
        "--include-subsets",
        "--include_subsets",
        dest="include_subsets",
        help="Comma-separated exact subset names to evaluate before max_samples.",
    )
    parser.add_argument("--max_total_length", type=int, default=LATENT_SFT_MAX_LENGTH)
    parser.add_argument("--max_latent_tokens", type=int, default=256)
    parser.add_argument("--max_answer_tokens", type=int, default=8)
    parser.add_argument("--topk_interpolation", type=int, default=10)
    parser.add_argument("--add_gumbel_noise", action="store_true")
    parser.add_argument("--gumbel_temperature", type=float, default=1.0)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--gumbel_seed", type=int, default=20260803)
    parser.add_argument(
        "--vote",
        type=int,
        default=1,
        help="Independent latent rollouts per example before majority voting.",
    )
    parser.add_argument("--device")
    parser.add_argument("--use_flash_attention_2", action="store_true")
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--vllm_target_active_rollouts",
        type=int,
        default=80,
        help="Target number of active native-n child rollouts for optimized vLLM.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_subset_filter(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    subsets = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not subsets:
        raise ValueError("--include-subsets must contain at least one subset name")
    return subsets


def select_eval_records(
    raw_data: list[dict],
    include_subsets: str | None,
    max_samples: int | None,
) -> list[dict]:
    """Normalize once, retain source indices, then filter and limit."""
    data = [
        normalize_eval_record(record, index)
        for index, record in enumerate(raw_data)
    ]
    requested = parse_subset_filter(include_subsets)
    if requested is not None:
        available = {str(example["subset"]) for example in data}
        missing = [subset for subset in requested if subset not in available]
        if missing:
            raise ValueError(
                "Requested evaluation subsets are absent from the data: "
                f"{missing}; available={sorted(available)}"
            )
        requested_set = set(requested)
        data = [
            example for example in data if str(example["subset"]) in requested_set
        ]
    if max_samples is not None:
        data = data[:max_samples]
    if not data:
        raise ValueError("Evaluation data is empty after subset filtering")
    return data


def _validate_args(args: argparse.Namespace) -> None:
    if args.topk_interpolation <= 0:
        raise ValueError("--topk_interpolation must be greater than 0")
    if args.gumbel_temperature <= 0:
        raise ValueError("--gumbel_temperature must be greater than 0")
    if args.noise_scale < 0:
        raise ValueError("--noise_scale must be non-negative")
    if args.max_latent_tokens <= 0 or args.max_answer_tokens <= 0:
        raise ValueError("Generation token limits must be greater than 0")
    if args.max_total_length <= 0:
        raise ValueError("--max_total_length must be greater than 0")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be greater than 0")
    parse_subset_filter(args.include_subsets)
    if args.vote <= 0:
        raise ValueError("--vote must be greater than 0")
    if not args.data_path.is_file():
        raise FileNotFoundError(f"Evaluation data does not exist: {args.data_path}")


def _coordinate_output_setup(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    part_paths: list[Path],
    summary_path: Path,
) -> None:
    error = [None]
    if rank == 0:
        try:
            args.output_path.parent.mkdir(parents=True, exist_ok=True)
            artifacts = [
                args.output_path,
                summary_path,
                *part_paths,
                *_all_part_paths(args.output_path),
                args.output_path.with_name(args.output_path.name + ".tmp"),
                summary_path.with_name(summary_path.name + ".tmp"),
            ]
            if args.overwrite:
                for path in artifacts:
                    path.unlink(missing_ok=True)
            elif not args.resume:
                existing = [str(path) for path in artifacts if path.exists()]
                if existing:
                    raise FileExistsError(
                        "Evaluation output already exists; use --resume or "
                        f"--overwrite: {existing}"
                    )
        # Rank zero must broadcast every setup failure or the other ranks can
        # block forever waiting at the following collective operation.
        except Exception as exc:  # noqa: BLE001
            error[0] = f"{type(exc).__name__}: {exc}"
    if world_size > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(error[0])
    if world_size > 1:
        dist.barrier()


def _result_row(
    example: dict,
    generated: dict,
    signature: str,
    rank: int,
) -> dict:
    prediction = parse_binary_label(generated["text"])
    expected = example["cot_answer"]
    result = {
        "evaluation_id": signature,
        "eval_index": example["eval_index"],
        "source_index": example.get("source_index"),
        "subset": example["subset"],
        "official_subset": example.get("official_subset")
        or official_subset_name(example),
        "exchange": example["exchange"],
        "expected": expected,
        "raw_prediction": generated["text"],
        "prediction": prediction,
        "correct": prediction == expected,
        "reached_latent_end": generated["stopped_early"],
        "latent_tokens": generated["generate_token_num"],
        "answer_tokens": generated["answer_token_num"],
        "total_tokens": (
            generated["generate_token_num"] + generated["answer_token_num"]
        ),
        "rank": rank,
    }
    generation_seconds = generated.get("generation_seconds")
    if isinstance(generation_seconds, (int, float)) and generation_seconds >= 0:
        result["generation_seconds"] = float(generation_seconds)
    return result


def _vote_result_row(
    example: dict,
    generated_rollouts: list[dict],
    signature: str,
    rank: int,
) -> dict:
    """Majority-vote independent latent rollouts for one judge example.

    Invalid generations do not cast a vote. Ties between Response A and
    Response B are resolved by the earliest valid rollout, so vote=1 exactly
    preserves legacy result fields and the aggregation is reproducible.
    """
    rollouts = [
        _result_row(example, generated, signature, rank)
        for generated in generated_rollouts
    ]
    if not rollouts:
        raise ValueError("At least one rollout is required for voting")

    valid_rollouts = [
        rollout for rollout in rollouts if rollout["prediction"] is not None
    ]
    counts = Counter(rollout["prediction"] for rollout in valid_rollouts)
    if valid_rollouts:
        highest_count = max(counts.values())
        winners = {label for label, count in counts.items() if count == highest_count}
        selected = next(
            rollout for rollout in valid_rollouts if rollout["prediction"] in winners
        )
        prediction = selected["prediction"]
    else:
        winners = set()
        selected = rollouts[0]
        prediction = None

    result = dict(selected)
    measured_rollout_seconds = [
        float(rollout["generation_seconds"])
        for rollout in rollouts
        if isinstance(rollout.get("generation_seconds"), (int, float))
        and rollout["generation_seconds"] >= 0
    ]
    result.update(
        {
            "prediction": prediction,
            "correct": prediction == example["cot_answer"],
            "vote_num": len(rollouts),
            "vote_valid_count": len(valid_rollouts),
            "vote_counts": {
                "Response A": counts.get("Response A", 0),
                "Response B": counts.get("Response B", 0),
            },
            "vote_tied": len(winners) > 1,
            "vote_selected_index": rollouts.index(selected),
            "vote_rollouts": [
                {
                    "raw_prediction": rollout["raw_prediction"],
                    "prediction": rollout["prediction"],
                    "correct": rollout["correct"],
                    "reached_latent_end": rollout["reached_latent_end"],
                    "latent_tokens": rollout["latent_tokens"],
                    "answer_tokens": rollout["answer_tokens"],
                    "total_tokens": rollout["total_tokens"],
                    **(
                        {"generation_seconds": rollout["generation_seconds"]}
                        if "generation_seconds" in rollout
                        else {}
                    ),
                }
                for rollout in rollouts
            ],
        }
    )
    if measured_rollout_seconds:
        # Votes are concurrent; completion latency is governed by the slowest
        # rollout rather than the sum of independent request latencies.
        result["generation_seconds"] = max(measured_rollout_seconds)
    return result


VLLM_CHUNK = 128


def generate_with_vllm(
    args: argparse.Namespace,
    examples: list[dict],
    num_data_rows: int,
    rank: int,
) -> tuple[dict[tuple[int, int], dict], float]:
    """Generate latent rollouts for ``examples`` with vLLM (latent-port).

    Returns ``{(eval_index, vote_index): generated_dict}``. Only the visible
    answer segment (after the latent-end token ``</think>``) is returned as
    ``text`` so the existing binary-label parser keeps working. Per-vote
    seeds reproduce the HF backend's Gumbel seed schedule.
    """
    # vLLM's EngineCore subprocess must not fork a CUDA-initialized process
    # (the parent eval process initializes torch CUDA in distributed_runtime).
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # Explicitly use the V2 model runner (vLLM 0.26 default, same path as the
    # baseline RubricRM evals). Latent-SFT is implemented on the V2 runner.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    # Drop torchrun env vars so vLLM's own TP init (multiproc executor) is not
    # confused by the RANK/LOCAL_RANK/WORLD_SIZE set by torch.distributed.run.
    for _k in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
    ):
        os.environ.pop(_k, None)
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt
    from transformers import AutoTokenizer
    from . import continuous_native_n


    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    end_ids = tok("</think>", add_special_tokens=False)["input_ids"]
    latent_end_token_id = end_ids[-1] if end_ids else None
    if latent_end_token_id is None:
        raise ValueError("Could not resolve the latent-end token id (`</think>`)")

    llm = LLM(
        model=args.model_path,
        enable_latent=True,
        latent_end_token_id=latent_end_token_id,
        latent_topk=args.topk_interpolation,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_total_length,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.gumbel_seed,
    )

    results: dict[tuple[int, int], dict] = {}
    prepared: list[tuple[dict, list[int]]] = []
    think_token_id = tok("<think>", add_special_tokens=False)["input_ids"][-1]
    reserved = 1 + args.max_latent_tokens + 1 + args.max_answer_tokens
    for example in examples:
        prefix = build_qwen_generation_prefix(tok, example)
        prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        prefix_budget = args.max_total_length - reserved
        prefix_ids = head_tail_truncate(prefix_ids, prefix_budget)
        prepared.append((example, prefix_ids + [think_token_id]))

    def sampling(example: dict, warm: bool = False) -> "SamplingParams":
        common = {
            "n": args.vote,
            "max_tokens": 2 if warm else args.max_latent_tokens + args.max_answer_tokens,
            "temperature": 0.0,
            "latent_end_token_id": -1 if warm else latent_end_token_id,
            "add_noise_gumbel_softmax": args.add_gumbel_noise,
            "noise_scale": args.noise_scale,
            "gumbel_softmax_temperature": args.gumbel_temperature,
            # Native parallel sampling derives child seed as base + vote_index.
            "seed": args.gumbel_seed + example["eval_index"] * 1009,
        }
        if warm:
            common.update(min_tokens=2, ignore_eos=True)
        return SamplingParams(**common)

    active_parents = max(
        1, math.ceil(args.vllm_target_active_rollouts / args.vote)
    )
    warm = prepared[: min(active_parents, len(prepared))]
    llm.generate(
        [TokensPrompt(prompt_token_ids=row[1]) for row in warm],
        [sampling(row[0], warm=True) for row in warm],
        use_tqdm=False,
    )
    llm.reset_prefix_cache()
    with tqdm(
        total=len(examples) * args.vote,
        desc="Latent vllm generate",
        unit="rollout",
    ) as pbar:
        round_started = time.monotonic()
        outputs = continuous_native_n.generate(
            llm,
            [TokensPrompt(prompt_token_ids=row[1]) for row in prepared],
            [sampling(row[0]) for row in prepared],
            active_requests=active_parents,
        )
        generation_only_seconds = time.monotonic() - round_started
        event_latency = {}
        for event in continuous_native_n.LAST_RUN_METRICS["child_finish_events"]:
            position = int(event["request_id"].rsplit("-", 1)[1])
            event_latency[(position, event["vote_index"])] = event["e2e_seconds"]
        for position, ((example, _), out) in enumerate(zip(prepared, outputs)):
            for gen in sorted(out.outputs, key=lambda value: value.index):
                current_vote_index = gen.index
                tokens = list(gen.token_ids)
                try:
                    split = tokens.index(latent_end_token_id)
                    reached = True
                except ValueError:
                    split = len(tokens)
                    reached = False
                answer_ids = tokens[split + 1 :] if reached else []
                generated = {
                    "text": tok.decode(
                        answer_ids, skip_special_tokens=True
                    ).strip(),
                    "stopped_early": reached,
                    "generate_token_num": split,
                    "answer_token_num": len(answer_ids),
                }
                latency = event_latency.get(
                    (position, current_vote_index), request_latency_seconds(out)
                )
                generated["generation_seconds"] = (
                    latency if latency is not None else generation_only_seconds
                )
                results[(example["eval_index"], current_vote_index)] = generated
                pbar.update(1)
    return results, generation_only_seconds


def main() -> None:
    args = parse_args()
    _validate_args(args)
    rank, world_size, device = distributed_runtime(args.device)
    started_at = time.monotonic()
    try:
        raw_data = read_jsonl(args.data_path)
        if not raw_data:
            raise ValueError("Evaluation data is empty")
        data = select_eval_records(
            raw_data, args.include_subsets, args.max_samples
        )
        signature = evaluation_signature(args, args.data_path)
        summary_path = args.output_path.with_suffix(
            args.output_path.suffix + ".summary.json"
        )
        part_paths = [
            args.output_path.with_name(
                args.output_path.name + f".rank{part_rank}.part.jsonl"
            )
            for part_rank in range(world_size)
        ]
        _coordinate_output_setup(args, rank, world_size, part_paths, summary_path)
        completed = _read_completed(
            [args.output_path, *_all_part_paths(args.output_path)], signature
        )
        assigned = [
            example
            for example in data
            if example["eval_index"] % world_size == rank
            and example["eval_index"] not in completed
        ]

        part_mode = "a" if args.resume else "w"
        generation_only_seconds = 0.0
        with part_paths[rank].open(
            part_mode, encoding="utf-8", buffering=1
        ) as output_file:
            if args.backend == "vllm":
                generated_map, generation_only_seconds = generate_with_vllm(
                    args, assigned, len(raw_data), rank
                )
                for example in tqdm(
                    assigned,
                    desc=f"Evaluating OpenRubric (vllm) rank {rank}",
                    disable=rank != 0,
                ):
                    generated_rollouts = [
                        generated_map[(example["eval_index"], vote_index)]
                        for vote_index in range(args.vote)
                    ]
                    row = _vote_result_row(
                        example, generated_rollouts, signature, rank
                    )
                    output_file.write(
                        json.dumps(row, ensure_ascii=False) + "\n"
                    )
            else:
                model = LatentSFTStage2SoftEmbedding(
                    latent_model_path=args.model_path,
                    lora_tune=bool(args.lora_path),
                    lora_path=args.lora_path,
                    bfloat16=device.type == "cuda",
                    use_flash_attention_2=(
                        args.use_flash_attention_2 and device.type == "cuda"
                    ),
                    training=False,
                    topk_interpolation=args.topk_interpolation,
                ).to(device)
                model.eval()
                if rank == 0 and args.vote > 1 and not args.add_gumbel_noise:
                    print(
                        "warning: --vote > 1 without --add_gumbel_noise repeats "
                        "deterministic rollouts and is not a TTS measurement",
                        flush=True,
                    )

                for example in tqdm(
                    assigned,
                    desc=f"Evaluating OpenRubric rank {rank}",
                    disable=rank != 0,
                ):
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    sample_started_at = time.monotonic()
                    generated_rollouts = []
                    for vote_index in range(args.vote):
                        gumbel_generator = None
                        if args.add_gumbel_noise:
                            gumbel_generator = torch.Generator(device=device)
                            # vote_index=0 retains the historical vote=1 seed.
                            gumbel_generator.manual_seed(
                                args.gumbel_seed
                                + example["eval_index"]
                                + vote_index * len(raw_data)
                            )
                        generated_rollouts.append(
                            model.one_example_generate_lora(
                                prepare_inputs(
                                    example,
                                    model,
                                    device,
                                    args.max_total_length,
                                    args.max_latent_tokens,
                                    args.max_answer_tokens,
                                ),
                                max_new_tokens=args.max_latent_tokens,
                                max_answer_tokens=args.max_answer_tokens,
                                do_sample=False,
                                add_gumbel_noise=args.add_gumbel_noise,
                                gumbel_temperature=args.gumbel_temperature,
                                noise_scale=args.noise_scale,
                                gumbel_generator=gumbel_generator,
                            )
                        )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    sample_elapsed = time.monotonic() - sample_started_at
                    row = _vote_result_row(
                        example, generated_rollouts, signature, rank
                    )
                    row["generation_seconds"] = sample_elapsed
                    output_file.write(
                        json.dumps(row, ensure_ascii=False) + "\n"
                    )

        if world_size > 1:
            dist.barrier()
        if rank == 0:
            completed = _read_completed(
                [args.output_path, *_all_part_paths(args.output_path)],
                signature,
            )
            expected_indices = {example["eval_index"] for example in data}
            actual_indices = set(completed)
            if actual_indices != expected_indices:
                missing = sorted(expected_indices - actual_indices)
                extra = sorted(actual_indices - expected_indices)
                raise RuntimeError(
                    "Evaluation shards are incomplete: "
                    f"missing={missing[:20]}, extra={extra[:20]}"
                )
            results = [completed[index] for index in sorted(expected_indices)]
            _atomic_write_jsonl(args.output_path, results)
            summary = summarize_results(
                results,
                elapsed_seconds=time.monotonic() - started_at,
                generation_only_seconds=generation_only_seconds,
            )
            # RewardBench2 数据额外附加官方评分（score_percent），
            # 保证无论 vote 多少、从哪个入口评测，summary 都含官方统计。
            if raw_data and all(
                key in raw_data[0]
                for key in (
                    "source_dataset_index",
                    "prompt_id",
                    "chosen_index",
                    "rejected_index",
                    "subset",
                    "exchange",
                )
            ):
                try:
                    # 延迟 import 避免与 summarize_rewardbench2 的循环导入。
                    from .summarize_rewardbench2 import summarize_rewardbench2

                    rb2 = summarize_rewardbench2(results, raw_data)
                    summary["rewardbench2"] = rb2["rewardbench2"]
                    summary["rewardbench2_diagnostics"] = rb2["rewardbench2_diagnostics"]
                except Exception as exc:
                    print(
                        "warning: rewardbench2 official scoring skipped: "
                        f"{exc}",
                        flush=True,
                    )
            summary.update(
                {
                    "evaluation_id": signature,
                    "data_path": str(args.data_path.resolve()),
                    "include_subsets": list(
                        parse_subset_filter(args.include_subsets) or ()
                    ),
                    "model_path": args.model_path,
                    "lora_path": args.lora_path,
                    "world_size": world_size,
                    "max_total_length": args.max_total_length,
                    "max_latent_tokens": args.max_latent_tokens,
                    "max_answer_tokens": args.max_answer_tokens,
                    "topk_interpolation": args.topk_interpolation,
                    "add_gumbel_noise": args.add_gumbel_noise,
                    "gumbel_temperature": args.gumbel_temperature,
                    "noise_scale": args.noise_scale,
                    "gumbel_seed": args.gumbel_seed,
                    "vote_num": args.vote,
                    "vllm_target_active_rollouts": args.vllm_target_active_rollouts,
                    "vllm_active_parent_requests": (
                        math.ceil(args.vllm_target_active_rollouts / args.vote)
                        if args.backend == "vllm" else None
                    ),
                    "vllm_nominal_active_rollouts": (
                        math.ceil(args.vllm_target_active_rollouts / args.vote)
                        * args.vote if args.backend == "vllm" else None
                    ),
                }
            )
            _atomic_write_json(summary_path, summary)
            for path in _all_part_paths(args.output_path):
                path.unlink(missing_ok=True)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        if world_size > 1:
            dist.barrier()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
