from __future__ import annotations


from collections import Counter


from typing import Any, Iterable, Sequence


DEFAULT_DATASET_ID = "data/rewardbench2"

DEFAULT_CACHE_DIR = "data/rewardbench2_cache"

DEFAULT_EXTRACT_DIR = "data/rewardbench2_extracted"

DEFAULT_OUTPUT_DIR = "data/rewardbench2_rubric"

DEFAULT_EVAL_SUBSETS = (
    "Factuality",
    "Precise IF",
    "Math",
    "Safety",
    "Focus",
    "Ties",
)

def _text_list(value: Any, field: str, dataset_index: int) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"RewardBench 2 row {dataset_index}: {field} must be a list")
    result = [str(item) for item in value]
    if not result or any(not item.strip() for item in result):
        raise ValueError(f"RewardBench 2 row {dataset_index}: {field} contains no usable responses")
    return result

def build_extracted_records(
    dataset: Iterable[dict[str, Any]],
    limit: int | None = None,
    subsets: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    subset_counts: Counter[str] = Counter()
    selected_subsets = set(subsets) if subsets is not None else None

    for source_dataset_index, raw in enumerate(dataset):
        subset = str(raw.get("subset") or "unknown")
        if selected_subsets is not None and subset not in selected_subsets:
            continue
        if limit is not None and len(prompts) >= limit:
            break
        instruction = raw.get("prompt")
        if instruction in (None, ""):
            raise ValueError(f"RewardBench 2 row {source_dataset_index}: missing prompt")
        chosen = _text_list(raw.get("chosen"), "chosen", source_dataset_index)
        rejected = _text_list(raw.get("rejected"), "rejected", source_dataset_index)
        prompt_id = str(raw.get("id", source_dataset_index))
        prompt_record = {
            "source_dataset_index": source_dataset_index,
            "prompt_id": prompt_id,
            "subset": subset,
            "instruction": str(instruction),
            "chosen": chosen,
            "rejected": rejected,
            "num_correct": int(raw.get("num_correct", len(chosen))),
            "num_incorrect": int(raw.get("num_incorrect", len(rejected))),
            "total_completions": int(raw.get("total_completions", len(chosen) + len(rejected))),
            "models": list(raw.get("models") or []),
            "additional_metadata": raw.get("additional_metadata") or {},
        }
        prompts.append(prompt_record)
        subset_counts[subset] += 1

        for chosen_index, response_a in enumerate(chosen):
            for rejected_index, response_b in enumerate(rejected):
                pair_index = len(pairs)
                pairs.append(
                    {
                        "dataset_index": pair_index,
                        "source_dataset_index": source_dataset_index,
                        "prompt_id": prompt_id,
                        "chosen_index": chosen_index,
                        "rejected_index": rejected_index,
                        "num_correct": len(chosen),
                        "num_incorrect": len(rejected),
                        "total_completions": len(chosen) + len(rejected),
                        "subset": subset,
                        "exchange": False,
                        "instruction": str(instruction),
                        "response_a": response_a,
                        "response_b": response_b,
                        "label": "response_a",
                    }
                )

    manifest = {
        "dataset": DEFAULT_DATASET_ID,
        "prompt_count": len(prompts),
        "pair_count": len(pairs),
        "bidirectional_pair_count": len(pairs) * 2,
        "subset_prompt_counts": dict(sorted(subset_counts.items())),
        "selected_subsets": list(subsets) if subsets is not None else None,
    }
    return prompts, pairs, manifest

def reverse_pair(row: dict[str, Any]) -> dict[str, Any]:
    if bool(row.get("exchange", False)):
        raise ValueError("only exchange=false rows can be reversed")
    result = dict(row)
    result["response_a"] = row["response_b"]
    result["response_b"] = row["response_a"]
    result["label"] = "response_b"
    result["exchange"] = True
    return result
