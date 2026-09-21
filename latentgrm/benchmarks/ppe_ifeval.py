"""Pair construction and scoring for ppe ifeval."""

from __future__ import annotations

from collections import Counter, defaultdict

from typing import Any

from .common import parse_list_like, score_percent

EXPECTED_QUESTIONS = 512

CONFLICT_PAIRS_PER_QUESTION = 5

RUBRIC = (
    "Select the response that follows every explicit instruction in the user "
    "prompt. A response satisfying all constraints is better than one that "
    "violates any constraint. Judge only instruction-following compliance."
)

def build_pair_records(
    source_rows: list[dict[str, Any]], bidirectional: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    originals: list[dict[str, Any]] = []
    model_counts: dict[str, int] = {}
    for source_index, row in enumerate(source_rows):
        scores = [bool(value) for value in parse_list_like(row["scores"], "scores")]
        pairs = parse_list_like(row["sampled_conflict_pairs"], "sampled_conflict_pairs")
        question_id = str(row["question_id"])
        model_name = str(row.get("model_name") or "unknown")
        model_counts[model_name] = model_counts.get(model_name, 0) + 1
        for pair_index, pair_value in enumerate(pairs):
            pair = [
                int(index) for index in parse_list_like(pair_value, "conflict pair")
            ]
            if len(pair) != 2:
                raise ValueError(f"{question_id}: conflict pair must have two indices")
            left, right = pair
            if not (0 <= left < len(scores) and 0 <= right < len(scores)):
                raise ValueError(f"{question_id}: conflict pair index out of range")
            if scores[left] == scores[right]:
                raise ValueError(
                    f"{question_id}: sampled pair has no correctness conflict"
                )
            record = {
                "dataset_index": len(originals),
                "source_dataset_index": source_index,
                "question_id": question_id,
                "pair_index": pair_index,
                "left_response_index": left,
                "right_response_index": right,
                "model_name": model_name,
                "subset": "ppe-ifeval",
                "exchange": False,
                "instruction": str(row["prompt"]),
                "rubric": RUBRIC,
                "response_a": str(row[f"response_{left + 1}"]),
                "response_b": str(row[f"response_{right + 1}"]),
                "label": "response_a" if scores[left] else "response_b",
            }
            originals.append(record)

    rows = list(originals)
    if bidirectional:
        rows.extend(
            {
                **record,
                "exchange": True,
                "response_a": record["response_b"],
                "response_b": record["response_a"],
                "label": (
                    "response_b" if record["label"] == "response_a" else "response_a"
                ),
            }
            for record in originals
        )
    manifest = {
        "benchmark": "PPE-IFEval-Best-of-K",
        "official_metric_for_llm_judges": "sampled_conflict_pair_accuracy",
        "question_count": len(source_rows),
        "original_pair_count": len(originals),
        "output_row_count": len(rows),
        "bidirectional": bidirectional,
        "model_question_counts": dict(sorted(model_counts.items())),
        "complete": (
            len(source_rows) == EXPECTED_QUESTIONS
            and len(originals) == EXPECTED_QUESTIONS * CONFLICT_PAIRS_PER_QUESTION
        ),
    }
    return rows, manifest

def _accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(row.get("correct", False)) for row in rows)
    score = correct / len(rows)
    return {
        "score_percent": score_percent(score),
        "samples": len(rows),
        "correct": correct,
    }

def _direction(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model_name"]), []).append(row)
    overall = _accuracy(rows)
    return {
        "Accuracy": {"score_percent": overall["score_percent"]},
        "Models": {
            model_name: {"score_percent": _accuracy(values)["score_percent"]}
            for model_name, values in sorted(by_model.items())
        },
        "samples": overall["samples"],
        "correct": overall["correct"],
    }

def summarize(
    results: list[dict[str, Any]], data: list[dict[str, Any]]
) -> dict[str, Any]:
    joined = []
    seen: set[int] = set()
    for result in results:
        index = result.get("eval_index")
        if not isinstance(index, int) or not 0 <= index < len(data):
            raise ValueError(f"invalid eval_index: {index!r}")
        if index in seen:
            raise ValueError(f"duplicate eval_index: {index}")
        seen.add(index)
        joined.append(
            {
                **result,
                "exchange": bool(data[index].get("exchange", False)),
                "model_name": str(data[index].get("model_name") or "unknown"),
            }
        )
    forward = _direction([row for row in joined if not row["exchange"]])
    reverse = _direction([row for row in joined if row["exchange"]])
    if forward and reverse:
        average_score = (
            forward["Accuracy"]["score_percent"] + reverse["Accuracy"]["score_percent"]
        ) / 2
        average = {"Score": {"score_percent": average_score}}
    else:
        selected = forward or reverse
        average = {"Score": selected["Accuracy"]} if selected else None

    def compact(value: dict[str, Any] | None) -> dict[str, Any] | None:
        return {"Score": value["Accuracy"]} if value else None

    return {
        "ppe_ifeval": {
            "average": average,
            "forward": compact(forward),
            "reverse": compact(reverse),
        },
        "ppe_ifeval_diagnostics": {
            "official_metric": "sampled_conflict_pair_accuracy",
            "official_dataset_order_score": compact(forward),
            "headline_is_bidirectional_extension": bool(forward and reverse),
            "unsupported_binary_judge_metrics": [
                "best_of_k",
                "maximum_achieved_performance",
                "end_score",
                "loss",
                "roc_auc",
            ],
            "forward_samples": forward["samples"] if forward else 0,
            "reverse_samples": reverse["samples"] if reverse else 0,
            "complete": bool(
                forward
                and reverse
                and forward["samples"]
                == EXPECTED_QUESTIONS * CONFLICT_PAIRS_PER_QUESTION
                and reverse["samples"]
                == EXPECTED_QUESTIONS * CONFLICT_PAIRS_PER_QUESTION
            ),
        },
    }
