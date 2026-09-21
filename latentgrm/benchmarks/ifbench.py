"""Pair construction and scoring for ifbench."""

from __future__ import annotations

from collections import Counter, defaultdict

from typing import Any

from .common import parse_list_like, score_percent

EXPECTED_SAMPLES = 444

EXPECTED_DIFFICULTY_COUNTS = {"simple": 47, "normal": 133, "hard": 264}

DIFFICULTIES = ("simple", "normal", "hard")

def difficulty_from_unsatisfied(count: int) -> str:
    if count <= 0:
        raise ValueError("rejected response must violate at least one constraint")
    if count == 1:
        return "hard"
    if count == 2:
        return "normal"
    return "simple"

def _constraints(row: dict[str, Any], source_index: int) -> list[str]:
    constraints = []
    for field in ("llm_constraints_used", "code_constraints_used"):
        values = row.get(field)
        if not isinstance(values, list):
            raise TypeError(f"row {source_index}: {field} must be a list")
        for value in values:
            if not isinstance(value, dict):
                raise TypeError(f"row {source_index}: constraint must be an object")
            constraint = value.get("constraint")
            if not isinstance(constraint, str) or not constraint.strip():
                raise ValueError(f"row {source_index}: constraint text is empty")
            constraints.append(constraint.strip())
    # The paper describes 3-5 constraints, while the released 444-row file has
    # two valid rows with two constraints. Preserve the complete released set.
    if not 2 <= len(constraints) <= 5:
        raise ValueError(
            f"row {source_index}: expected 2 to 5 constraints, got {len(constraints)}"
        )
    return constraints

def _response(
    row: dict[str, Any], field: str, source_index: int
) -> tuple[str, list[Any]]:
    value = row.get(field)
    if not isinstance(value, dict):
        raise TypeError(f"row {source_index}: {field} must be an object")
    content = value.get("content")
    unsatisfied = value.get("unsatisfied_constraints")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"row {source_index}: {field}.content is empty")
    if not isinstance(unsatisfied, list):
        raise TypeError(
            f"row {source_index}: {field}.unsatisfied_constraints must be a list"
        )
    return content, unsatisfied

def render_rubric(constraints: list[str]) -> str:
    rendered = "\n".join(
        f"{index}. The response must {constraint.rstrip('.')}."
        for index, constraint in enumerate(constraints, 1)
    )
    return (
        "Select the response that follows every requirement in the instruction. "
        "A response satisfying all constraints is better than one violating any "
        "constraint. Evaluate these requirements:\n" + rendered
    )

def build_pair_records(
    source_rows: list[dict[str, Any]], bidirectional: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    originals = []
    seen_ids: set[str] = set()
    difficulty_counts: Counter[str] = Counter()
    for source_index, row in enumerate(source_rows):
        source_id = str(row.get("id"))
        if source_id in seen_ids:
            raise ValueError(f"duplicate IFBench id: {source_id}")
        seen_ids.add(source_id)
        instruction = row.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"row {source_index}: instruction is empty")
        chosen, chosen_unsatisfied = _response(row, "chosen", source_index)
        rejected, rejected_unsatisfied = _response(row, "rejected", source_index)
        if chosen_unsatisfied:
            raise ValueError(
                f"row {source_index}: chosen response has unsatisfied constraints"
            )
        difficulty = difficulty_from_unsatisfied(len(rejected_unsatisfied))
        difficulty_counts[difficulty] += 1
        constraints = _constraints(row, source_index)
        originals.append(
            {
                "dataset_index": len(originals),
                "source_dataset_index": source_index,
                "source_id": source_id,
                "source": str(row.get("source") or "unknown"),
                "subset": "ifbench",
                "difficulty": difficulty,
                "unsatisfied_constraint_count": len(rejected_unsatisfied),
                "exchange": False,
                "instruction": instruction.strip(),
                "rubric": render_rubric(constraints),
                "response_a": chosen,
                "response_b": rejected,
                "label": "response_a",
            }
        )

    rows = list(originals)
    if bidirectional:
        rows.extend(
            {
                **record,
                "exchange": True,
                "response_a": record["response_b"],
                "response_b": record["response_a"],
                "label": "response_b",
            }
            for record in originals
        )
    manifest = {
        "benchmark": "THU-KEG/IFBench",
        "paper": "Agentic Reward Modeling (Peng et al., ACL 2025)",
        "official_metric": "micro_accuracy_over_all_444_pairs",
        "source_samples": len(source_rows),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "released_constraint_count_note": (
            "paper states 3-5; released data contains two rows with 2 constraints"
        ),
        "original_pair_count": len(originals),
        "output_row_count": len(rows),
        "bidirectional": bidirectional,
        "complete": (
            len(source_rows) == EXPECTED_SAMPLES
            and dict(difficulty_counts) == EXPECTED_DIFFICULTY_COUNTS
        ),
    }
    return rows, manifest

def _bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(row.get("correct", False)) for row in rows)
    return {
        "score_percent": score_percent(correct / len(rows)),
        "samples": len(rows),
        "correct": correct,
        "invalid_outputs": sum(row.get("prediction") is None for row in rows),
    }

def _direction(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_difficulty[row["difficulty"]].append(row)
    overall = _bucket(rows)
    return {
        "Score": {"score_percent": overall["score_percent"]},
        "_difficulties": {
            difficulty.title(): _bucket(by_difficulty[difficulty])
            for difficulty in DIFFICULTIES
            if by_difficulty[difficulty]
        },
        "_samples": overall["samples"],
        "_invalid_outputs": overall["invalid_outputs"],
        "_complete": overall["samples"] == EXPECTED_SAMPLES
        and {difficulty: len(by_difficulty[difficulty]) for difficulty in DIFFICULTIES}
        == EXPECTED_DIFFICULTY_COUNTS,
    }

def _compact(value: dict[str, Any] | None) -> dict[str, Any] | None:
    return {"Score": value["Score"]} if value else None

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
        source = data[index]
        difficulty = str(source.get("difficulty", "")).lower()
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"invalid IFBench difficulty: {difficulty!r}")
        joined.append(
            {
                **result,
                "exchange": bool(source.get("exchange", False)),
                "difficulty": difficulty,
            }
        )
    forward = _direction([row for row in joined if not row["exchange"]])
    reverse = _direction([row for row in joined if row["exchange"]])
    if forward and reverse:
        average = {
            "Score": {
                "score_percent": (
                    forward["Score"]["score_percent"]
                    + reverse["Score"]["score_percent"]
                )
                / 2
            }
        }
    else:
        average = _compact(forward or reverse)
    return {
        "ifbench": {
            "average": average,
            "forward": _compact(forward),
            "reverse": _compact(reverse),
        },
        "ifbench_diagnostics": {
            "dataset": "THU-KEG/IFBench",
            "official_metric": "micro_accuracy_over_all_444_pairs",
            "official_dataset_order_score": _compact(forward),
            "reported_average_is_order_robust_extension": bool(forward and reverse),
            "forward_samples": forward["_samples"] if forward else 0,
            "reverse_samples": reverse["_samples"] if reverse else 0,
            "forward_invalid_outputs": forward["_invalid_outputs"] if forward else 0,
            "reverse_invalid_outputs": reverse["_invalid_outputs"] if reverse else 0,
            "forward_difficulties": forward["_difficulties"] if forward else {},
            "reverse_difficulties": reverse["_difficulties"] if reverse else {},
            "complete": bool(
                forward and reverse and forward["_complete"] and reverse["_complete"]
            ),
        },
    }
