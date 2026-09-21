"""Pair construction and scoring for helpsteer3."""

from __future__ import annotations

from collections import Counter, defaultdict

from typing import Any

from .common import parse_list_like, score_percent

OFFICIAL_DATASET_ID = "nvidia/HelpSteer3"

OFFICIAL_REVISION = "f6d145777bcbde96137596340fab89793acd1031"

EXPECTED_VALIDATION_SAMPLES = 2017

DOMAINS = ("general", "stem", "code", "multilingual")

DOMAIN_LABELS = {
    "general": "General",
    "stem": "STEM",
    "code": "Code",
    "multilingual": "Multilingual",
}

STRENGTH_LABELS = {1: "Slightly Better", 2: "Better", 3: "Much Better"}

RUBRIC = (
    "Choose the response that is more helpful overall. Prioritize instruction "
    "following and factual correctness/completeness, then relevance, coherence, "
    "clarity, and appropriate style. Response length alone is not evidence of "
    "higher quality."
)

def render_context(context: Any) -> str:
    if not isinstance(context, list) or not context:
        raise ValueError("HelpSteer3 context must be a non-empty message list")
    rendered = []
    for index, message in enumerate(context):
        if not isinstance(message, dict):
            raise TypeError(f"context message {index} is not an object")
        role = str(message.get("role", "")).strip().lower()
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"context message {index} has invalid role: {role!r}")
        if not isinstance(content, str):
            raise TypeError(f"context message {index} content is not text")
        rendered.append(f"{role.capitalize()}:\n{content.strip()}")
    return "Conversation context:\n\n" + "\n\n".join(rendered)

def build_pair_records(
    source_rows: list[dict[str, Any]], bidirectional: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    originals = []
    tie_count = 0
    source_domain_counts: dict[str, int] = defaultdict(int)
    scored_domain_counts: dict[str, int] = defaultdict(int)
    for source_index, row in enumerate(source_rows):
        domain = str(row.get("domain", "")).strip().lower()
        if domain not in DOMAINS:
            raise ValueError(f"row {source_index}: invalid domain {domain!r}")
        source_domain_counts[domain] += 1
        preference = row.get("overall_preference")
        if not isinstance(preference, int) or not -3 <= preference <= 3:
            raise ValueError(
                f"row {source_index}: overall_preference must be an integer in [-3, 3]"
            )
        if preference == 0:
            tie_count += 1
            continue
        response1 = row.get("response1")
        response2 = row.get("response2")
        if response1 is None or response2 is None:
            continue
        if not isinstance(response1, str) or not isinstance(response2, str):
            raise TypeError(f"row {source_index}: responses must be text")
        if not response1.strip() or not response2.strip():
            print(f"row {source_index}: empty response, skipping", flush=True)
            continue
        scored_domain_counts[domain] += 1
        originals.append(
            {
                "dataset_index": len(originals),
                "source_dataset_index": source_index,
                "subset": f"helpsteer3-{domain}",
                "domain": domain,
                "language": str(row.get("language") or "unknown"),
                "preference_strength": abs(preference),
                "overall_preference": preference,
                "exchange": False,
                "instruction": render_context(row.get("context")),
                "rubric": RUBRIC,
                "response_a": response1,
                "response_b": response2,
                # Official labels: negative means response 1 is better.
                "label": "response_a" if preference < 0 else "response_b",
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
                "label": (
                    "response_b" if record["label"] == "response_a" else "response_a"
                ),
            }
            for record in originals
        )
    manifest = {
        "dataset": OFFICIAL_DATASET_ID,
        "benchmark_kind": "held_out_preference_accuracy",
        "source_samples": len(source_rows),
        "source_domain_counts": dict(sorted(source_domain_counts.items())),
        "scored_domain_counts": dict(sorted(scored_domain_counts.items())),
        "tie_samples_excluded": tie_count,
        "tie_policy": "exclude, matching the paper's reward-model training protocol",
        "original_pair_count": len(originals),
        "output_row_count": len(rows),
        "bidirectional": bidirectional,
        "complete_official_validation": len(source_rows) == EXPECTED_VALIDATION_SAMPLES,
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
    domains: dict[str, list[dict[str, Any]]] = defaultdict(list)
    strengths: dict[int, list[dict[str, Any]]] = defaultdict(list)
    languages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        domains[row["domain"]].append(row)
        strengths[row["preference_strength"]].append(row)
        languages[row["language"]].append(row)
    overall = _bucket(rows)
    return {
        **{
            DOMAIN_LABELS[domain]: {
                "score_percent": _bucket(domains[domain])["score_percent"]
            }
            for domain in DOMAINS
            if domains[domain]
        },
        "Score": {"score_percent": overall["score_percent"]},
        "_strengths": {
            STRENGTH_LABELS[strength]: _bucket(values)
            for strength, values in sorted(strengths.items())
        },
        "_languages": {
            language: _bucket(values) for language, values in sorted(languages.items())
        },
        "_samples": overall["samples"],
        "_invalid_outputs": overall["invalid_outputs"],
    }

def _compact(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"Score": value["Score"]}

def _average_directions(
    forward: dict[str, Any] | None, reverse: dict[str, Any] | None
) -> dict[str, Any] | None:
    if forward is None or reverse is None:
        return _compact(forward or reverse)
    keys = ["Score"]
    return {
        key: {
            "score_percent": (
                forward[key]["score_percent"] + reverse[key]["score_percent"]
            )
            / 2
        }
        for key in keys
        if key in forward and key in reverse
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
        source = data[index]
        joined.append(
            {
                **result,
                "exchange": bool(source.get("exchange", False)),
                "domain": str(source["domain"]),
                "language": str(source.get("language") or "unknown"),
                "preference_strength": int(source["preference_strength"]),
            }
        )
    forward = _direction([row for row in joined if not row["exchange"]])
    reverse = _direction([row for row in joined if row["exchange"]])
    expected_forward = sum(1 for row in data if not row.get("exchange", False))
    expected_reverse = len(data) - expected_forward
    return {
        "helpsteer3": {
            "average": _average_directions(forward, reverse),
            "forward": _compact(forward),
            "reverse": _compact(reverse),
        },
        "helpsteer3_diagnostics": {
            "metric": "openrubrics_validation_pairwise_preference_accuracy",
            "headline_aggregation": "micro_average_over_non_tie_validation_rows",
            "tie_policy": "overall_preference == 0 excluded",
            "official_dataset_order_score": _compact(forward),
            "headline_is_bidirectional_extension": bool(forward and reverse),
            "forward_samples": forward["_samples"] if forward else 0,
            "reverse_samples": reverse["_samples"] if reverse else 0,
            "forward_invalid_outputs": forward["_invalid_outputs"] if forward else 0,
            "reverse_invalid_outputs": reverse["_invalid_outputs"] if reverse else 0,
            "forward_strengths": forward["_strengths"] if forward else {},
            "reverse_strengths": reverse["_strengths"] if reverse else {},
            "forward_languages": forward["_languages"] if forward else {},
            "reverse_languages": reverse["_languages"] if reverse else {},
            "complete": bool(
                forward
                and reverse
                and forward["_samples"] == expected_forward
                and reverse["_samples"] == expected_reverse
            ),
        },
    }
