"""Shared generation-token and latency aggregation for judge evaluations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

REWARDBENCH_OFFICIAL_SECTIONS = {
    "Chat": {
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    },
    "Chat Hard": {
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    },
    "Safety": {
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    },
    "Reasoning": {
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    },
}
REWARDBENCH2_OFFICIAL_SUBSETS = {
    "Factuality",
    "Precise IF",
    "Math",
    "Safety",
    "Focus",
    "Ties",
}


def official_subset_name(row: dict[str, Any]) -> str:
    """Map benchmark cache metadata to the benchmark's official reporting group."""
    raw = str(row.get("subset") or "unknown")
    for section, members in REWARDBENCH_OFFICIAL_SECTIONS.items():
        if raw in members:
            return section
    if raw in REWARDBENCH2_OFFICIAL_SUBSETS:
        return raw
    if raw.startswith("rm-bench-"):
        return str(row.get("domain") or raw.removeprefix("rm-bench-")).title()
    if raw.startswith("helpsteer3-"):
        domain = str(row.get("domain") or raw.removeprefix("helpsteer3-")).lower()
        return "STEM" if domain == "stem" else domain.title()
    if raw == "ifbench":
        difficulty = row.get("difficulty")
        return str(difficulty).title() if difficulty else "IFBench"
    if raw == "ppe-ifeval":
        return "PPE-IFEval"
    return raw


def request_latency_seconds(request_output: Any) -> float | None:
    """Return vLLM request latency from monotonic engine timestamps when present."""
    metrics = getattr(request_output, "metrics", None)
    if metrics is None:
        return None
    end = getattr(metrics, "last_token_ts", None)
    if not isinstance(end, (int, float)) or end <= 0:
        return None
    for field in ("queued_ts", "scheduled_ts"):
        start = getattr(metrics, field, None)
        if isinstance(start, (int, float)) and start > 0 and end >= start:
            return float(end) - float(start)
    return None


def _stats(values: list[float | int]) -> dict[str, float | int]:
    total = sum(values)
    return {
        "count": len(values),
        "total": total,
        "mean": total / len(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_efficiency(
    rows: list[dict[str, Any]],
    token_count: Callable[[dict[str, Any]], int | None],
) -> dict[str, Any]:
    """Aggregate generated tokens and measured generation latency.

    ``per_sample`` token counts sum every vote for an input. Sample latency is
    measured once around the complete vote set (or by one vLLM request for the
    baseline). Missing timing fields, such as rows from an older resume file,
    are excluded and remain visible through the reported ``count``.
    """
    rollout_tokens: list[int] = []
    sample_tokens: list[int] = []
    rollout_latencies: list[float] = []
    sample_latencies: list[float] = []

    for row in rows:
        rollouts = row.get("vote_rollouts") or [row]
        row_tokens: list[int] = []
        for rollout in rollouts:
            value = token_count(rollout)
            if value is not None:
                value = int(value)
                rollout_tokens.append(value)
                row_tokens.append(value)
            latency = rollout.get("generation_seconds")
            if isinstance(latency, (int, float)) and latency >= 0:
                rollout_latencies.append(float(latency))
        if row_tokens:
            sample_tokens.append(sum(row_tokens))
        latency = row.get("generation_seconds")
        if isinstance(latency, (int, float)) and latency >= 0:
            sample_latencies.append(float(latency))

    result: dict[str, Any] = {}
    if rollout_tokens:
        result["generation_tokens"] = {
            "per_rollout": _stats(rollout_tokens),
            "per_sample": _stats(sample_tokens),
        }
    if sample_latencies:
        timing: dict[str, Any] = {"per_sample": _stats(sample_latencies)}
        if rollout_latencies:
            timing["per_rollout"] = _stats(rollout_latencies)
        result["generation_latency_seconds"] = timing
    return result
