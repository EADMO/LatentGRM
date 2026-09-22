"""Parsing utilities for model outputs and dataset labels."""

from __future__ import annotations

import re
from typing import Any


def normalize_winner(value: Any) -> str:
    """Normalize common winner labels to ``response_a`` or ``response_b``."""
    if value is None:
        raise ValueError("winner is missing")

    text = str(value).strip().lower()
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text)

    if text in {"a", "response a", "assistant a", "model a"}:
        return "response_a"
    if text in {"b", "response b", "assistant b", "model b"}:
        return "response_b"
    if "response a" in text or text.endswith(": a"):
        return "response_a"
    if "response b" in text or text.endswith(": b"):
        return "response_b"

    raise ValueError(f"unsupported winner label: {value!r}")


def rewardbench_label_from_record(record: dict[str, Any]) -> str:
    """Infer the correct label for a normalized RewardBench record."""
    for key in ("winner", "label", "chosen_label"):
        if key in record and record[key] not in (None, ""):
            return normalize_winner(record[key])

    # Official RewardBench-style datasets commonly expose chosen/rejected.
    if "chosen" in record and "rejected" in record:
        return "response_a"

    raise ValueError("could not infer RewardBench label from record")
