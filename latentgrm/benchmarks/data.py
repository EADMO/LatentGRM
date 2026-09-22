from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .parsing import normalize_winner

REQUIRED_FIELDS = {
    "instruction",
    "rubric",
    "source",
    "judge",
    "response_a",
    "response_b",
    "winner",
}

def validate_openrubrics_columns(column_names: Iterable[str]) -> None:
    columns = set(column_names)
    missing = sorted(REQUIRED_FIELDS - columns)
    if missing:
        raise ValueError(f"OpenRubrics dataset is missing required fields: {missing}")

def validate_openrubrics_record(record: dict[str, Any]) -> None:
    validate_openrubrics_columns(record.keys())
    for field in ("instruction", "rubric", "judge", "response_a", "response_b"):
        if not str(record.get(field, "")).strip():
            raise ValueError(f"record field {field!r} is empty")
    normalize_winner(record["winner"])
