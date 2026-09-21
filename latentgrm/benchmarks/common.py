"""Shared benchmark data helpers."""
import ast
import math
from typing import Any

def to_builtin(value: Any) -> Any:
    """Convert numpy/pandas containers from parquet rows into Python values."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return int(value)
        return value
    if hasattr(value, "tolist"):
        return to_builtin(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value

def parse_list_like(value: Any, field: str) -> list[Any]:
    value = to_builtin(value)
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"cannot parse {field} as a list") from exc
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    raise ValueError(f"{field} must be a list, got {type(value).__name__}")

def score_percent(value: float | None) -> float | None:
    return value * 100 if value is not None else None
