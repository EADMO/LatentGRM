from __future__ import annotations


from pathlib import Path

from typing import Any


from .parsing import rewardbench_label_from_record

from .paths import data_path


DEFAULT_REWARDBENCH_ID = data_path("rewardbench")

DEFAULT_CACHE_DIR = Path(data_path("rewardbench_cache"))

def normalize_rewardbench_record(record: dict[str, Any], exchange: bool = False) -> dict[str, Any]:
    instruction = (
        record.get("prompt")
        or record.get("instruction")
        or record.get("question")
        or record.get("input")
    )
    if instruction is None:
        raise ValueError("RewardBench record has no prompt/instruction/question/input field")

    if "response_a" in record and "response_b" in record:
        response_a = record["response_a"]
        response_b = record["response_b"]
        label = rewardbench_label_from_record(record)
    elif "chosen" in record and "rejected" in record:
        if exchange:
            response_a = record["rejected"]
            response_b = record["chosen"]
            label = "response_b"
        else:
            response_a = record["chosen"]
            response_b = record["rejected"]
            label = "response_a"
    else:
        raise ValueError("RewardBench record needs response_a/response_b or chosen/rejected fields")

    if exchange and "response_a" in record and "response_b" in record:
        response_a, response_b = response_b, response_a
        label = "response_b" if label == "response_a" else "response_a"

    subset = (
        record.get("subset")
        or record.get("category")
        or record.get("section")
        or record.get("source")
        or "unknown"
    )
    return {
        "instruction": str(instruction),
        "response_a": str(response_a),
        "response_b": str(response_b),
        "label": label,
        "subset": str(subset),
        "exchange": exchange,
    }
