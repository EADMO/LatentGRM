"""Semantic Chunking boundary settings."""
from dataclasses import dataclass
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class Config:
    parser: str = "spacy"
    parser_model: str = "models/en_core_web_sm"
    displacement: int = 4
    lexical_rescue_displacement: int = 5
    length_weight: float = 0.5
    variance_weight: float = 0.02
    displacement_weight: float = 0.10
    risk_weight: float = 3.0
    semantic_margin: float = 0.5
    compact_span_tokens: int = 8
    sentence_reward: float = 3.0


def load_config():
    path = os.environ.get("LATENTGRM_CHUNK_CONFIG")
    if not path:
        return Config()
    payload = json.loads(Path(path).read_text())
    if "displacement" in payload and "lexical_rescue_displacement" not in payload:
        payload["lexical_rescue_displacement"] = min(int(payload["displacement"]) + 1, 5)
    return Config(**payload)
