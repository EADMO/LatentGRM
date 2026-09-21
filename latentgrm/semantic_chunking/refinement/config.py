"""Semantic Chunking boundary settings."""
from dataclasses import dataclass
import json
import math
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

    def __post_init__(self):
        if self.parser not in {"spacy", "rules"}:
            raise ValueError("parser must be spacy or the explicit rules ablation")
        # 5 is used only by the internal lexical-rescue retry. The persisted
        # default policy still starts at displacement=4.
        if type(self.displacement) is not int or not 0 <= self.displacement <= 5:
            raise ValueError("displacement must be an integer in [0, 5]")
        if (type(self.lexical_rescue_displacement) is not int
                or self.lexical_rescue_displacement not in {
                    self.displacement, self.displacement + 1
                }):
            raise ValueError(
                "lexical_rescue_displacement must equal displacement or displacement + 1"
            )
        if type(self.compact_span_tokens) is not int or self.compact_span_tokens < 1:
            raise ValueError("compact_span_tokens must be positive")
        for name in ("length_weight", "variance_weight", "displacement_weight",
                     "risk_weight", "semantic_margin", "sentence_reward"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}: {value}")


def load_config():
    path = os.environ.get("LATENTGRM_CHUNK_CONFIG")
    if not path:
        return Config()
    payload = json.loads(Path(path).read_text())
    if "displacement" in payload and "lexical_rescue_displacement" not in payload:
        payload["lexical_rescue_displacement"] = min(int(payload["displacement"]) + 1, 5)
    return Config(**payload)
