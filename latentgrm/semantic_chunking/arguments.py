from dataclasses import dataclass, field
from typing import Optional



@dataclass
class DataArguments:
    train_data_path: str = field(metadata={"help": "Canonical phase-1 JSONL"})
    compression_rate: int = field(default=8)
    use: str = field(default="semantic")
    stage1_cache_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional pre-tokenized deterministic Stage-1 cache"},
    )

    def __post_init__(self):
        from latentgrm.semantic_chunking.structure import normalize_use

        if self.compression_rate <= 0:
            raise ValueError("compression_rate must be positive")
        self.use = normalize_use(self.use)


@dataclass
class ExtraArguments:
    stage: str = field(default="encoder")

    def __post_init__(self):
        if self.stage not in {"encoder", "decoder", "union"}:
            raise ValueError(f"Unsupported stage: {self.stage}")
