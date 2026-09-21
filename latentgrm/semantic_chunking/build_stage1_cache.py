#!/usr/bin/env python
"""Build the Semantic Chunking token cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latentgrm.semantic_chunking.structure import normalize_use
from latentgrm.semantic_chunking.stage1_cache import build_stage1_cache
from latentgrm.training.stage1.data import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compression-rate", type=int, default=8)
    parser.add_argument("--use", required=True)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.use = normalize_use(args.use)
    if args.compression_rate <= 0 or args.shard_size <= 0 or args.workers <= 0:
        parser.error("compression-rate, shard-size and workers must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|compress_token|>"]}
    )
    model_view = SimpleNamespace(
        tokenizer=tokenizer,
        decoder_name_or_path=args.tokenizer,
        compress_token_id=tokenizer.convert_tokens_to_ids("<|compress_token|>"),
        latent_token_ids=tokenizer(
            ["<think>", "</think>"], add_special_tokens=False
        )["input_ids"],
    )
    data = read_jsonl(args.data)
    manifest = build_stage1_cache(
        args.output,
        args.data,
        data,
        args.compression_rate,
        args.use,
        model_view,
        shard_size=args.shard_size,
        workers=args.workers,
    )
    print(json.dumps(manifest["stats"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
