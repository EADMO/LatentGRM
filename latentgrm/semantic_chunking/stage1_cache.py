"""Cache tokenization and Semantic Chunking boundaries for Stage 1.

Only deterministic preprocessing is cached.  ``supervision_frontier`` remains
sampled in ``Stage1Dataset.__getitem__`` so caching does not change the training
objective.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import multiprocessing
from pathlib import Path

import torch

from latentgrm.semantic_chunking.structure import (
    CHUNK_RANGE_TAG,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    PREFERRED_MAX_CHUNK_TOKENS,
    PREFERRED_MIN_CHUNK_TOKENS,
    PROTECTED_ATOM_POLICY,
    normalize_use,
)
from latentgrm.semantic_chunking.stage1_data import prepare_stage1_record
from latentgrm.prompting import LATENT_SFT_MAX_LENGTH


CACHE_VERSION = 5
SEQUENCE_FIELDS = ("prefix_ids", "suffix_ids", "explicit_ids", "frontier_ends")


def _initialize_worker(model, rate, use):
    global _worker_model, _worker_rate, _worker_use
    torch.set_num_threads(1)
    _worker_model, _worker_rate, _worker_use = model, rate, use


def _prepare_worker(example):
    return prepare_stage1_record(example, _worker_model, _worker_rate, _worker_use)


def expected_manifest(
    data_path: str,
    num_records: int,
    compression_rate: int,
    use: str,
    model,
    shard_size: int,
) -> dict:
    return {
        "version": CACHE_VERSION,
        "num_records": int(num_records),
        "compression_rate": int(compression_rate),
        "use": normalize_use(use),
        "min_chunk_tokens": MIN_CHUNK_TOKENS,
        "max_chunk_tokens": MAX_CHUNK_TOKENS,
        "preferred_min_chunk_tokens": PREFERRED_MIN_CHUNK_TOKENS,
        "preferred_max_chunk_tokens": PREFERRED_MAX_CHUNK_TOKENS,
        "chunk_range_tag": CHUNK_RANGE_TAG,
        "protected_atom_policy": PROTECTED_ATOM_POLICY,
        "max_length": LATENT_SFT_MAX_LENGTH,
        "latent_left_ids": list(model.latent_token_ids[0]),
        "latent_right_ids": list(model.latent_token_ids[1]),
        "shard_size": int(shard_size),
    }


def _pack_sequences(sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    values = []
    for sequence in sequences:
        values.extend(sequence)
        offsets.append(len(values))
    return (
        torch.tensor(values, dtype=torch.int32),
        torch.tensor(offsets, dtype=torch.int64),
    )


def _save_shard(records: list[dict], path: Path) -> None:
    payload = {}
    for field in SEQUENCE_FIELDS:
        values, offsets = _pack_sequences([record[field] for record in records])
        payload[f"{field}_values"] = values
        payload[f"{field}_offsets"] = offsets
    if any("plan_metadata" in record for record in records):
        payload["plan_metadata"] = [record.get("plan_metadata", {}) for record in records]
    torch.save(payload, path)


def _update_stats(stats: dict, record: dict) -> None:
    explicit_length = len(record["explicit_ids"])
    frontiers = record["frontier_ends"]
    previous = 0
    for end in frontiers:
        size = end - previous
        if size <= 0:
            raise ValueError((previous, end))
        stats["chunk_histogram"][size] += 1
        previous = end
    if previous != explicit_length:
        raise ValueError((previous, explicit_length))
    stats["explicit_tokens"] += explicit_length
    stats["latent_tokens"] += len(frontiers)
    stats["semantic_segments"] += int(record["semantic_segment_count"])
    metadata = record.get("plan_metadata", {})
    if metadata.get("version") == 2:
        stats["semantic"]["records"] += 1
        stats["semantic"]["parser_fallback_records"] += int(bool(metadata["parser"]["fallback"]))
        for segment in metadata["segments"]:
            stats["semantic"]["segments"] += 1
            stats["semantic"]["accepted_segments"] += int(segment["accepted"])
            for key in ("moved_frontiers", "baseline_semantic_cost", "final_semantic_cost",
                        "baseline_risk2", "final_risk2", "baseline_intra_word", "final_intra_word"):
                stats["semantic"][key] += segment[key]


def _finalize_stats(stats: dict, compression_rate: int, records: int) -> dict:
    histogram = stats["chunk_histogram"]
    total_chunks = sum(histogram.values())
    exact = histogram.get(compression_rate, 0)
    below = sum(count for size, count in histogram.items() if size < compression_rate)
    above = sum(count for size, count in histogram.items() if size > compression_rate)
    preferred_below = sum(
        count for size, count in histogram.items()
        if size < PREFERRED_MIN_CHUNK_TOKENS
    )
    preferred_above = sum(
        count for size, count in histogram.items()
        if size > PREFERRED_MAX_CHUNK_TOKENS
    )
    preferred_outside = preferred_below + preferred_above
    return {
        "records": records,
        "semantic": dict(stats.get("semantic", {})),
        "explicit_tokens": stats["explicit_tokens"],
        "latent_tokens": stats["latent_tokens"],
        "semantic_segments": stats["semantic_segments"],
        "mean_explicit_tokens": stats["explicit_tokens"] / records,
        "mean_latent_tokens": stats["latent_tokens"] / records,
        "mean_semantic_segments": stats["semantic_segments"] / records,
        "effective_tokens_per_latent": stats["explicit_tokens"] / total_chunks,
        "min_chunk_tokens": min(histogram),
        "max_chunk_tokens": max(histogram),
        "chunks_equal_rate": exact,
        "chunks_below_rate": below,
        "chunks_above_rate": above,
        "fraction_equal_rate": exact / total_chunks,
        "fraction_below_rate": below / total_chunks,
        "fraction_above_rate": above / total_chunks,
        "preferred_min_chunk_tokens": PREFERRED_MIN_CHUNK_TOKENS,
        "preferred_max_chunk_tokens": PREFERRED_MAX_CHUNK_TOKENS,
        "chunks_below_preferred_range": preferred_below,
        "chunks_above_preferred_range": preferred_above,
        "chunks_outside_preferred_range": preferred_outside,
        "chunks_within_preferred_range": total_chunks - preferred_outside,
        "fraction_outside_preferred_range": preferred_outside / total_chunks,
        "chunk_histogram": {str(k): histogram[k] for k in sorted(histogram)},
    }


def build_stage1_cache(
    cache_path: str,
    data_path: str,
    data: list[dict],
    compression_rate: int,
    use: str,
    model,
    shard_size: int = 1000,
    workers: int = 1,
) -> dict:
    """Build a sharded token cache, or reuse an existing one."""
    cache_dir = Path(cache_path)
    if not data or shard_size <= 0 or workers <= 0:
        raise ValueError("A nonempty dataset and positive shard size/workers are required")
    expected = expected_manifest(
        data_path, len(data), compression_rate, use, model, shard_size
    )
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"Reusing Stage-1 cache: {cache_dir}")
        return manifest
    if cache_dir.exists():
        raise RuntimeError(f"Incomplete Stage-1 cache directory: {cache_dir}")

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_dir.with_name(f"{cache_dir.name}.tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    shards = []
    stats = {
        "explicit_tokens": 0,
        "latent_tokens": 0,
        "semantic_segments": 0,
        "chunk_histogram": Counter(),
        "semantic": Counter(),
    }
    pool_context = (
        ProcessPoolExecutor(max_workers=workers,
                            mp_context=multiprocessing.get_context("spawn"),
                            initializer=_initialize_worker,
                            initargs=(model, compression_rate, normalize_use(use)))
        if workers > 1 else nullcontext(None)
    )
    try:
        with pool_context as pool:
            for start in range(0, len(data), shard_size):
                end = min(start + shard_size, len(data))
                if pool:
                    records = list(pool.map(_prepare_worker, data[start:end], chunksize=4))
                else:
                    records = [prepare_stage1_record(example, model, compression_rate, normalize_use(use))
                               for example in data[start:end]]
                for record in records:
                    _update_stats(stats, record)
                filename = f"batch_{start}_{end}.pt"
                _save_shard(records, temporary / filename)
                shards.append({"file": filename, "start": start, "end": end})
                print(f"Cached Stage-1 records {start}:{end}", flush=True)

        manifest = {
            **expected,
            "shards": shards,
            "stats": _finalize_stats(stats, compression_rate, len(data)),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, cache_dir)
        print(f"Built Stage-1 cache: {cache_dir}")
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class Stage1Cache:
    """Random-access reader backed by memory-mapped packed int32 tensors."""

    def __init__(
        self,
        cache_path: str,
        *,
        data_path: str | None = None,
        num_records: int | None = None,
        compression_rate: int | None = None,
        use: str | None = None,
        model=None,
    ):
        self.cache_dir = Path(cache_path)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Stage-1 cache manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.shards = self.manifest["shards"]
        self.shard_size = self.manifest["shard_size"]
        self._loaded_shards = {}

    def __len__(self):
        return self.manifest["num_records"]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_loaded_shards"] = {}
        return state

    def _load_shard(self, shard_index: int) -> dict:
        if shard_index not in self._loaded_shards:
            path = self.cache_dir / self.shards[shard_index]["file"]
            self._loaded_shards[shard_index] = torch.load(
                path, map_location="cpu", mmap=True, weights_only=True
            )
        return self._loaded_shards[shard_index]

    @staticmethod
    def _sequence(payload: dict, field: str, row: int) -> list[int]:
        offsets = payload[f"{field}_offsets"]
        start = int(offsets[row])
        end = int(offsets[row + 1])
        return payload[f"{field}_values"][start:end].tolist()

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = index // self.shard_size
        shard_meta = self.shards[shard_index]
        row = index - shard_meta["start"]
        if not 0 <= row < shard_meta["end"] - shard_meta["start"]:
            raise IndexError((index, shard_meta))
        payload = self._load_shard(shard_index)
        record = {
            field: self._sequence(payload, field, row)
            for field in SEQUENCE_FIELDS
        }
        if "plan_metadata" in payload:
            record["plan_metadata"] = payload["plan_metadata"][row]
        return record
