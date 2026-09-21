#!/usr/bin/env python
"""Export sparse latent targets using the training cache and joint encoder."""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latentgrm.semantic_chunking.structure import normalize_use
from latentgrm.semantic_chunking.stage1_cache import Stage1Cache
from latentgrm.semantic_chunking.stage1_data import (
    materialize_encoder_cot,
    prepare_encoder_cot,
)
from latentgrm.training.export_utils import (
    atomic_json_save,
    atomic_torch_save,
    build_latent_token_induction_mask,
    read_jsonl,
    right_pad_2d,
)
from latentgrm.modeling.modeling_stage1 import LatentSFTStage1Union
from latentgrm.modeling.utils import (
    get_input_embeddings,
    get_output_embeddings,
    latent_vocab_projection,
    latent_vocab_projection_topk,
)


class VariantGenerator:
    def __init__(
        self,
        encoder_model_path,
        decoder_model_path,
        lora_path,
        use,
        mp_size=1,
        dtype="bfloat16",
        compression_rate=8,
        topk_interpolation=10,
        full_vocab=False,
        batch_size=16,
        stage1_cache_path=None,
        data_path=None,
        num_records=None,
    ):
        self.encoder_model_path = encoder_model_path
        self.decoder_model_path = decoder_model_path
        self.lora_path = lora_path
        self.use = normalize_use(use)
        self.mp_size = mp_size
        self.dtype = dtype
        self.compression_rate = compression_rate
        self.topk_interpolation = topk_interpolation
        self.full_vocab = full_vocab
        self.batch_size = batch_size
        self.stage1_cache_path = stage1_cache_path
        ctx = mp.get_context("spawn")
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.processes = []
        for rank in range(mp_size):
            process = ctx.Process(
                target=self._worker,
                args=(
                    encoder_model_path,
                    decoder_model_path,
                    lora_path,
                    self.use,
                    rank,
                    self.input_queue,
                    self.output_queue,
                    compression_rate,
                    topk_interpolation,
                    full_vocab,
                    stage1_cache_path,
                    data_path,
                    num_records,
                ),
            )
            process.start()
            self.processes.append(process)

    @staticmethod
    def _worker(
        encoder_model_path,
        decoder_model_path,
        lora_path,
        use,
        rank,
        input_queue,
        output_queue,
        compression_rate,
        topk_interpolation,
        full_vocab,
        stage1_cache_path,
        data_path,
        num_records,
    ):
        torch.multiprocessing.set_sharing_strategy("file_system")
        device = torch.device(f"cuda:{rank}")
        model = LatentSFTStage1Union(
            encoder_name_or_path=encoder_model_path,
            decoder_name_or_path=decoder_model_path,
            lora_path=lora_path,
            lora_tune=True,
            bfloat16=True,
            use_flash_attention_2=False,
            training=False,
        ).to(device)
        model.eval()
        cache = None
        if stage1_cache_path:
            cache = Stage1Cache(
                stage1_cache_path,
                data_path=data_path,
                num_records=num_records,
                compression_rate=compression_rate,
                use=use,
                model=model,
            )
        pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
        with torch.no_grad():
            while True:
                task = input_queue.get()
                if task is None:
                    break
                batch_id, examples, cache_indices = task
                if cache is not None:
                    cot_ids = [
                        materialize_encoder_cot(cache[index], model)
                        for index in cache_indices
                    ]
                else:
                    cot_ids = [
                        prepare_encoder_cot(
                            example, model, encoder_model_path, compression_rate, use
                        )[0]
                        for example in examples
                    ]
                cot_tensor = right_pad_2d(cot_ids, fill_value=pad_id).to(device)
                mask = build_latent_token_induction_mask(
                    cot_tensor,
                    [model.compress_token_id],
                    pad_id,
                    dtype=torch.bfloat16,
                )
                hidden = model.encoder(cot_tensor, attention_mask=mask).last_hidden_state
                compress_mask = cot_tensor == model.compress_token_id
                decoder_inputs = get_input_embeddings(model.decoder)
                decoder_outputs = get_output_embeddings(model.decoder)
                states = []
                for row in range(len(cot_ids)):
                    indices = compress_mask[row].nonzero(as_tuple=False).squeeze(-1)
                    if full_vocab:
                        _, probs = latent_vocab_projection(
                            hidden[row, indices],
                            decoder_outputs,
                            decoder_inputs,
                            temperature=1.0,
                            use_cosine=False,
                        )
                        states.append(probs.cpu())
                    else:
                        _, probs, vocab_indices = latent_vocab_projection_topk(
                            hidden[row, indices],
                            decoder_outputs,
                            decoder_inputs,
                            top_k=topk_interpolation,
                            temperature=1.0,
                            use_cosine=False,
                        )
                        states.append((probs.cpu(), vocab_indices.cpu()))
                output_queue.put((batch_id, states))

    def generate(self, examples, start_index=0):
        tasks = [
            (
                start,
                None if self.stage1_cache_path else examples[start : start + self.batch_size],
                list(
                    range(
                        start_index + start,
                        start_index + min(start + self.batch_size, len(examples)),
                    )
                ),
            )
            for start in range(0, len(examples), self.batch_size)
        ]
        next_task = 0
        pending = 0
        results = []
        max_pending = self.mp_size * 4
        with tqdm(total=len(examples), desc=f"soft-labels/{self.use}") as progress:
            while len(results) < len(tasks):
                while next_task < len(tasks) and pending < max_pending:
                    self.input_queue.put(tasks[next_task])
                    next_task += 1
                    pending += 1
                try:
                    batch_id, shared = self.output_queue.get(timeout=30.0)
                except queue.Empty:
                    failed = [
                        (rank, process.pid, process.exitcode)
                        for rank, process in enumerate(self.processes)
                        if not process.is_alive()
                    ]
                    if failed:
                        raise RuntimeError(
                            "Soft-label worker exited before returning all batches: "
                            + ", ".join(
                                f"rank={rank} pid={pid} exitcode={exitcode}"
                                for rank, pid, exitcode in failed
                            )
                        )
                    continue
                clean = []
                for item in shared:
                    clean.append(
                        (item[0].clone(), item[1].clone())
                        if isinstance(item, tuple)
                        else item.clone()
                    )
                results.append((batch_id, clean))
                pending -= 1
                progress.update(len(clean))
        results.sort(key=lambda item: item[0])
        return [state for _, batch in results for state in batch]

    def close(self, timeout=30.0):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for _ in self.processes:
            self.input_queue.put(None)
        deadline = time.monotonic() + timeout
        for process in self.processes:
            process.join(max(0.0, deadline - time.monotonic()))
        forced = any(process.is_alive() for process in self.processes)
        if forced:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
            for process in self.processes:
                process.join(5.0)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(5.0)
        for process in self.processes:
            if not process.is_alive():
                process.close()
        for queue in (self.input_queue, self.output_queue):
            if forced:
                queue.cancel_join_thread()
                queue.close()
            else:
                queue.close()
                queue.join_thread()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_model_path", required=True)
    parser.add_argument("--decoder_model_path", required=True)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--use", required=True)
    parser.add_argument("--mp_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--compression_rate", type=int, default=8)
    parser.add_argument("--topk_interpolation", type=int, default=10)
    parser.add_argument("--stage1_cache_path")
    parser.add_argument("--full_vocab", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.use = normalize_use(args.use)
    if args.compression_rate <= 0 or args.chunk_size <= 0:
        parser.error("compression_rate and chunk_size must be positive")

    logging.basicConfig(level=logging.ERROR)
    data = read_jsonl(args.data_path)
    if args.use == "semantic" and not args.stage1_cache_path:
        parser.error("Semantic Chunking requires --stage1_cache_path")
    os.makedirs(args.save_path, exist_ok=True)
    manifest_path = os.path.join(args.save_path, "manifest.json")
    completed_path = os.path.join(args.save_path, ".completed")
    manifest = {
        "version": 3,
        "use": args.use,
        "data_path": os.path.realpath(args.data_path),
        "num_records": len(data),
        "chunk_size": args.chunk_size,
        "compression_rate": args.compression_rate,
        "topk_interpolation": args.topk_interpolation,
        "full_vocab": args.full_vocab,
        "dtype": args.dtype,
        "encoder": args.encoder_model_path,
        "decoder": args.decoder_model_path,
        "lora": args.lora_path,
    }
    atomic_json_save(manifest, manifest_path)

    missing = []
    for start in range(0, len(data), args.chunk_size):
        end = min(start + args.chunk_size, len(data))
        path = os.path.join(args.save_path, f"batch_{start}_{end}.pt")
        if not (args.resume and os.path.isfile(path)):
            missing.append((start, end, path))
    if not missing:
        Path(completed_path).touch()
        print("All soft labels are complete")
        return

    generator = VariantGenerator(
        args.encoder_model_path,
        args.decoder_model_path,
        args.lora_path,
        args.use,
        mp_size=args.mp_size,
        dtype=args.dtype,
        compression_rate=args.compression_rate,
        topk_interpolation=args.topk_interpolation,
        full_vocab=args.full_vocab,
        batch_size=args.batch_size,
        stage1_cache_path=args.stage1_cache_path,
        data_path=args.data_path,
        num_records=len(data),
    )
    try:
        for start, end, path in missing:
            states = generator.generate(data[start:end], start_index=start)
            if len(states) != end - start:
                raise RuntimeError("Soft-label count mismatch")
            atomic_torch_save(states, path)
    finally:
        generator.close()
    Path(completed_path).touch()
    print(f"Completed {len(data)} soft labels in {args.save_path}.", flush=True)
    # Avoid a multiprocessing/CUDA atexit hang after durable output completion.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
