#!/usr/bin/env python
"""Stage 1 encoder, decoder, and joint training with Semantic Chunking."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import HfArgumentParser, set_seed

from latentgrm.semantic_chunking.arguments import DataArguments, ExtraArguments
from latentgrm.training.stage1.arguments import ModelArguments, Stage1TrainingArguments
from latentgrm.semantic_chunking.stage1_data import Stage1Dataset
from latentgrm.training.stage1.data import DataCollatorForDynamicPadding
from latentgrm.modeling.modeling_stage1 import (
    LatentSFTStage1Decoder,
    LatentSFTStage1Encoder,
    LatentSFTStage1Union,
)
from latentgrm.training.stage1.trainer import (
    Stage1DecoderTrainer,
    Stage1EncoderTrainer,
    Stage1UnionTrainer,
)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, ExtraArguments, Stage1TrainingArguments)
    )
    model_args, data_args, extra_args, training_args = parser.parse_args_into_dataclasses()
    if data_args.use == "semantic" and not data_args.stage1_cache_path:
        raise ValueError("Semantic Chunking training requires a Stage-1 cache")
    if data_args.stage1_cache_path:
        from latentgrm.semantic_chunking.stage1_cache import Stage1Cache
        # Fail before allocating the training models. The dataset additionally
        # checks the actual model tokenizer after model construction.
        Stage1Cache(data_args.stage1_cache_path, data_path=data_args.train_data_path,
                    compression_rate=data_args.compression_rate, use=data_args.use)
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
        and training_args.resume_from_checkpoint is None
    ):
        raise ValueError(f"Non-empty output directory: {training_args.output_dir}")

    logging.basicConfig(level=logging.INFO)
    set_seed(training_args.seed)
    model_cls = {
        "encoder": LatentSFTStage1Encoder,
        "decoder": LatentSFTStage1Decoder,
        "union": LatentSFTStage1Union,
    }[extra_args.stage]
    trainer_cls = {
        "encoder": Stage1EncoderTrainer,
        "decoder": Stage1DecoderTrainer,
        "union": Stage1UnionTrainer,
    }[extra_args.stage]
    model = model_cls(
        encoder_name_or_path=model_args.encoder_name_or_path,
        decoder_name_or_path=model_args.decoder_name_or_path,
        bfloat16=model_args.bfloat16,
        use_flash_attention_2=model_args.use_flash_attention_2,
        lora_tune=training_args.lora_tune,
        lora_path=training_args.lora_path,
        lora_rank=training_args.lora_rank,
        lora_dropout=training_args.lora_dropout,
        save_path=training_args.output_dir,
        topk_interpolation=model_args.topk_interpolation,
    )
    dataset = Stage1Dataset(data_args.train_data_path, data_args, model)
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForDynamicPadding(
            model.tokenizer.pad_token_id,
            model.compress_token_id,
            model.latent_token_ids[-1],
            mask_dtype=next(model.encoder.parameters()).dtype,
        ),
        processing_class=model.tokenizer,
    )
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()
    if trainer.is_world_process_zero():
        model.tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
