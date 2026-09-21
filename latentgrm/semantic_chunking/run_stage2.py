#!/usr/bin/env python
"""Train the latent judge from joint-encoder targets and the merged decoder."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import HfArgumentParser, set_seed

from latentgrm.modeling.modeling_stage2 import LatentSFTStage2SoftEmbedding
from latentgrm.training.stage2.arguments import DataArguments, ModelArguments, Stage2TrainingArguments
from latentgrm.training.stage2.data import DataCollatorForDynamicPadding, Stage2Dataset
from latentgrm.training.stage2.trainer import Stage2Trainer


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, Stage2TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if (training_args.lr_decay_steps is not None and training_args.deepspeed
            and 'scheduler' in training_args.hf_deepspeed_config.config):
        raise ValueError('Stage 2 supplies its scheduler through the Trainer. Use train.py, or omit the scheduler section from the DeepSpeed configuration.')
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
        and training_args.resume_from_checkpoint is None
    ):
        raise ValueError(f"Non-empty output directory: {training_args.output_dir}")
    set_seed(training_args.seed)
    model = LatentSFTStage2SoftEmbedding(
        latent_model_path=model_args.latent_model_path,
        ce_w=model_args.ce_w,
        kl_w=model_args.kl_w,
        bfloat16=model_args.bfloat16,
        use_flash_attention_2=model_args.use_flash_attention_2,
        lora_tune=training_args.lora_tune,
        lora_path=training_args.lora_path,
        lora_rank=training_args.lora_rank,
        lora_dropout=training_args.lora_dropout,
        save_path=training_args.output_dir,
        training=training_args.training,
        topk_interpolation=model_args.topk_interpolation,
    )
    dataset = Stage2Dataset(
        path=data_args.train_data_path,
        train_latent_soft_label_path=data_args.train_latent_soft_label_path,
        args=data_args,
        model=model,
        add_gumbel_noise=data_args.add_gumbel_noise,
        gumbel_temperature=data_args.gumbel_temperature,
        noise_scale=data_args.noise_scale,
    )
    trainer = Stage2Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForDynamicPadding(model.tokenizer.pad_token_id),
        processing_class=model.tokenizer,
    )
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()
