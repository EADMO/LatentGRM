import logging
import os
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers.trainer import Trainer

logger = logging.getLogger(__name__)


class Stage2Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The wrapped model has ``**kwargs`` in forward, so Transformers 5.x
        # assumes it consumes ``num_items_in_batch`` and skips its normal
        # gradient-accumulation loss scaling.  Our custom loss is already a
        # mean and intentionally ignores num_items_in_batch; force Trainer to
        # apply the standard 1 / gradient_accumulation_steps normalization.
        self.model_accepts_loss_kwargs = False

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        decay_steps = getattr(self.args, "lr_decay_steps", None)
        if decay_steps is None:
            return super().create_scheduler(num_training_steps, optimizer=optimizer)
        if self.args.warmup_steps < 1 or decay_steps <= self.args.warmup_steps:
            raise ValueError("WarmupDecayLR requires 0 < warmup_steps < lr_decay_steps")
        if self.lr_scheduler is None:
            from deepspeed.runtime.lr_schedules import WarmupDecayLR
            self.lr_scheduler = WarmupDecayLR(
                optimizer if optimizer is not None else self.optimizer,
                total_num_steps=decay_steps,
                warmup_num_steps=self.args.warmup_steps,
                warmup_min_lr=0.0,
                warmup_max_lr=self.args.learning_rate,
                warmup_type="log",
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)

        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')

        if self.model.lora_tune:
            if self.is_world_process_zero():
                self.model.save(os.path.join(output_dir, 'lora_adapter'))

                # Merge LoRA into base model and save as HF format.
                model = AutoModelForCausalLM.from_pretrained(
                    os.path.join(self.model.save_path, 'base_model'),
                    torch_dtype=torch.bfloat16,
                    use_cache=False,
                    trust_remote_code=True
                )
                model = PeftModel.from_pretrained(
                    model,
                    os.path.join(output_dir, 'lora_adapter')
                )
                model = model.merge_and_unload()
                model.save_pretrained(os.path.join(output_dir, 'hf'))
                self.model.tokenizer.save_pretrained(os.path.join(output_dir, 'hf'))
                del model
        else:
            self.model.save_pretrained(os.path.join(output_dir, 'hf'))

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
