import logging
from dataclasses import dataclass
from typing import Optional
import os
import torch
import torch.distributed as dist
from torch import nn, Tensor
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers.file_utils import ModelOutput
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

from latentgrm.modeling.utils import (
    get_input_embeddings,
    get_output_embeddings,
    latent_vocab_projection_topk,
)


logger = logging.getLogger(__name__)

@dataclass
class CompressOutput(ModelOutput):
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None

def _is_main_process() -> bool:
    """Return True when running in a non-distributed context or on the rank-0 process."""
    return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)


def _get_decoder_embed_tokens(decoder):
    """Backward-compatible alias for the shared embedding accessor."""
    return get_input_embeddings(decoder)


def _mean_ce_per_example(logits: Tensor, labels: Tensor) -> Tensor:
    """Apply Eq. (4)'s per-suffix mean before averaging the batch."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
    token_losses = F.cross_entropy(
        shift_logits.transpose(1, 2), shift_labels,
        ignore_index=-100, reduction="none",
    )
    valid = shift_labels.ne(-100)
    counts = valid.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Every LTSuM view must contain at least one target token")
    return ((token_losses * valid).sum(dim=1) / counts).mean()


class LatentSFTStage1Encoder(nn.Module):
    """Stage-1 Encoder training wrapper.

    Architecture
    ------------
    * ``encoder`` — an AutoModel loaded from ``encoder_name_or_path`` and extended
      with a single learnable ``<|compress_token|>``. Its hidden states at the
      compress positions are projected onto the decoder vocabulary with its
      LM output head W, then reconstructed from its input embedding basis E
      (see ``latent_vocab_projection_topk``).
    * ``decoder`` — a frozen AutoModelForCausalLM used only to compute the
      language modeling loss. Its embedding table provides the basis for the
      superposition embedding that replaces latent-token positions.

    LoRA target modules
    -------------------
    ``embed_tokens`` is intentionally included in ``target_modules`` because the
    tokenizer adds a new ``<|compress_token|>`` and the encoder therefore needs
    to learn a representation for it; the LoRA adapters on ``embed_tokens``
    provide the trainable parameters for this new token while the rest of the
    vocabulary stays close to the pretrained embedding.
    """

    def __init__(self,
        encoder_name_or_path: str = None,
        decoder_name_or_path: str = None,
        bfloat16: bool = False,
        use_flash_attention_2: bool = False,
        lora_tune: bool = False,
        lora_path: str = None,
        lora_rank: int = 32,
        lora_dropout: float = 0.1,
        save_path: str = None,
        topk_interpolation: int = 5,
        training: bool = True,
        **kwargs
    ):
        super().__init__()
        self.encoder_name_or_path = encoder_name_or_path
        self.decoder_name_or_path = decoder_name_or_path
        # NOTE: we intentionally use sdpa instead of flash-attention-2 here.
        # The training code relies on a custom (non-causal) attention mask for
        # the latent-token induction / supervision masks, which flash-attn-2
        # does not support.
        self.encoder = AutoModel.from_pretrained(
            self.encoder_name_or_path,
            attn_implementation='sdpa',
            use_cache=False,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name_or_path)
        self.compress_token = '<|compress_token|>'
        self.topk_interpolation = topk_interpolation

        if training:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            special_tokens_dict = {'additional_special_tokens': [self.compress_token]}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.encoder.resize_token_embeddings(len(self.tokenizer))

        # Persist the base encoder (vocab-resized, pre-LoRA) exactly once on
        # the main process, and skip it when a previous run already wrote it
        # so that resuming training does not silently overwrite it.
        if training and _is_main_process() and save_path is not None:
            base_model_dir = os.path.join(save_path, 'base_model')
            if not os.path.exists(base_model_dir) or not os.listdir(base_model_dir):
                self.save(base_model_dir)

        self.compress_token_id = self.tokenizer.convert_tokens_to_ids(self.compress_token)

        self.latent_token_ids = self.tokenizer(['<think>','</think>'], add_special_tokens=False)['input_ids']

        self.lora_tune = lora_tune
        self.save_path = save_path

        if lora_tune:
            if lora_path is not None:
                self.encoder = PeftModel.from_pretrained(
                    self.encoder, lora_path
            )
            else:
                lora_cfg = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank * 2,
                    # ``embed_tokens`` is included so that LoRA can learn the
                    # newly added ``<|compress_token|>`` embedding.
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj","embed_tokens"],
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=TaskType.FEATURE_EXTRACTION,
                )

                self.encoder = get_peft_model(self.encoder, lora_cfg)


        self.decoder = AutoModelForCausalLM.from_pretrained(  self.decoder_name_or_path,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16,
            use_cache=True,
            trust_remote_code=True
        )
        self.init_decoder()

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def freeze_model(self, model):
        for _, param in model.named_parameters():
            param.requires_grad = False

    def print_trainable_parameters(self, model):
        trainable_parameters = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_parameters += param.numel()
        print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")

    def init_decoder(self):
        # The decoder participates only in forward passes (frozen); no need to
        # enable gradient checkpointing on it.
        self.freeze_model(self.decoder)
        self.decoder.eval()

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.config.use_cache = False
        self.encoder.enable_input_require_grads()
        self.encoder.gradient_checkpointing_enable(**kwargs)


    def _compress(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):

        compress_outputs = self.encoder(input_ids, attention_mask = attention_mask)

        return compress_outputs.last_hidden_state


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        cot_ids: torch.LongTensor = None,
        cot_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ):

        compress_embedding = self._compress(cot_ids, cot_attention_mask)
        bs, _, _ = compress_embedding.shape

        compress_mask = (cot_ids ==  self.compress_token_id)

        decoder_mask = (input_ids == -100)  #bs:seq torch.BoolTensor
        # Do not mutate the caller's tensor in place; Trainer may reuse the
        # same batch across multiple forwards (e.g. gradient accumulation or
        # loss recomputation), in which case an in-place write would erase the
        # -100 markers for subsequent passes.
        input_ids = input_ids.clone()
        input_ids[decoder_mask] = self.tokenizer.pad_token_id # Temporarily use pad_token_id as a placeholder
        decoder_embed_tokens = _get_decoder_embed_tokens(self.decoder)
        decoder_lm_head = get_output_embeddings(self.decoder)
        inputs_embeds = decoder_embed_tokens(input_ids)  # [bs, seq, h]

        bs, seq_len, h = inputs_embeds.shape

        # === Scatter new_b back to [seq, h], stacking sample-wise ===
        emb_list = []
        for b in range(bs):
            compress_idx = compress_mask[b].nonzero(as_tuple=False).squeeze(-1)   # [m]
            decoder_idx  = decoder_mask[b].nonzero(as_tuple=False).squeeze(-1)    # [m]
            if compress_idx.numel() == 0:
                # No replacement needed; use the base directly
                emb_list.append(inputs_embeds[b])
                continue

            if decoder_idx.numel() > compress_idx.numel():
                raise ValueError(
                    f"Sample {b}: frontier needs {decoder_idx.numel()} latents, "
                    f"encoder produced {compress_idx.numel()}"
                )
            compress_idx = compress_idx[:decoder_idx.numel()]

            # 1) Extract the segment to be mapped to the vocabulary (from encoder, requires gradient flow)
            x_b = compress_embedding[b, compress_idx]          # [m, h]

            # 2) alpha=softmax(W h) from the LM head, then z=E alpha
            # from the input-embedding basis.
            new_b, _, _ = latent_vocab_projection_topk(
                x_b,
                decoder_lm_head,
                decoder_embed_tokens,
                self.topk_interpolation,
                temperature=1.0,
                use_cosine=False
            )                                                  # [m, h] requires gradient
            new_b = new_b.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

            # 3) Use scatter with row indices to write new_b back into [seq, h]
            idx_exp = decoder_idx.unsqueeze(-1).expand(-1, h)  # [m, h]
            replaced = inputs_embeds[b].scatter(0, idx_exp, new_b)      # [seq, h]

            emb_list.append(replaced)

        inputs_embeds = torch.stack(emb_list, dim=0)           # [bs, seq, h]

        decoder_outputs = self.decoder(inputs_embeds=inputs_embeds,
            attention_mask = attention_mask,
            position_ids = position_ids
        )

        logits = decoder_outputs.logits

        loss = None
        if labels is not None:
            loss = _mean_ce_per_example(logits, labels)

        return CompressOutput(
            loss=loss,
            logits=logits,
        )

    def save(self, output_dir: str):
        state_dict = self.encoder.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()})
        self.encoder.save_pretrained(output_dir, state_dict=state_dict)

class LatentSFTStage1Decoder(nn.Module):
    def __init__(self,
        encoder_name_or_path: str = None,
        decoder_name_or_path: str = None,
        bfloat16: bool = False,
        use_flash_attention_2: bool = False,
        lora_tune: bool = False,
        lora_path: str = None,
        lora_rank: int = 32,
        lora_dropout: float = 0.1,
        save_path: str = None,
        topk_interpolation: int = 10,
        training: bool = True,
        **kwargs
    ):
        super().__init__()
        self.encoder_name_or_path = encoder_name_or_path
        self.decoder_name_or_path = decoder_name_or_path
        # NOTE: keep sdpa here; the custom latent induction mask is not
        # compatible with flash-attention-2.
        self.encoder = AutoModel.from_pretrained(
            self.encoder_name_or_path,
            attn_implementation='sdpa',
            use_cache=False,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16
        )
        self.freeze_model(self.encoder)
        self.encoder.eval()
        self.topk_interpolation = topk_interpolation

        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name_or_path )
        self.compress_token = '<|compress_token|>'
        self.decoder = AutoModelForCausalLM.from_pretrained(  self.decoder_name_or_path,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16,
            use_cache=True,
            trust_remote_code=True
        )

        # Persist the base decoder exactly once on the main process; skip if a
        # previous run already wrote it so that resume does not silently
        # overwrite it.
        if training and _is_main_process() and save_path is not None:
            base_model_dir = os.path.join(save_path, 'base_model')
            if not os.path.exists(base_model_dir) or not os.listdir(base_model_dir):
                self.save(base_model_dir)

        self.compress_token_id = self.tokenizer.convert_tokens_to_ids(self.compress_token)
        self.latent_token_ids = self.tokenizer(['<think>','</think>'], add_special_tokens=False)['input_ids']

        self.lora_tune = lora_tune
        self.save_path = save_path

        if lora_tune:
            if lora_path is not None:
                self.decoder = PeftModel.from_pretrained(
                    self.decoder, lora_path
            )
            else:
                lora_cfg = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank * 2,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )

                self.decoder = get_peft_model(self.decoder, lora_cfg)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)


    def freeze_model(self, model):
        for _, param in model.named_parameters():
            param.requires_grad = False

    def print_trainable_parameters(self, model):
        trainable_parameters = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_parameters += param.numel()
        print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")

    def gradient_checkpointing_enable(self, **kwargs):
        self.decoder.config.use_cache = False
        self.decoder.enable_input_require_grads()
        self.decoder.gradient_checkpointing_enable(**kwargs)

    def _compress(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):

        compress_outputs = self.encoder(input_ids, attention_mask=attention_mask)

        return compress_outputs.last_hidden_state


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        cot_ids: torch.LongTensor = None,
        cot_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ):

        compress_embedding = self._compress(cot_ids, cot_attention_mask)
        bs, _, _ = compress_embedding.shape

        compress_mask = (cot_ids ==  self.compress_token_id) # bs:seq torch.BoolTensor

        decoder_mask = (input_ids == -100)  #bs:seq torch.BoolTensor

        # Clone to avoid mutating the caller's tensor across repeated forwards.
        input_ids = input_ids.clone()
        input_ids[decoder_mask] = self.tokenizer.pad_token_id
        decoder_embed_tokens = _get_decoder_embed_tokens(self.decoder)
        decoder_lm_head = get_output_embeddings(self.decoder)
        inputs_embeds = decoder_embed_tokens(input_ids)

        bs, seq_len, h = inputs_embeds.shape

        emb_list = []
        for b in range(bs):
            compress_idx = compress_mask[b].nonzero(as_tuple=False).squeeze(-1)   # [m]
            decoder_idx  = decoder_mask[b].nonzero(as_tuple=False).squeeze(-1)    # [m]
            if compress_idx.numel() == 0:
                emb_list.append(inputs_embeds[b])
                continue

            if decoder_idx.numel() > compress_idx.numel():
                raise ValueError(
                    f"Sample {b}: frontier needs {decoder_idx.numel()} latents, "
                    f"encoder produced {compress_idx.numel()}"
                )
            compress_idx = compress_idx[:decoder_idx.numel()]

            new_b, _, _ = latent_vocab_projection_topk(
                compress_embedding[b, compress_idx]  ,
                decoder_lm_head,
                decoder_embed_tokens,
                top_k=self.topk_interpolation,
                temperature=1.0,
                use_cosine=False
            )                                                  # [m, h]
            new_b = new_b.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

            idx_exp = decoder_idx.unsqueeze(-1).expand(-1, h)  # [m, h]
            replaced = inputs_embeds[b].scatter(0, idx_exp, new_b)      # [seq, h]

            emb_list.append(replaced)

        inputs_embeds = torch.stack(emb_list, dim=0)           # [bs, seq, h]

        decoder_outputs = self.decoder(inputs_embeds=inputs_embeds,
            attention_mask = attention_mask,
            position_ids = position_ids
        )

        logits = decoder_outputs.logits

        loss = None
        if labels is not None:
            loss = _mean_ce_per_example(logits, labels)

        return CompressOutput(
            loss=loss,
            logits=logits,
        )

    def save(self, output_dir: str):
        state_dict = self.decoder.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()})
        self.decoder.save_pretrained(output_dir, state_dict=state_dict)


class LatentSFTStage1Union(nn.Module):
    def __init__(self,
        encoder_name_or_path: str = None,
        decoder_name_or_path: str = None,
        bfloat16: bool = False,
        use_flash_attention_2: bool = False,
        lora_tune: bool = False,
        lora_path: str = None,
        lora_rank: int = 32,
        lora_dropout: float = 0.1,
        save_path: str = None,
        topk_interpolation: int = 10,
        training: bool = True,
        **kwargs
    ):
        super().__init__()
        self.encoder_name_or_path = encoder_name_or_path
        self.decoder_name_or_path = decoder_name_or_path
        # NOTE: keep sdpa here; the custom latent induction mask is not
        # compatible with flash-attention-2.
        self.encoder = AutoModel.from_pretrained(
            self.encoder_name_or_path,
            attn_implementation='sdpa',
            use_cache=False,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name_or_path )
        self.compress_token = '<|compress_token|>'
        self.decoder = AutoModelForCausalLM.from_pretrained(  self.decoder_name_or_path,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16,
            use_cache=True,
            trust_remote_code=True
        )
        self.topk_interpolation = topk_interpolation

        self.compress_token_id = self.tokenizer.convert_tokens_to_ids(self.compress_token)
        self.latent_token_ids = self.tokenizer(['<think>','</think>'], add_special_tokens=False)['input_ids']

        self.lora_tune = lora_tune
        self.save_path = save_path


        if lora_tune:
            if lora_path is not None:
                self.encoder = PeftModel.from_pretrained(
                    self.encoder, os.path.join(lora_path,'encoder_weight')
            )
                self.decoder = PeftModel.from_pretrained(
                    self.decoder, os.path.join(lora_path,'decoder_weight')
            )

            else:
                encoder_lora_cfg = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank * 2,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj","embed_tokens"],
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=TaskType.FEATURE_EXTRACTION,
                )

                self.encoder = get_peft_model(self.encoder, encoder_lora_cfg)

                decoder_lora_cfg = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank * 2,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )

                self.decoder = get_peft_model(self.decoder, decoder_lora_cfg)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def freeze_model(self, model):
        for _, param in model.named_parameters():
            param.requires_grad = False

    def print_trainable_parameters(self, model):
        trainable_parameters = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_parameters += param.numel()
        print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.config.use_cache = False
        self.encoder.enable_input_require_grads()
        self.encoder.gradient_checkpointing_enable(**kwargs)

        self.decoder.config.use_cache = False
        self.decoder.enable_input_require_grads()
        self.decoder.gradient_checkpointing_enable(**kwargs)

    def _compress(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):

        compress_outputs = self.encoder(input_ids, attention_mask = attention_mask)

        return compress_outputs.last_hidden_state

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        cot_ids: torch.LongTensor = None,
        cot_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ):

        compress_embedding = self._compress(cot_ids, cot_attention_mask)
        bs, _, _ = compress_embedding.shape

        compress_mask = (cot_ids ==  self.compress_token_id) # bs:seq torch.BoolTensor

        decoder_mask = (input_ids == -100)  #bs:seq torch.BoolTensor

        # Clone to avoid mutating the caller's tensor across repeated forwards.
        input_ids = input_ids.clone()
        input_ids[decoder_mask] = self.tokenizer.pad_token_id
        decoder_embed_tokens = _get_decoder_embed_tokens(self.decoder)
        decoder_lm_head = get_output_embeddings(self.decoder)
        base = decoder_embed_tokens(input_ids) # [bs, seq, h], requires_grad=False
        inputs_embeds = base.clone()

        bs, seq_len, h = inputs_embeds.shape

        emb_list = []
        for b in range(bs):
            compress_idx = compress_mask[b].nonzero(as_tuple=False).squeeze(-1)   # [m]
            decoder_idx  = decoder_mask[b].nonzero(as_tuple=False).squeeze(-1)    # [m]

            if decoder_idx.numel() > compress_idx.numel():
                raise ValueError(
                    f"Sample {b}: frontier needs {decoder_idx.numel()} latents, "
                    f"encoder produced {compress_idx.numel()}"
                )
            compress_idx = compress_idx[:decoder_idx.numel()]

            new_b, _, _ = latent_vocab_projection_topk(
                compress_embedding[b, compress_idx]  ,
                decoder_lm_head,
                decoder_embed_tokens,
                top_k=self.topk_interpolation,
                temperature=1.0,
                use_cosine=False
            )                                                  # [m, h]
            new_b = new_b.to(device=base.device, dtype=base.dtype)

            idx_exp = decoder_idx.unsqueeze(-1).expand(-1, h)  # [m, h]
            replaced = base[b].scatter(0, idx_exp, new_b)      # [seq, h]

            emb_list.append(replaced)

        inputs_embeds = torch.stack(emb_list, dim=0)           # [bs, seq, h]

        decoder_outputs = self.decoder(inputs_embeds=inputs_embeds,
            attention_mask = attention_mask,
            position_ids = position_ids

        )

        logits = decoder_outputs.logits

        loss = None
        if labels is not None:
            loss = _mean_ce_per_example(logits, labels)

        return CompressOutput(
            loss=loss,
            logits=logits,
        )
    def save(self, output_dir: str):
        state_dict = self.encoder.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()})
        self.encoder.save_pretrained(output_dir+'/encoder_weight', state_dict=state_dict)

        state_dict = self.decoder.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()})
        self.decoder.save_pretrained(output_dir+'/decoder_weight', state_dict=state_dict)
