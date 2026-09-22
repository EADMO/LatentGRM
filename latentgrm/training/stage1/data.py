import json

import torch


def read_jsonl(input_file_path):
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line))
    return data

def build_latent_token_induction_mask(
    input_ids: torch.Tensor,
    special_token_ids: list[int],
    pad_token_id: int,
    dtype: torch.dtype | None = None,   # None → bool ；other → float/-inf
) -> torch.Tensor:
    """
    Generate latent token induction mask.
    - Shape: [B, 1, T, T]
    - If dtype is None: bool mask (True = keep, False = mask)
    - Else: float additive mask (keep = 0, mask = -inf), auto-cast to the given dtype
    """
    B, T = input_ids.shape
    device = input_ids.device

    # ---- 1. Classical Causal Attention (row ≥ col) ----
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))      # [T,T]
    causal = causal.unsqueeze(0).expand(B, -1, -1)                              # [B,T,T]

    # ---- 2. Prevent attending to earlier special tokens ----
    is_special = torch.isin(input_ids, torch.tensor(special_token_ids, device=device))  # [B,T]

    # row_gt_col[i,j] = True if i > j
    row_gt_col = torch.arange(T, device=device).view(-1, 1) > torch.arange(T, device=device).view(1, -1)  # [T,T]

    # Mask if key is special and i > j
    forbid = is_special.unsqueeze(1) & row_gt_col.unsqueeze(0)                  # [B,T,T]

    # ---- 3. padding ----
    not_pad = (input_ids != pad_token_id)
    pad_ok  = not_pad.unsqueeze(-1) & not_pad.unsqueeze(-2)                     # [B,T,T]

    # ---- 4. Final kept positions ----
    keep = causal & ~forbid & pad_ok                                            # bool [B,T,T]
    keep = keep.unsqueeze(1)                                                    # [B,1,T,T]

    # ---- 5. Output ----
    if dtype is None:                           # bool mask
        return keep
    else:                                       # additive mask
        add_mask = torch.zeros_like(keep, dtype=dtype)
        add_mask = add_mask.masked_fill(~keep,  torch.finfo(dtype).min)
        return add_mask

class DataCollatorForDynamicPadding:
    def __init__(self, pad_token_id, compress_token_id, latent_token_id_right=None, pad_to_multiple_of=None, mask_dtype=torch.bfloat16):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.compress_token_id = compress_token_id
        self.latent_token_id_right = latent_token_id_right
        self.mask_dtype = mask_dtype
    def __call__(self, examples):
        # print(examples)
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
        cot_ids = [torch.tensor(example["cot_ids"], dtype=torch.long) for example in examples]
        labels = [torch.tensor(example["labels"], dtype=torch.long) for example in examples]
        position_ids = [torch.tensor(example["position_ids"], dtype=torch.long) for example in examples]

        input_lengths = [len(sequence) for sequence in input_ids]
        input_ids = self.dynamic_padding(input_ids, fill_value=self.pad_token_id)
        attention_mask = torch.zeros_like(input_ids)
        for row, length in enumerate(input_lengths):
            attention_mask[row, :length] = 1

        cot_ids = self.dynamic_padding(cot_ids, fill_value=self.pad_token_id)
        cot_attention_mask = build_latent_token_induction_mask(
            cot_ids, [self.compress_token_id], self.pad_token_id, self.mask_dtype
        )

        labels = self.dynamic_padding(labels)
        position_ids = self.dynamic_padding(position_ids, fill_value=0)

        batch = {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "cot_ids": cot_ids,
            "cot_attention_mask": cot_attention_mask,
            "labels": labels,
            "position_ids": position_ids}
        return batch
    def dynamic_padding(self, sequences, fill_value=-100):
        max_length = max(len(x) for x in sequences)
        if self.pad_to_multiple_of:
            max_length = ((max_length - 1) // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of
        padded_sequences = torch.full((len(sequences), max_length), fill_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq
        return padded_sequences
