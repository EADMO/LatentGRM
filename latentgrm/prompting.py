import math
from typing import Any

LATENT_SFT_MAX_LENGTH = 6144


def head_tail_truncate(token_ids: list[int], max_tokens: int) -> list[int]:
    """Keep both instruction/task headers and the final response/judgment."""
    if max_tokens < 0:
        raise ValueError(f"max_tokens must be non-negative, got {max_tokens}")
    if len(token_ids) <= max_tokens:
        return token_ids
    left = (max_tokens + 1) // 2
    right = max_tokens - left
    return token_ids[:left] + (token_ids[-right:] if right else [])


def truncate_stage1_tokens(
    prefix_ids: list[int],
    cot_ids: list[int],
    think_start_ids: list[int],
    think_end_ids: list[int],
    compression_rate: int,
    max_length: int = LATENT_SFT_MAX_LENGTH,
) -> tuple[list[int], list[int]]:
    """Fit the Stage-1 encoder sequence under a hard token limit.

    The encoder is the longest Stage-1 view because it contains explicit CoT
    tokens plus one compress placeholder per segment.  Prompt truncation keeps
    both ends so pairwise reward examples retain Response A and Response B.
    """
    if compression_rate <= 0:
        raise ValueError("compression_rate must be greater than 0")
    if not cot_ids:
        raise ValueError("cot_ids must not be empty")

    def encoded_length(prefix_len: int, cot_len: int) -> int:
        latent_count = math.ceil(cot_len / compression_rate)
        return (
            prefix_len
            + len(think_start_ids)
            + cot_len
            + latent_count
            + len(think_end_ids)
        )

    prefix_ids = list(prefix_ids)
    cot_ids = list(cot_ids)
    excess = encoded_length(len(prefix_ids), len(cot_ids)) - max_length
    if excess <= 0:
        return prefix_ids, cot_ids

    # Preserve at least a modest prompt frame when possible, then reduce CoT.
    prompt_floor = min(1024, len(prefix_ids))
    removable_prompt = len(prefix_ids) - prompt_floor
    remove = min(excess, removable_prompt)
    prefix_ids = head_tail_truncate(prefix_ids, len(prefix_ids) - remove)

    while encoded_length(len(prefix_ids), len(cot_ids)) > max_length:
        excess = encoded_length(len(prefix_ids), len(cot_ids)) - max_length
        if len(cot_ids) > 1:
            # Removing CoT tokens can also remove compress placeholders, so a
            # bounded loop is clearer and exact around segment boundaries.
            remove = min(excess, len(cot_ids) - 1)
            cot_ids = head_tail_truncate(cot_ids, len(cot_ids) - remove)
        elif prefix_ids:
            prefix_ids = head_tail_truncate(
                prefix_ids, max(0, len(prefix_ids) - excess)
            )
        else:
            raise ValueError(
                f"Structural tokens alone exceed max_length={max_length}"
            )
    return prefix_ids, cot_ids


def truncate_stage2_prefix(
    prefix_ids: list[int],
    latent_length: int,
    think_start_ids: list[int],
    think_end_ids: list[int],
    answer_ids: list[int],
    max_length: int = LATENT_SFT_MAX_LENGTH,
) -> list[int]:
    """Fit a Stage-2 teacher-forced sequence while preserving both prompt ends."""
    fixed_length = (
        latent_length
        + len(think_start_ids)
        + len(think_end_ids)
        + len(answer_ids)
    )
    budget = max_length - fixed_length
    if budget < 0:
        raise ValueError(
            "Latent chain and answer exceed the Stage-2 max length: "
            f"fixed={fixed_length}, max={max_length}"
        )
    return head_tail_truncate(list(prefix_ids), budget)


def build_qwen_generation_prefix(
    tokenizer: Any,
    example: dict[str, Any],
) -> str:
    """Build the assistant prefix shared by every Qwen training stage."""
    messages = [{"role": "user", "content": example["problem"]}]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
