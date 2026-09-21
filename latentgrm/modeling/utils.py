import torch
import torch.nn.functional as F
from torch import nn


def get_input_embeddings(model) -> nn.Module:
    """Return input embeddings through the Transformers/PEFT public API."""
    embeddings = model.get_input_embeddings()
    if embeddings is None:
        raise ValueError(f"{type(model).__name__} does not expose input embeddings")
    return embeddings


def get_output_embeddings(model) -> nn.Module:
    """Return the LM output projection through the Transformers/PEFT public API."""
    embeddings = model.get_output_embeddings()
    if embeddings is None:
        raise ValueError(f"{type(model).__name__} does not expose output embeddings")
    return embeddings


def latent_vocab_projection(
    hidden_states: torch.Tensor,
    output_embeddings: nn.Module,
    input_embeddings: nn.Module,
    temperature: float = 1.0,
    use_cosine: bool = False,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute alpha=softmax(W h) and reconstruct z=E alpha over the full vocabulary."""
    projection_weight, basis_weight = _validate_vocab_matrices(
        output_embeddings, input_embeddings
    )
    logits = _vocab_logits(
        hidden_states,
        projection_weight,
        temperature=temperature,
        use_cosine=use_cosine,
        eps=eps,
    )
    probs = torch.softmax(logits.float(), dim=-1).to(
        dtype=basis_weight.dtype,
        device=basis_weight.device,
    )
    latent = probs @ basis_weight.detach()
    return latent, probs


def latent_vocab_projection_topk(
    hidden_states: torch.Tensor,
    output_embeddings: nn.Module,
    input_embeddings: nn.Module,
    top_k: int = 50,
    temperature: float = 1.0,
    use_cosine: bool = False,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Top-k approximation of alpha=softmax(W h), followed by z=E alpha.

    ``output_embeddings`` is the language-model head W used to select and
    score vocabulary items. ``input_embeddings`` is the vocabulary basis E
    used to reconstruct the latent embedding. They may be the same module for
    tied models, but must remain distinct for untied models such as Qwen3-8B.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be greater than 0, got {top_k}")

    projection_weight, basis_weight = _validate_vocab_matrices(
        output_embeddings, input_embeddings
    )
    logits = _vocab_logits(
        hidden_states,
        projection_weight,
        temperature=temperature,
        use_cosine=use_cosine,
        eps=eps,
    )

    if top_k < logits.size(-1):
        selected_logits, selected_indices = torch.topk(logits, k=top_k, dim=-1)
        selected_probs = torch.softmax(selected_logits.float(), dim=-1).to(
            dtype=basis_weight.dtype,
            device=basis_weight.device,
        )
        selected_indices = selected_indices.to(basis_weight.device)
        selected_embeddings = F.embedding(selected_indices, basis_weight.detach())
        latent = (selected_embeddings * selected_probs.unsqueeze(-1)).sum(dim=-2)
        return latent, selected_probs, selected_indices

    probs = torch.softmax(logits.float(), dim=-1).to(
        dtype=basis_weight.dtype,
        device=basis_weight.device,
    )
    latent = probs @ basis_weight.detach()
    return latent, probs, None


def _validate_vocab_matrices(
    output_embeddings: nn.Module,
    input_embeddings: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    projection_weight = output_embeddings.weight
    basis_weight = input_embeddings.weight
    if projection_weight.ndim != 2 or basis_weight.ndim != 2:
        raise ValueError("input and output embedding weights must be rank-2 matrices")
    if projection_weight.shape != basis_weight.shape:
        raise ValueError(
            "input/output vocabulary matrices must have matching shapes: "
            f"W={tuple(projection_weight.shape)}, E={tuple(basis_weight.shape)}"
        )
    return projection_weight, basis_weight


def _vocab_logits(
    hidden_states: torch.Tensor,
    projection_weight: torch.Tensor,
    temperature: float,
    use_cosine: bool,
    eps: float,
) -> torch.Tensor:
    hidden_states = hidden_states.to(
        dtype=projection_weight.dtype,
        device=projection_weight.device,
    )
    weight = projection_weight.detach()
    if use_cosine:
        hidden_states = F.normalize(hidden_states, p=2, dim=-1, eps=eps)
        weight = F.normalize(weight, p=2, dim=-1, eps=eps)
    logits = F.linear(hidden_states, weight)
    if temperature != 1.0:
        logits = logits / temperature
    return logits
