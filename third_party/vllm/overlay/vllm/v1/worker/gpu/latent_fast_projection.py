"""Batched latent projection, preserving one PyTorch RNG stream per request."""

import numpy as np
import torch

try:
    from .latent_noise import fill_exponential
except ImportError:
    from latent_noise import fill_exponential


def device_tensor(values, device, dtype=None):
    # torch.tensor(..., device='cuda') blocks the host for every small metadata
    # copy. A pinned source lets PyTorch retain it until the async copy completes.
    device = torch.device(device)
    return torch.tensor(values, dtype=dtype, pin_memory=device.type == "cuda").to(
        device, non_blocking=True)


def ensure_gpu_state(runner):
    if hasattr(runner, "_latent_gpu_on"):
        return
    runner._latent_gpu_on = torch.zeros(runner.max_num_reqs, dtype=torch.bool,
                                        device=runner.device)
    runner._latent_gpu_has_soft = torch.zeros_like(runner._latent_gpu_on)
    runner._latent_gpu_soft = torch.empty(
        (runner.max_num_reqs, runner.model_config.get_hidden_size()),
        dtype=runner.dtype, device=runner.device)


def initialize_requests(runner, indices):
    ensure_gpu_state(runner)
    ids = device_tensor(indices, runner.device, torch.long)
    runner._latent_gpu_on.index_fill_(0, ids, True)
    runner._latent_gpu_has_soft.index_fill_(0, ids, False)


def project(runner, batch, logits):
    state = runner.req_states
    ready = (batch.num_computed_prefill_tokens_np + batch.num_scheduled_tokens
             >= batch.prefill_len_np)
    indices = np.flatnonzero(state.latent_mode[batch.idx_mapping_np] & ready).tolist()
    if not indices:
        return None
    weight = runner._latent_embed_weight()
    device_indices = device_tensor(indices, logits.device, torch.long)
    values, tokens = torch.topk(
        logits.index_select(0, device_indices).float(),
        k=min(runner.latent_topk, logits.shape[-1]), dim=-1,
    )
    log_probs = torch.log_softmax(values, dim=-1)
    noise = torch.empty_like(values)
    scales, temperatures, noisy_rows, one_sided = [], [], [], []
    generators = []
    for j, i in enumerate(indices):
        ridx = batch.idx_mapping_np[i]
        sp = state.latent_sampling_params.get(ridx)
        use_noise = getattr(sp, "add_noise_gumbel_softmax", False)
        noisy_rows.append(use_noise)
        scales.append(getattr(sp, "noise_scale", 1.0))
        temperatures.append(getattr(sp, "gumbel_softmax_temperature", 1.0))
        one_sided.append(getattr(sp, "use_one_sided_gumbel_noise", False))
        if use_noise:
            generator = state.latent_generators.get(ridx)
            if generator is None:
                generator = torch.Generator(device=values.device)
                seed = getattr(sp, "seed", None)
                if seed is not None:
                    generator.manual_seed(seed)
                state.latent_generators[ridx] = generator
            generators.append(generator)
        else:
            generators.append(None)
    fill_exponential(noise, generators)
    noise.log_().neg_()
    def column(values, dtype=None):
        return device_tensor(values, logits.device, dtype)[:, None]
    noise = torch.where(column(one_sided), noise.clamp_min(0.0), noise)
    perturbed = log_probs + column(scales, torch.float32) * noise
    scores = torch.where(column(noisy_rows), perturbed, values)
    probs = torch.softmax(scores / column(temperatures, torch.float32), -1)
    probs = probs.to(weight.dtype)
    embeddings = weight[tokens]
    soft = torch.bmm(probs.unsqueeze(1), embeddings).squeeze(1)
    # Keep stop decisions on the GPU. Text steps may draw unused noise, but
    # never consume a latent embedding or alter the request's text sampling.
    return indices, device_indices, tokens[:, 0], tokens[:, 0], soft


def apply_sample(runner, batch, output, projected):
    _, device_indices, tokens, _, soft = projected
    ensure_gpu_state(runner)
    state_indices = batch.idx_mapping.index_select(0, device_indices).long()
    active = runner._latent_gpu_on.index_select(0, state_indices)
    sampled = output.sampled_token_ids[:, 0].clone()
    values = torch.where(active, tokens.to(sampled.dtype),
                         sampled.index_select(0, device_indices))
    sampled.index_copy_(0, device_indices, values)
    runner._latent_gpu_soft.index_copy_(0, state_indices, soft)
    runner._latent_gpu_has_soft.index_fill_(0, state_indices, True)
    runner._latent_gpu_on.index_copy_(
        0, state_indices, active & (tokens != runner.latent_end_token_id))
    output.sampled_token_ids = sampled.unsqueeze(-1)


def prepare_embeds(runner, batch):
    ensure_gpu_state(runner)
    embeds = runner.model.embed_input_ids(batch.input_ids)
    if not hasattr(runner.input_buffers, "latent_inputs_embeds"):
        runner.input_buffers.latent_inputs_embeds = torch.empty(
            (runner.max_num_tokens, embeds.shape[-1]),
            dtype=embeds.dtype, device=embeds.device,
        )
    target = runner.input_buffers.latent_inputs_embeds[:len(embeds)]
    target.copy_(embeds)
    state_indices = batch.idx_mapping.long()
    positions = batch.logits_indices.long()
    computed_positions = batch.positions.index_select(0, positions)
    prompt_lengths = runner.req_states.prefill_len.gpu.index_select(0, state_indices)
    active = (runner._latent_gpu_on.index_select(0, state_indices)
              & runner._latent_gpu_has_soft.index_select(0, state_indices)
              & (computed_positions >= prompt_lengths))
    soft = runner._latent_gpu_soft.index_select(0, state_indices)
    normal = target.index_select(0, positions)
    target.index_copy_(0, positions, torch.where(active[:, None], soft, normal))
    return target
