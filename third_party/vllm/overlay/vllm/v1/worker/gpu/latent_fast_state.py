"""Exact vocabulary-parallel top-k for the fixed latent greedy/Gumbel protocol."""
import numpy as np
import torch
from . import latent_fast_projection as projection

project = projection.project
apply_sample = projection.apply_sample
prepare_embeds = projection.prepare_embeds
ensure_gpu_state = projection.ensure_gpu_state
initialize_requests = projection.initialize_requests


def sample_hidden(runner, hidden, batch):
    """Avoid gathering full-vocabulary logits; gather only each shard's top-k."""
    # This candidate is deliberately restricted to the exact evaluation contract.
    if (batch.has_structured_output_reqs or batch.num_draft_tokens != 0
            or hidden.shape[0] != batch.num_reqs):
        return None
    params = [runner.req_states.latent_sampling_params.get(i)
              for i in batch.idx_mapping_np]
    if not all(p is not None and getattr(p, 'temperature', None) == 0
               and getattr(p, 'add_noise_gumbel_softmax', False)
               and getattr(p, 'noise_scale', 1.) == 1.
               and getattr(p, 'gumbel_softmax_temperature', 1.) == 1.
               and not getattr(p, 'use_one_sided_gumbel_noise', False)
               for p in params):
        return None
    processor = runner.model.logits_processor
    head = runner.model.lm_head
    local = processor._apply_head(head, hidden, None)
    num_pad = head.shard_indices.num_org_vocab_padding
    if num_pad:
        local[..., -num_pad:] = -float('inf')
    k = min(runner.latent_topk, local.shape[-1])
    values, indices = torch.topk(local.float(), k=k, dim=-1)
    indices = indices + head.shard_indices.org_vocab_start_index
    # float32 exactly represents the vocabulary indices (< 2**24).
    packed = torch.stack((values, indices.float()), dim=1)
    from vllm.distributed import tensor_model_parallel_all_gather
    packed = tensor_model_parallel_all_gather(packed, dim=-1)
    candidate_values, order = torch.topk(packed[:, 0], k=k, dim=-1)
    candidate_tokens = packed[:, 1].gather(-1, order).long()
    return _finish(runner, batch, candidate_values, candidate_tokens)


def _finish(runner, batch, values, tokens):
    state = runner.req_states
    ready = batch.num_computed_prefill_tokens_np + batch.num_scheduled_tokens >= batch.prefill_len_np
    rows = np.flatnonzero(state.latent_mode[batch.idx_mapping_np] & ready).tolist()
    device_rows = projection.device_tensor(rows, values.device, torch.long)
    selected_values = values.index_select(0, device_rows)
    selected_tokens = tokens.index_select(0, device_rows)
    generators = []
    for row in rows:
        slot = batch.idx_mapping_np[row]
        generator = state.latent_generators.get(slot)
        if generator is None:
            generator = torch.Generator(device=values.device)
            seed = getattr(state.latent_sampling_params.get(slot), 'seed', None)
            if seed is not None:
                generator.manual_seed(seed)
            state.latent_generators[slot] = generator
        generators.append(generator)
    noise = torch.empty_like(selected_values)
    projection.fill_exponential(noise, generators)
    probs = torch.softmax(torch.log_softmax(selected_values, -1) + noise.log().neg(), -1)
    weight = runner._latent_embed_weight()
    soft = torch.bmm(probs.to(weight.dtype).unsqueeze(1), weight[selected_tokens]).squeeze(1)
    projected = (rows, device_rows, selected_tokens[:, 0], selected_tokens[:, 0], soft)
    # Normal output after </think> is greedy. Choose max value, breaking ties by
    # the lowest global token id, matching full-logit argmax.
    maxima = values[:, :1]
    ties = torch.where(values == maxima, tokens, torch.iinfo(tokens.dtype).max)
    sampled = ties.min(-1, keepdim=True).values
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    num_sampled, num_rejected = get_num_sampled_and_rejected(
        batch.seq_lens.new_ones(batch.num_reqs), batch.seq_lens,
        batch.cu_num_logits, batch.idx_mapping,
        runner.req_states.prefill_len.gpu)
    output = SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None,
                           num_nans=None, num_sampled=num_sampled,
                           num_rejected=num_rejected)
    projection.apply_sample(runner, batch, output, projected)
    return output, output.num_sampled, output.num_rejected
