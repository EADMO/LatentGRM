"""Latent-only state/noise kernels; no request scheduling changes."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from triton.language.random import philox


@triton.jit
def _state_exponential(out, slots, seeds, offsets, parameters, active,
                       K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(slots + row).to(tl.int64)
    enabled = tl.load(active + slot) & (tl.load(parameters + slot * 4 + 2) != 0)
    column = tl.arange(0, BLOCK)
    if enabled:
        seed = tl.load(seeds + slot).to(tl.uint64)
        offset = tl.load(offsets + slot).to(tl.uint64)
        counter = offset // 4
        random, _, _, _ = philox(seed, counter.to(tl.uint32),
            (counter >> 32).to(tl.uint32), column.to(tl.uint32),
            tl.full((BLOCK,), 0, tl.uint32))
        uniform = random.to(tl.float32) * 2.3283064365386963e-10 + 1.1641532182693481e-10
        log = tl.where(uniform >= 1.0 - 5.960464477539063e-8,
                       -5.960464477539063e-8, libdevice.log(uniform))
        value = -log
        tl.store(offsets + slot, offset + 4)
    else:
        value = tl.full((BLOCK,), 1.0, tl.float32)
    tl.store(out + row*K + column, value, column < K)


def fill_state_exponential(output, slots, runner):
    _state_exponential[(len(slots),)](output, slots, runner._latent_gpu_seeds,
        runner._latent_gpu_offsets, runner._latent_gpu_sampling, runner._latent_gpu_on,
        output.shape[1], triton.next_power_of_2(output.shape[1]), num_warps=1)


@triton.jit
def _blend(embeds, positions, token_positions, slots, prompt_lengths,
           active, has_soft, soft, H: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slots + row).to(tl.int64)
    pos = tl.load(positions + row).to(tl.int64)
    valid = (tl.load(active + slot) & tl.load(has_soft + slot)
             & (tl.load(token_positions + pos) >= tl.load(prompt_lengths + slot)))
    if valid:
        col = tile * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(soft + slot*H + col, col < H, other=0)
        tl.store(embeds + pos*H + col, values, col < H)


def blend(runner, batch, target):
    h = target.shape[1]
    _blend[(batch.num_reqs, triton.cdiv(h, 1024))](target, batch.logits_indices,
        batch.positions, batch.idx_mapping, runner.req_states.prefill_len.gpu,
        runner._latent_gpu_on, runner._latent_gpu_has_soft, runner._latent_gpu_soft,
        h, 1024, num_warps=4)


@triton.jit
def _cached_decode(target, input_ids, positions, slots, prefill_len, active,
                   has_soft, soft, weight, NREQ, VOCAB: tl.constexpr,
                   H: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    col = tile*BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slots + row, row < NREQ, other=0).to(tl.int64)
    latent = (row < NREQ) & tl.load(active + slot) & tl.load(has_soft + slot)
    latent = latent & (tl.load(positions + row) >= tl.load(prefill_len + slot))
    if latent:
        value = tl.load(soft + slot*H + col, col < H, other=0)
    else:
        token = tl.load(input_ids + row).to(tl.int64)
        value = tl.load(weight + token*H + col,
                        (col < H) & (token >= 0) & (token < VOCAB), other=0)
    tl.store(target + row*H + col, value, col < H)


def cached_decode(runner, batch, target, weight):
    _cached_decode[(len(batch.input_ids), triton.cdiv(weight.shape[1], 1024))](
        target, batch.input_ids, batch.positions, batch.idx_mapping,
        runner.req_states.prefill_len.gpu, runner._latent_gpu_on,
        runner._latent_gpu_has_soft, runner._latent_gpu_soft, weight,
        batch.num_reqs, weight.shape[0], weight.shape[1], 1024, num_warps=4)


@triton.jit
def _update(sampled, batch_rows, tokens, soft, slots, active, has_soft, state_soft,
            TOKEN_STRIDE: tl.constexpr, H: tl.constexpr, END: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    batch_row = tl.load(batch_rows + row).to(tl.int64)
    slot = tl.load(slots + batch_row).to(tl.int64)
    on = tl.load(active + slot)
    if on:
        token = tl.load(tokens + row * TOKEN_STRIDE)
        tl.store(sampled + batch_row, token)
        if token != END:
            # One CTA per request avoids racing the mode update against other
            # hidden-dimension tiles that also read the old active flag.
            col = tl.arange(0, BLOCK)
            value = tl.load(soft + row*H + col, col < H, other=0)
            tl.store(state_soft + slot*H + col, value, col < H)
            tl.store(has_soft + slot, True)
        else:
            tl.store(active + slot, False)


def update(runner, batch, output, projected):
    _, batch_rows, tokens, _, soft = projected
    h = soft.shape[1]
    _update[(len(batch_rows),)](output.sampled_token_ids, batch_rows, tokens, soft,
        batch.idx_mapping, runner._latent_gpu_on, runner._latent_gpu_has_soft,
        runner._latent_gpu_soft, tokens.stride(0), h, runner.latent_end_token_id,
        triton.next_power_of_2(h), num_warps=4)
