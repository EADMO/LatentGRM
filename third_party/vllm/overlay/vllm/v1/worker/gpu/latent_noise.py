"""Batch PyTorch-compatible CUDA exponential draws for short top-k rows."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from triton.language.random import philox


@triton.jit
def _exponential(out, seeds, offsets, enabled, K: tl.constexpr,
                 BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    seed = tl.load(seeds + row)
    offset = tl.load(offsets + row)
    active = tl.load(enabled + row)
    # PyTorch's short contiguous float32 draw uses one CUDA thread per element,
    # curand_uniform4 word x, and advances the generator by four per invocation.
    counter = offset // 4
    random, _, _, _ = philox(seed, counter.to(tl.uint32),
        (counter >> 32).to(tl.uint32), column.to(tl.uint32),
        tl.full((BLOCK,), 0, tl.uint32))
    uniform = random.to(tl.float32) * 2.3283064365386963e-10 + 1.1641532182693481e-10
    log = tl.where(uniform >= 1.0 - 5.960464477539063e-8,
                   -5.960464477539063e-8, libdevice.log(uniform))
    tl.store(out + row * K + column, tl.where(active, -log, 1.0), column < K)


def fill_exponential(output, generators):
    if output.dtype != torch.float32 or not 1 <= output.shape[1] <= 256:
        for row, generator in enumerate(generators):
            if generator is None:
                output[row].fill_(1.0)
            else:
                output[row].exponential_(generator=generator)
        return
    seeds, offsets = [], []
    for generator in generators:
        seeds.append(generator.initial_seed() if generator is not None else 0)
        offset = generator.get_offset() if generator is not None else 0
        offsets.append(offset)
        if generator is not None:
            generator.set_offset(offset + 4)
    def device_array(values, dtype):
        return torch.tensor(values, dtype=dtype, pin_memory=True).to(
            output.device, non_blocking=True)
    seeds = device_array(seeds, torch.uint64)
    offsets = device_array(offsets, torch.uint64)
    enabled = device_array([g is not None for g in generators], torch.bool)
    _exponential[(output.shape[0],)](output, seeds, offsets, enabled,
        output.shape[1], triton.next_power_of_2(output.shape[1]), num_warps=1)
