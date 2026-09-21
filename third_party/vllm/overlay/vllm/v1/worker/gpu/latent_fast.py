"""Reuse the already cached full embedding for pure latent/text decode input."""
from .latent_fast_state import *
from .latent_fast_state import prepare_embeds as reference_prepare_embeds


def cached_input_compatible(runner):
    if not hasattr(runner,'_latent_cached_input_compatible'):
        from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
        embedding=getattr(getattr(runner.model,'model',None),'embed_tokens',None)
        runner._latent_cached_input_compatible=bool(
            type(runner.model).__name__=='Qwen3ForCausalLM' and runner.lora_config is None
            and embedding is not None and isinstance(embedding.quant_method,UnquantizedEmbeddingMethod)
            and embedding.num_embeddings==embedding.org_vocab_size==embedding.num_embeddings_padded)
        from vllm.logger import init_logger
        init_logger(__name__).info('Latent cached input compatible: %s',runner._latent_cached_input_compatible)
    return runner._latent_cached_input_compatible


def prepare_embeds(runner,batch):
    weight=getattr(runner,'_latent_embed_weight_full',None)
    if (weight is not None and not batch.is_prefilling_np.any()
            and (batch.num_scheduled_tokens==1).all() and cached_input_compatible(runner)):
        from .latent_step_kernels import cached_decode
        target=runner.input_buffers.latent_inputs_embeds[:len(batch.input_ids)]
        cached_decode(runner,batch,target,weight)
        return target
    return reference_prepare_embeds(runner,batch)
