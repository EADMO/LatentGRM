# Optimized vLLM overlay

This directory contains 13 changed or additional Python modules for **vLLM 0.26.0**. Unmodified Python files and compiled libraries come from the installed wheel.

The selected implementation combines:

- tensor-parallel local top-k candidates instead of full-vocabulary communication;
- cached embeddings for decode inputs;
- GPU latent projection, state updates, and Gumbel sampling;
- compatibility with the model runner's default CUDA graph and asynchronous execution paths.

`latentgrm/evaluation/continuous_native_n.py` adds a continuous native multi-vote queue whose admission accounts for child rollouts. The default target is 80 active rollouts.

Build the runtime with `python -m latentgrm.vllm_support`, and evaluate through `python evaluate.py ...`. The wrapper activates the overlay and makes it visible to worker processes. Stock vLLM cannot execute the latent projection interface used by this evaluator.

Use the evaluator's command-line options to set the context length, latent budget, vote count, tensor-parallel size, and number of active rollouts.

Upstream configuration documentation: [vLLM 0.26.0](https://docs.vllm.ai/en/v0.26.0/cli/serve/). Upstream files and modifications are distributed under [Apache-2.0](LICENSE).
