# Third-party components

- The latent encoder/decoder training infrastructure incorporates Latent-SFT code under the MIT license; see `third_party/Latent-SFT-LICENSE`.
- The vLLM overlay incorporates vLLM 0.26.0 Python modules under Apache-2.0. Existing source notices are retained; see `third_party/vllm/LICENSE`. Modified files are under `third_party/vllm/overlay/`.
- Model weights, tokenizers, the spaCy parser, and datasets are downloaded from their original publishers. Their respective licenses and dataset terms continue to apply. The pinned repository IDs and revisions are listed in `configs/assets.json` and `download.py`.
- Benchmark preparation and rubric prompts follow the OpenRubrics data format. RewardBench and RewardBench 2 retain their benchmark-specific aggregation rules.

This code distribution contains no model weights, benchmark responses, or generated evaluation rubrics.
