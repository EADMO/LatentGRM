"""Datasets and subsets used for LatentGRM evaluation."""

BENCHMARKS = ('rewardbench', 'rewardbench2', 'ppe-ifeval', 'ifbench', 'rm-bench', 'helpsteer3')
DEFAULT_SUBSETS = {
    'rewardbench': (
        'alpacaeval-easy', 'alpacaeval-length', 'alpacaeval-hard',
        'mt-bench-easy', 'mt-bench-med', 'mt-bench-hard', 'llmbar-natural',
        'llmbar-adver-neighbor', 'llmbar-adver-GPTInst',
        'llmbar-adver-GPTOut', 'llmbar-adver-manual',
    ),
    'rewardbench2': ('Precise IF', 'Focus'),
    'rm-bench': ('rm-bench-chat',),
}
