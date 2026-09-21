"""Benchmark-specific aggregation of pairwise judgments."""
from importlib import import_module

def summarize_benchmark(benchmark, results, data):
    if benchmark == 'rewardbench2':
        from .summarize_rewardbench2 import summarize_rewardbench2
        return summarize_rewardbench2(results, data)
    if benchmark in {'ppe-ifeval', 'ifbench', 'rm-bench', 'helpsteer3'}:
        module = import_module('latentgrm.benchmarks.' + benchmark.replace('-', '_'))
        return module.summarize(results, data)
    return {}
