"""Evaluate LatentGRM with HF or the optimized vLLM runtime."""
from pathlib import Path
import argparse,os,sys
from latentgrm.benchmarks.registry import BENCHMARKS, DEFAULT_SUBSETS

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='outputs/Qwen3-8B/stage2/hf')
    parser.add_argument('--benchmark',choices=BENCHMARKS,required=True)
    parser.add_argument('--backend',choices=['hf','vllm'],default='vllm')
    parser.add_argument('--vote',type=int,default=5)
    parser.add_argument('--tensor-parallel-size',type=int,default=8)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--max-latent-tokens',type=int,default=256)
    parser.add_argument('--max-total-length',type=int,default=6144)
    parser.add_argument('--active-rollouts',type=int,default=80)
    parser.add_argument('--output')
    parser.add_argument('--max-samples',type=int)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args();os.chdir(ROOT)
    if args.backend=='vllm':
        from latentgrm.vllm_support import activate
        activate()
    from latentgrm.evaluation.judge import main as run
    model=Path(args.model);run_name=model.parent.name if model.name=='hf' else model.name
    output=args.output or f'outputs/evaluation/{args.benchmark}/{run_name}/vote{args.vote}/results.jsonl'
    sys.argv=['evaluate.py','--benchmark',args.benchmark,'--data_path',f'data/eval/{args.benchmark}.jsonl','--model_path',args.model,
        '--output_path',output,'--backend',args.backend,'--vote',str(args.vote),
        '--tensor_parallel_size',str(args.tensor_parallel_size),'--gpu_memory_utilization','0.9',
        '--max_total_length',str(args.max_total_length),'--max_latent_tokens',str(args.max_latent_tokens),
        '--max_answer_tokens','8','--topk_interpolation','10','--add_gumbel_noise','--gumbel_temperature','1',
        '--noise_scale','1','--gumbel_seed',str(args.seed),'--vllm_target_active_rollouts',str(args.active_rollouts),
        '--overwrite' if args.overwrite else '--resume']
    if args.benchmark in DEFAULT_SUBSETS:
        sys.argv+=['--include-subsets',','.join(DEFAULT_SUBSETS[args.benchmark])]
    if args.max_samples: sys.argv+=['--max_samples',str(args.max_samples)]
    run()

if __name__=='__main__': main()
