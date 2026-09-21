"""Generate one rubric per distinct request, then join it to both pair orders."""
from pathlib import Path
import argparse, hashlib, json, os
from prepare_data import write
from latentgrm.evaluation.progress import read_complete_rows

ROOT=Path(__file__).resolve().parent

def read_progress(path):
    result={}
    if not path.exists(): return result
    for row in read_complete_rows(path):
        key=row['request_id']
        if key in result and result[key]!=row['rubric']: raise ValueError('Conflicting rubric records')
        if not row['rubric'].strip(): raise ValueError('Empty saved rubric')
        result[key]=row['rubric']
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',choices=['rewardbench','rewardbench2'],required=True)
    parser.add_argument('--model',default='models/rubric-generator')
    parser.add_argument('--backend',choices=['transformers','vllm'],default='vllm')
    parser.add_argument('--tensor-parallel-size',type=int,default=8)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--max-new-tokens',type=int,default=1024)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args();os.chdir(ROOT)
    if args.batch_size<1: parser.error('--batch-size must be positive')
    source=ROOT/f'data/benchmark_inputs/{args.benchmark}.jsonl'
    rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    requests={hashlib.sha256(row['instruction'].encode()).hexdigest():row['instruction'] for row in rows}
    output=ROOT/f'data/rubrics/{args.benchmark}';output.mkdir(parents=True,exist_ok=True)
    settings=vars(args)
    meta=output/'generation.json'
    if meta.exists() and json.loads(meta.read_text())!=settings: raise ValueError('Rubric generation settings changed. Select a fresh data directory.')
    meta.write_text(json.dumps(settings,indent=2)+'\n')
    progress=output/'progress.jsonl';completed=read_progress(progress)
    if set(completed)-set(requests): raise ValueError('Progress contains requests absent from the benchmark.')
    todo=[(key,value) for key,value in requests.items() if key not in completed]
    if todo:
        from latentgrm.benchmarks.infer import load_single_generator,GenerationConfig
        from latentgrm.benchmarks.templates import build_rubric_prompt
        from transformers import set_seed
        set_seed(args.seed)
        generator=load_single_generator(model_dir=args.model,backend=args.backend,
            vllm_tensor_parallel_size=args.tensor_parallel_size,vllm_gpu_memory_utilization=0.9)
        config=GenerationConfig(max_new_tokens=args.max_new_tokens,temperature=0.0)
        with progress.open('a',encoding='utf-8') as stream:
            for start in range(0,len(todo),args.batch_size):
                batch=todo[start:start+args.batch_size]
                outputs=generator.generate_batch([build_rubric_prompt(text) for _,text in batch],config)
                if len(outputs)!=len(batch): raise RuntimeError('Generator batch size mismatch')
                for (key,_),rubric in zip(batch,outputs):
                    rubric=rubric.strip()
                    if not rubric: raise RuntimeError('Empty generated rubric')
                    stream.write(json.dumps({'request_id':key,'rubric':rubric},ensure_ascii=False)+'\n');stream.flush()
                    completed[key]=rubric
                print(f'{len(completed)}/{len(requests)} rubrics',flush=True)
    joined=[{**row,'rubric':completed[hashlib.sha256(row['instruction'].encode()).hexdigest()]} for row in rows]
    write(f'data/eval/{args.benchmark}.jsonl',joined)
    print(f'Ready: data/eval/{args.benchmark}.jsonl ({len(joined)} directional pairs)')

if __name__=='__main__': main()
