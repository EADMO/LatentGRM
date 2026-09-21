"""Prepare canonical training records or the main benchmark pairs."""
from pathlib import Path
import argparse, json, os

ROOT=Path(__file__).resolve().parent

def write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        for row in rows: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    temp.replace(path)

def prepare_train(raw,output):
    from latentgrm.benchmarks.data import judge_sft_record
    from latentgrm.openrubric_utils import convert_judge_record
    rows=[];skipped=[]
    for index,row in enumerate(raw):
        try: source=judge_sft_record(dict(row))
        except ValueError as exc:
            skipped.append({'index':index,'reason':str(exc)});continue
        rows.append(convert_judge_record(source,len(rows)))
    write(output,rows)
    print(f'Prepared {len(rows)} examples; skipped {len(skipped)} empty responses.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['train','benchmarks'])
    parser.add_argument('--benchmark',choices=['all','rewardbench','rewardbench2'],default='all')
    args=parser.parse_args();os.chdir(ROOT)
    from datasets import load_dataset
    if args.action=='train':
        prepare_train(load_dataset('data/raw/openrubrics',split='train'),Path('data/train.jsonl'));return
    if args.benchmark in ['all','rewardbench']:
        from latentgrm.benchmarks.eval_rewardbench import normalize_rewardbench_record
        from latentgrm.evaluation.judge import REWARDBENCH_SUBSET_MAPPING
        selected=set(REWARDBENCH_SUBSET_MAPPING['Chat']+REWARDBENCH_SUBSET_MAPPING['Chat Hard'])
        data=load_dataset('data/raw/rewardbench',split='filtered')
        rows=[]
        for exchange in [False,True]:
            for index,source in enumerate(data):
                row=normalize_rewardbench_record(dict(source),exchange)
                if row['subset'] in selected: rows.append({'dataset_index':index,**row})
        write('data/benchmark_inputs/rewardbench.jsonl',rows)
        print(f'RewardBench Chat / Chat Hard: {len(rows)} directional pairs')
    if args.benchmark in ['all','rewardbench2']:
        from latentgrm.benchmarks.rewardbench2_rubrics import build_extracted_records,reverse_pair
        raw=load_dataset('data/raw/rewardbench2',split='test')
        _,pairs,manifest=build_extracted_records(raw,subsets=['Precise IF','Focus'])
        write('data/benchmark_inputs/rewardbench2.jsonl',pairs+[reverse_pair(row) for row in pairs])
        print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
