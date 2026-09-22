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
    from latentgrm.benchmarks.data import validate_openrubrics_record
    from latentgrm.openrubric_utils import convert_openrubrics_record
    rows=[];skipped=[]
    for index,row in enumerate(raw):
        try: validate_openrubrics_record(dict(row))
        except ValueError as exc:
            skipped.append({'index':index,'reason':str(exc)});continue
        rows.append(convert_openrubrics_record(dict(row),len(rows)))
    write(output,rows)
    print(f'Prepared {len(rows)} examples; skipped {len(skipped)} empty responses.')

def prepare_benchmark(benchmark):
    from datasets import load_dataset
    root = Path('data/raw') / benchmark
    if benchmark == 'rewardbench':
        from latentgrm.benchmarks.eval_rewardbench import normalize_rewardbench_record
        raw = load_dataset(str(root), split='filtered')
        rows = [
            {'dataset_index': index, **normalize_rewardbench_record(dict(row), exchange)}
            for exchange in [False, True] for index, row in enumerate(raw)
        ]
    elif benchmark == 'rewardbench2':
        from latentgrm.benchmarks.rewardbench2_rubrics import build_extracted_records, reverse_pair
        raw = load_dataset(str(root), split='test')
        _, pairs, _ = build_extracted_records(raw)
        rows = pairs + [reverse_pair(row) for row in pairs]
    elif benchmark == 'ppe-ifeval':
        from latentgrm.benchmarks.ppe_ifeval import build_pair_records
        raw = list(load_dataset('parquet', data_files=str(root / 'data/train-00000-of-00001.parquet'), split='train'))
        rows, _ = build_pair_records(raw)
    elif benchmark == 'ifbench':
        from latentgrm.benchmarks.ifbench import build_pair_records
        rows, _ = build_pair_records(json.loads((root / 'IFBench.json').read_text()))
    elif benchmark == 'rm-bench':
        from latentgrm.benchmarks.rm_bench import build_pair_records
        rows, _ = build_pair_records(json.loads((root / 'total_dataset.json').read_text()))
    elif benchmark == 'helpsteer3':
        import gzip
        from latentgrm.benchmarks.helpsteer3 import build_pair_records
        with gzip.open(root / 'preference/validation.jsonl.gz', 'rt', encoding='utf-8') as stream:
            raw = [json.loads(line) for line in stream if line.strip()]
        rows, _ = build_pair_records(raw)
    # Rubrics are generated from prompts by generate_rubrics.py.
    rows = [{key: value for key, value in row.items() if key != 'rubric'} for row in rows]
    write(f'data/benchmark_inputs/{benchmark}.jsonl', rows)
    print(f'{benchmark}: {len(rows)} directional pairs')


def main():
    from latentgrm.benchmarks.registry import BENCHMARKS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'benchmarks'])
    parser.add_argument('--benchmark', choices=['all', *BENCHMARKS], default='all')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == 'train':
        from datasets import load_dataset
        prepare_train(load_dataset('data/raw/openrubrics', split='train'), Path('data/train.jsonl'))
    else:
        for benchmark in BENCHMARKS if args.benchmark == 'all' else [args.benchmark]:
            prepare_benchmark(benchmark)


if __name__ == '__main__':
    main()
