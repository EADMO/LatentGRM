"""Download pinned assets for training and the four main evaluation domains."""
from pathlib import Path
import argparse, json, os, shutil, tempfile, urllib.request, zipfile

ROOT=Path(__file__).resolve().parent
PARSER_URL='https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'

def download_parser(local_wheel=None):
    target=ROOT/'models/en_core_web_sm'
    if (target/'config.cfg').exists():
        meta=json.loads((target/'meta.json').read_text())
        if meta.get('version')!='3.8.0': raise ValueError('Parser version must be 3.8.0')
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temp:
        wheel=Path(local_wheel) if local_wheel else Path(temp)/'parser.whl'
        if not local_wheel:
            try:
                with urllib.request.urlopen(PARSER_URL,timeout=60) as response, wheel.open('wb') as output:
                    shutil.copyfileobj(response,output)
            except OSError as exc:
                raise RuntimeError('Cannot download the spaCy release. Download the official wheel on a connected machine and pass --parser-wheel <file>.') from exc
        with zipfile.ZipFile(wheel) as archive:
            prefix='en_core_web_sm/en_core_web_sm-3.8.0/'
            for name in archive.namelist():
                if not name.startswith(prefix) or name.endswith('/'): continue
                relative=Path(name[len(prefix):])
                if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe archive path')
                output=Path(temp)/'model'/relative
                output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(archive.read(name))
        shutil.move(str(Path(temp)/'model'),target)

def main():
    manifest=json.loads((ROOT/'configs/assets.json').read_text())
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',nargs='+',choices=['all','parser',*manifest],default=['qwen3-8b','openrubrics','parser','rubric-generator','rewardbench','rewardbench2'])
    parser.add_argument('--max-workers',type=int,default=2)
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--parser-wheel',type=Path,help='Use a previously downloaded official spaCy 3.8.0 wheel.')
    args=parser.parse_args()
    if args.list:
        print(json.dumps(manifest,indent=2));return
    if args.max_workers<1: parser.error('--max-workers must be positive')
    from huggingface_hub import snapshot_download
    names=[*manifest,'parser'] if 'all' in args.assets else args.assets
    for name in dict.fromkeys(names):
        print(f'Downloading {name}',flush=True)
        if name=='parser': download_parser(args.parser_wheel);continue
        asset=manifest[name]
        snapshot_download(repo_id=asset['repo_id'],repo_type=asset['type'],revision=asset['revision'],
                          local_dir=str(ROOT/asset['path']),max_workers=args.max_workers,token=os.environ.get('HF_TOKEN'))
    print('Assets ready. Run python prepare_data.py train.')

if __name__=='__main__': main()
