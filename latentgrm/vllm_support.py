"""Install and activate the latent inference extension for vLLM."""
from pathlib import Path
import importlib, importlib.metadata, os, shutil, sys

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / '.runtime/vllm'

def install():
    if importlib.metadata.version('vllm') != '0.26.0':
        raise RuntimeError('Install vllm==0.26.0 in the inference environment.')
    source = Path(importlib.metadata.distribution('vllm').locate_file('vllm'))
    target = OVERLAY / 'vllm'
    if not (OVERLAY / '.installed').exists():
        def copy(src, dst):
            if str(src).endswith('.so'):
                if os.path.lexists(dst): os.unlink(dst)
                os.symlink(Path(src).resolve(), dst)
            else:
                shutil.copy2(src, dst)
            return dst
        shutil.copytree(source, target, dirs_exist_ok=True, copy_function=copy,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (OVERLAY / '.installed').touch()
    shutil.copytree(ROOT / 'third_party/vllm/overlay/vllm', target, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    return OVERLAY

def activate():
    path = install()
    sys.path.insert(0, str(path))
    os.environ['PYTHONPATH'] = os.pathsep.join([str(path), str(ROOT), os.environ.get('PYTHONPATH', '')])
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '1'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    importlib.invalidate_caches()

if __name__ == '__main__':
    print(install())
