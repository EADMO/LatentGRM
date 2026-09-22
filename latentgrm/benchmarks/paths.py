"""Resolve local model paths relative to the repository."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()
