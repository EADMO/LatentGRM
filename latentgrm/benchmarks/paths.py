"""Repository-local path checks for datasets and models."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
MODEL_ROOT = REPO_ROOT / "models"
DATA_RELATIVE_ROOT = Path("data")
MODEL_RELATIVE_ROOT = Path("models")


def data_path(*parts: str) -> str:
    return str(DATA_RELATIVE_ROOT.joinpath(*parts))


def model_path(*parts: str) -> str:
    return str(MODEL_RELATIVE_ROOT.joinpath(*parts))


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=False)


def require_under(path: str | Path, root: Path, label: str) -> Path:
    resolved = resolve_repo_path(path)
    root = root.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be under {root}: {resolved}") from exc
    return resolved


def require_existing_dir(path: str | Path, root: Path, label: str) -> Path:
    resolved = require_under(path, root, label)
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"{label} not found: {resolved}. Expected an existing local path under {root}; "
            "this command will not download missing datasets or models."
        )
    return resolved
