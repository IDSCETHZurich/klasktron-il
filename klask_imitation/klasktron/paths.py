"""Project path helpers for repo-relative defaults and environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    data_dir: Path
    trained_models_dir: Path
    docs_dir: Path


def get_project_paths() -> ProjectPaths:
    """Return project paths, allowing env var overrides for runtime artifacts."""
    repo_root = _resolve_repo_root()
    data_dir = Path(os.environ.get("KLASKTRON_DATA_DIR", repo_root / "data")).expanduser()
    trained_models_dir = Path(
        os.environ.get("KLASKTRON_TRAINED_MODELS_DIR", repo_root / "trained_models")
    ).expanduser()
    docs_dir = repo_root / "docs"
    return ProjectPaths(
        repo_root=repo_root,
        data_dir=data_dir,
        trained_models_dir=trained_models_dir,
        docs_dir=docs_dir,
    )
