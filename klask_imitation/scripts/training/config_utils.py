"""Shared helpers for release-facing training config files."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, TypeVar


T = TypeVar("T")


def repo_root() -> Path:
    """Return the repository root from the training scripts directory."""
    return Path(__file__).resolve().parents[3]


def training_config_dir() -> Path:
    """Return the canonical config directory for training entry points."""
    return repo_root() / "configs" / "training"


def default_config_path(filename: str) -> Path:
    """Return the repo-relative path to a named training config file."""
    return training_config_dir() / filename


def resolve_repo_path(path_str: Optional[str]) -> Optional[str]:
    """Resolve a filesystem path against the repo root unless already absolute."""
    if path_str is None:
        return None
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((repo_root() / candidate).resolve())


def resolve_config_path(path_str: str) -> Path:
    """Resolve a config file path from cwd or, failing that, from the repo root."""
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    repo_candidate = repo_root() / candidate
    if repo_candidate.exists():
        return repo_candidate.resolve()
    return repo_candidate


def load_json_file(path_str: str | Path) -> Dict[str, Any]:
    """Load a JSON object from disk."""
    path = Path(path_str)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}.")
    return payload


def deep_merge_dicts(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge overlay into base without mutating either input."""
    merged = deepcopy(dict(base))
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def merge_config_file(
    defaults: Mapping[str, Any],
    config_path: str | Path,
    *,
    label: str,
    allow_unknown_keys: bool = False,
) -> Dict[str, Any]:
    """Merge a JSON config file onto defaults with optional key validation."""
    path = resolve_config_path(str(config_path))
    overrides = load_json_file(path)
    if not allow_unknown_keys:
        unknown = sorted(set(overrides) - set(defaults))
        if unknown:
            raise KeyError(f"Unknown top-level keys in {label} config {path}: {unknown}")
    merged = deep_merge_dicts(defaults, overrides)
    merged["_config_path"] = str(path)
    return merged


def dataclass_to_dict(instance: T) -> Dict[str, Any]:
    """Convert a dataclass instance to a plain dictionary."""
    if not is_dataclass(instance):
        raise TypeError("dataclass_to_dict() expects a dataclass instance.")
    return asdict(instance)


def dataclass_from_mapping(instance: T, payload: Mapping[str, Any], *, label: str) -> T:
    """Return a new dataclass instance with fields overridden from payload."""
    if not is_dataclass(instance):
        raise TypeError("dataclass_from_mapping() expects a dataclass instance.")
    field_names = {field.name for field in fields(instance)}
    unknown = sorted(set(payload) - field_names)
    if unknown:
        raise KeyError(f"Unknown keys in {label}: {unknown}")
    values = dataclass_to_dict(instance)
    values.update(dict(payload))
    return type(instance)(**values)
