"""Download KlaskTron dataset tiers from Hugging Face."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from klasktron.paths import get_project_paths

DEFAULT_ENV_REPO_ID = "KLASKTRON_HF_REPO_ID"
DEFAULT_REPO_ID = "KlaskLab/klasktron-il-benchmark"


@dataclass(frozen=True)
class DatasetTier:
    name: str
    subdir: str
    steps: str
    description: str


DATASET_TIERS = {
    "human": DatasetTier(
        name="human",
        subdir="human",
        steps="80k",
        description="Real human demonstrations after reestimation and cleaning.",
    ),
    "augmented": DatasetTier(
        name="augmented",
        subdir="human_augmented",
        steps="325k",
        description="Human demonstrations plus benchmark augmentation.",
    ),
    "synthetic": DatasetTier(
        name="synthetic",
        subdir="synthetic",
        steps="5M+",
        description="Synthetic expert rollouts collected in simulation.",
    ),
}


def _resolve_repo_id(repo_id: str | None) -> str:
    resolved = repo_id or os.environ.get(DEFAULT_ENV_REPO_ID) or DEFAULT_REPO_ID
    if not resolved:
        raise SystemExit(
            "Missing Hugging Face dataset repo id. Pass --repo-id or set "
            f"{DEFAULT_ENV_REPO_ID}."
        )
    return resolved


def _allow_patterns_for_tier(tier: str, include_docs: bool) -> List[str]:
    if tier == "all":
        patterns = [f"{item.subdir}/**" for item in DATASET_TIERS.values()]
    else:
        patterns = [f"{DATASET_TIERS[tier].subdir}/**"]
    if include_docs:
        patterns.extend(["dataset_card.md", "README.md", "docs/**"])
    return patterns


def _snapshot_download(
    *,
    repo_id: str,
    allow_patterns: Iterable[str],
    local_dir: Path,
    revision: str | None,
) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for dataset download. "
            "Install with `pip install -e ./klask_imitation[data]`."
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    patterns = list(allow_patterns)
    result = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(local_dir),
        revision=revision,
    )
    _assert_patterns_matched(
        repo_id=repo_id, patterns=patterns, revision=revision
    )
    return result


def _assert_patterns_matched(
    *, repo_id: str, patterns: Iterable[str], revision: str | None
) -> None:
    """Raise if a tier pattern matched no files in the remote repo.

    A mismatch between the tier subdir here and the layout on the Hub
    otherwise downloads nothing and still reports success.
    """
    import fnmatch

    try:
        from huggingface_hub import HfApi

        remote = HfApi().list_repo_files(
            repo_id, repo_type="dataset", revision=revision
        )
    except Exception:  # offline or API change: do not block a good download
        return

    tier_patterns = [p for p in patterns if p.endswith("/**")]
    unmatched = [
        p
        for p in tier_patterns
        if not any(fnmatch.fnmatch(f, p) for f in remote)
    ]
    if unmatched:
        raise SystemExit(
            f"No files in {repo_id} matched: {', '.join(sorted(unmatched))}. "
            "The dataset layout on the Hub does not match the tier definitions "
            "in DATASET_TIERS; nothing was downloaded for those tiers."
        )


def download_dataset_tier(
    *,
    repo_id: str,
    tier: str,
    local_dir: str | Path | None = None,
    revision: str | None = None,
    include_docs: bool = True,
) -> Path:
    """Download one KlaskTron dataset tier or the full dataset repo."""
    if tier not in {*DATASET_TIERS.keys(), "all"}:
        raise ValueError(
            f"Unsupported tier={tier!r}. Expected one of "
            f"{sorted(DATASET_TIERS.keys()) + ['all']}."
        )

    paths = get_project_paths()
    target_dir = Path(local_dir or paths.data_dir).expanduser()
    resolved_repo_id = _resolve_repo_id(repo_id)
    allow_patterns = _allow_patterns_for_tier(tier, include_docs=include_docs)
    _snapshot_download(
        repo_id=resolved_repo_id,
        allow_patterns=allow_patterns,
        local_dir=target_dir,
        revision=revision,
    )
    return target_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download KlaskTron datasets from Hugging Face.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=(
            "Hugging Face dataset repo id. If omitted, read from "
            f"{DEFAULT_ENV_REPO_ID}."
        ),
    )
    parser.add_argument(
        "--tier",
        choices=["human", "augmented", "synthetic", "all"],
        default="all",
        help="Dataset tier to download.",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Destination directory. Defaults to KLASKTRON_DATA_DIR or ./data.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional Hugging Face revision, branch, or tag.",
    )
    parser.add_argument(
        "--no-docs",
        action="store_true",
        help="Skip dataset documentation files during download.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    target_dir = download_dataset_tier(
        repo_id=args.repo_id,
        tier=args.tier,
        local_dir=args.local_dir,
        revision=args.revision,
        include_docs=not args.no_docs,
    )
    print(f"Downloaded KlaskTron dataset tier '{args.tier}' into {target_dir}")


if __name__ == "__main__":
    main()
