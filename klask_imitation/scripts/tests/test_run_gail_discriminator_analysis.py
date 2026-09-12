from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "run_gail_discriminator_analysis.py"
    )
    spec = importlib.util.spec_from_file_location("run_gail_discriminator_analysis", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_tags():
    mod = _load_module()
    tags = mod._parse_tags("GAIL_Human, GAIL_DW_EXPERT_250k")
    assert tags == ["GAIL_Human", "GAIL_DW_EXPERT_250k"]


def test_resolve_latest_runs_by_tag():
    mod = _load_module()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "klask_gail_20260301_120000_GAIL_Human").mkdir()
        (root / "klask_gail_20260302_120000_GAIL_Human").mkdir()
        (root / "klask_gail_20260301_130000_GAIL_DW_EXPERT_250k").mkdir()
        resolved = mod._resolve_latest_runs(root, ["GAIL_Human", "GAIL_DW_EXPERT_250k"])
        assert len(resolved) == 2
        assert resolved[0]["run_name"] == "klask_gail_20260302_120000_GAIL_Human"
        assert resolved[1]["run_name"] == "klask_gail_20260301_130000_GAIL_DW_EXPERT_250k"


def test_plot_overfit_gap_writes_png():
    mod = _load_module()
    with tempfile.TemporaryDirectory() as tmp:
        out_png = Path(tmp) / "overfit_gap_by_dataset.png"
        rows = [
            {
                "run_tag": "GAIL_Human",
                "expert_positive_acc_train_minus_val": 0.12,
                "balanced_acc_total_train_minus_val": 0.08,
            },
            {
                "run_tag": "GAIL_DW_EXPERT_250k",
                "expert_positive_acc_train_minus_val": 0.05,
                "balanced_acc_total_train_minus_val": 0.03,
            },
        ]
        mod._plot_overfit_gap(rows, out_png)
        assert out_png.exists()
        assert out_png.stat().st_size > 0

