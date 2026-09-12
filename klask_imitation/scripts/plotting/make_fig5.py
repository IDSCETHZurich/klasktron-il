#!/usr/bin/env python3
"""Regenerate Figure 5 (BC+RL 2x2 termination modes) with seed-level error bars.

The submitted Fig. 5 is a stacked composition bar with no uncertainty. Stacked
segments cannot carry honest error bars -- each segment's position depends on every
segment below it, so a whisker on one is not a statement about that quantity. This
version keeps the same measurements but splits them into grouped bars per
termination mode, faceted by reward regime, so each bar carries a real 95% CI over
the 5 training seeds.

Source: artifacts/bcrl_2x2_multiseed/<run-id>/eval/*/summary.json -- each policy
scored over 1000 episodes against the frozen PPO reference (Elo 1283).

Usage:  python make_fig5.py [--run-id <run>] [--out <dir>]

Reads the sweep produced by scripts/experiments/run_bcrl_multiseed.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_REPO_ROOT = Path(__file__).resolve().parents[3]
KLASKTRON_ROOT = Path(os.environ.get("KLASKTRON_ROOT", _REPO_ROOT))
ART = str(KLASKTRON_ROOT / "artifacts" / "bcrl_2x2_multiseed")

# Validated categorical slots 1 and 2 (dataviz reference palette, light mode).
# node scripts/validate_palette.js "#2a78d6,#eb6834" --mode light -> ALL CHECKS PASS
# worst adjacent CVD dE 24.7 (protan), normal-vision dE 33.6, both >= 3:1 on surface.
C_RANDOM = "#2a78d6"
C_BC = "#eb6834"

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"

TERMS = [
    ("term_rate_time_out", "Time-out"),
    ("term_rate_goal_scored", "Goal\nscored"),
    ("term_rate_goal_conceded", "Goal\nconceded"),
    ("term_rate_player_in_goal", "Player\nin goal"),
    ("term_rate_opponent_in_goal", "Opponent\nin goal"),
]

# Two-sided 95% t critical values, small n.
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
          8: 2.306, 9: 2.262}


def mean_ci95(v):
    """Mean and half-width of the t-based 95% CI over seeds."""
    n = len(v)
    m = st.fmean(v)
    if n < 2:
        return m, 0.0
    return m, T_CRIT.get(n - 1, 1.96) * st.stdev(v) / math.sqrt(n)


def load(run_id):
    """condition -> metric -> list of per-seed values."""
    out = {}
    for path in sorted(glob.glob(f"{ART}/{run_id}/eval/*/summary.json")):
        job = os.path.basename(os.path.dirname(path))
        cond, _, _seed = job.rpartition("_seed")
        d = json.load(open(path))
        out.setdefault(cond, {})
        for key, _ in TERMS:
            out[cond].setdefault(key, []).append(float(d.get(key, float("nan"))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=os.environ.get("KLASKTRON_BCRL_RUN_ID",
                                                   "camera_ready_a10_20260911"))
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    data = load(args.run_id)
    missing = [c for c in ("sparse_random", "sparse_bc", "dense_random", "dense_bc")
               if c not in data]
    if missing:
        raise SystemExit(f"Missing conditions: {missing}")
    n_seeds = len(data["sparse_random"][TERMS[0][0]])

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    x = range(len(TERMS))
    width = 0.36

    for ax, regime, title in ((axes[0], "sparse", "Sparse reward"),
                              (axes[1], "dense", "Dense reward")):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

        for offset, init, color in ((-width / 2, "random", C_RANDOM),
                                    (+width / 2, "bc", C_BC)):
            cond = f"{regime}_{init}"
            means, errs = [], []
            for key, _ in TERMS:
                m, e = mean_ci95(data[cond][key])
                means.append(m)
                errs.append(e)
            ax.bar([i + offset for i in x], means, width * 0.92, color=color,
                   zorder=2, linewidth=0)
            # Error bars in ink, not the series colour: they are not a third series.
            ax.errorbar([i + offset for i in x], means, yerr=errs, fmt="none",
                        ecolor=INK_2, elinewidth=1.2, capsize=3, capthick=1.2, zorder=3)

        ax.set_xticks(list(x))
        ax.set_xticklabels([lbl for _, lbl in TERMS], fontsize=8, color=INK_2)
        ax.tick_params(axis="both", length=0, labelsize=8, colors=INK_2)
        ax.set_title(title, fontsize=10, color=INK, pad=8)
        ax.set_ylim(0, 0.75)

    axes[0].set_ylabel("Termination rate per episode", fontsize=9, color=INK_2)

    # Legend present for 2 series; identity is never colour-alone in the caption.
    fig.legend(handles=[Patch(facecolor=C_RANDOM, label="Random init"),
                        Patch(facecolor=C_BC, label="BC init")],
               loc="upper right", bbox_to_anchor=(0.995, 1.02), frameon=False,
               fontsize=9, labelcolor=INK_2, handlelength=1.1, handleheight=1.1)

    fig.suptitle(f"Termination modes across the BC+RL 2x2  "
                 f"(mean $\\pm$ 95% CI over {n_seeds} training seeds, vs. frozen PPO)",
                 fontsize=10, color=INK, x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    for ext in ("pdf", "png"):
        p = os.path.join(args.out, f"fig5_bcrl_2x2_seeds.{ext}")
        fig.savefig(p, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        print(f"[out] {p}")

    # Print the numbers behind the bars so the figure is checkable without reading pixels.
    print(f"\n{'condition':<15s}" + "".join(f"{lbl.replace(chr(10),' '):>20s}"
                                            for _, lbl in TERMS))
    for cond in ("sparse_random", "sparse_bc", "dense_random", "dense_bc"):
        row = f"{cond:<15s}"
        for key, _ in TERMS:
            m, e = mean_ci95(data[cond][key])
            row += f"{m:>12.3f}+-{e:<6.3f}"
        print(row)


if __name__ == "__main__":
    main()
