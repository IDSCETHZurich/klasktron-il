# Plotting Scripts

This folder contains plot-only scripts. Evaluation and experiment runners should
write raw outputs first; these scripts then turn those outputs into figures.

## Scripts

- `plot_sparse_rl_eval_summary.py`: aggregate repeated sparse-RL evaluations into a compact summary figure set.
- `plot_data_wall_eval_results.py`: generate matchup-wise data-wall evaluation plots and score-grid heatmaps from `results_raw.csv`.
- `plot_data_wall_summary_figures.py`: generate paper-style summary figures from `results_index.json`.

## Design Rule

- `scripts/eval/` should evaluate runs and write summaries.
- `scripts/experiments/` should generate data or orchestrate experiments.
- `scripts/plotting/` should only read saved outputs and generate figures/tables/manifests.
