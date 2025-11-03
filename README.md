# Health Metrics Insights

Utilities for exploring Oura Ring blood-oxygen (SpO₂) trends alongside sleep,
activity, readiness, and readings from the Mi Body Composition Scale 2.

## Quick start

The analysis entrypoint is `analyze_health_metrics.py`. It expects:

1. The `data/oura/` folder populated with the Oura export CSVs
   (`dailyspo2.csv`, `dailyreadiness.csv`, etc.).
2. One or more scale CSVs (e.g. Mi Body Composition Scale 2) supplied via the `--body-file`
   flag—by default we use `data/scale/body.csv`.

With [uv](https://github.com/astral-sh/uv) installed, you can regenerate every
plot and correlation table with the defaults baked into the script:

```bash
UV_CACHE_DIR=.uv-cache uv run analyze_health_metrics.py --output-dir plots
```

The script falls back to `data/oura/` for the wearable export CSVs and
`data/scale/body.csv` for the scale data. Override them with `--oura-app-data`
and `--body-file` when you want to analyse alternate exports.

The script produces:

- Daily SpO₂ and a rolling 7-day average.
- Weekly comparisons between SpO₂ and body metrics (weight, body fat %, visceral fat).
- Weekly SpO₂ vs resting heart rate.
- Weekly restfulness vs body metrics.
- Weekly deep sleep duration vs visceral fat (including a pre-cutoff view).
- Pearson correlations (standard and 1-week lead/lag relationships) across all aligned metrics.

All output PNGs land in the directory provided via `--output-dir` (default
`plots/`). Re-run the command whenever you refresh your exports.
