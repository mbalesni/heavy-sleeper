# Oura Data Insights

Utilities for exploring Oura Ring blood-oxygen (SpO₂) trends alongside sleep,
activity, readiness, and Renpho scale measurements.

## Quick start

The plotting workflow lives in `scripts/analyze_oura.py`. It expects:

1. The `data/oura/` folder populated with the Oura export CSVs
   (`dailyspo2.csv`, `dailyreadiness.csv`, etc.).
2. One or more scale CSVs (e.g. Renpho) supplied via the `--body-file`
   flag—by default we use `data/scale/body.csv`.

With [uv](https://github.com/astral-sh/uv) installed, you can regenerate every
plot and correlation table with:

```bash
UV_CACHE_DIR=.uv-cache uv run scripts/analyze_oura.py \
  --oura-app-data data/oura \
  --body-file data/scale/body.csv \
  --output-dir plots
```

The script produces:

- Daily SpO₂ and a rolling 7-day average.
- Weekly comparisons between SpO₂ and body metrics (weight, body fat %, visceral fat).
- Weekly SpO₂ vs resting heart rate.
- Weekly restfulness vs body metrics.
- Weekly deep sleep duration vs visceral fat (including a pre-cutoff view).
- Pearson correlations (standard and 1-week lead/lag relationships) across all aligned metrics.

All output PNGs land in the directory provided via `--output-dir` (default
`plots/`). Re-run the command whenever you refresh your exports.

## Hugging Face dataset workflow

If you upload the same directory contents to a (private) Hugging Face dataset
repository—e.g. `your-account/oura-data` containing `oura/*.csv` and
`scale/body.csv`—the script can download it on the fly:

```bash
export HF_TOKEN=...  # set to a read token with access to the private repo
UV_CACHE_DIR=.uv-cache uv run --with huggingface_hub scripts/analyze_oura.py \
  --hf-repo-id your-account/oura-data \
  --output-dir plots
```

You can still mix in local overrides with `--oura-app-data` or additional
`--body-file` flags if you want to swap out individual files.
