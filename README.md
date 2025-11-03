# My sleep quality vs weight

This codebase stitches together several years of Oura Ring (Gen 3) sleep
exports and Mi Body Composition Scale 2 weigh-ins to explore how restfulness
and other sleep features move with weight, blood oxygenation, and similar metrics.

## Quick start

The analysis entrypoint is `analyze_health_metrics.py`. It expects:

1. Oura export CSVs, supplied via the `--oura-app-data` flag. By default we use the `./data/oura/` folder.
2. Export csv from Mi Body Composition Scale 2, supplied via the `--body-file`. By default we use the `./data/scale/body.csv` file.

Install the Python dependencies with pip (a virtual environment is recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Regenerate every plot and correlation table with the defaults baked into the script:

```bash
python analyze_health_metrics.py --output-dir plots
```

The script produces weekly trend plots, high-level correlations, and effect-size summaries (variance explained, elasticities, standardized slopes) showing how sleep, weight, and blood oxygenation move together.

All output PNGs land in the directory provided via `--output-dir` (default
`plots/`). Re-run the command whenever you refresh your exports.

## Methodology
Let's ensure we don't double report. 
- **Devices & cadence** – [Oura Ring Gen 3](https://support.ouraring.com/hc/en-us/articles/4409072131091-Oura-Ring-Generation-3) nightly exports
  (sleep, readiness, activity, sleep model, SpO₂) plus [Mi Body Composition Scale 2](https://www.mi.com/uk/product/mi-body-composition-scale-2/)
  weights logged frequently from November 2020 through November 2024 (≈4 readings/week).
- **Aggregation window** – Daily observations are averaged (when duplicates exist)
  and then rolled up into Monday-aligned calendar-week means.
- **Minimal pre-processing** – No de-trending, winsorization, or manual exclusions
  (travel, illness, alcohol-heavy days) are applied. Weeks are included when at least
  one valid day exists for the metric, and all calculations work directly on the
  observed weekly averages. Plots only render when at least two overlapping weeks
  are available.
- **Highlighted sleep metrics** – two Oura metrics stand out in this analysis:
  - *Restfulness Score* tracks wake-ups, movement, and time out of bed during sleep; higher values indicate more restorative nights. My scores have ranged from ~21 to 99 (median ≈69) prior to weekly averaging. [Oura's documentation.](https://support.ouraring.com/hc/en-us/articles/360057792293-Sleep-Contributors#01GVH5KM619AD9XZ7N7CQ2BFWA)
  - *Restless Periods* counts how many restless periods a night had (definition not publicly documented). We aggregate the nightly counts to weekly averages and also report a duration-normalised rate. My rate has ranged from 25 to 60 periods per sleeping hour (median ≈43/hr).
- **Correlations & significance** – Weekly Pearson r is computed for each overlapping
  pair of series, along with two-tailed p-values.
- **β (restless periods/hr/kg)** – Derived from an ordinary least squares fit of weekly
  restless-rate averages against weight. The
  slope becomes β and the regression line underpins the scatter plot confidence band.
