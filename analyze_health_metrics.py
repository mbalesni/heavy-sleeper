#!/usr/bin/env python3
"""Analyse wearable and scale exports and generate plots/correlations."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np

from health_analysis.constants import DEFAULT_PRE_CUTOFF
from health_analysis.data_sources import HuggingFaceOptions, resolve_data_paths
from health_analysis.loaders import (
    load_body_metrics,
    load_daily_activity_metrics,
    load_daily_readiness_metrics,
    load_daily_sleep_metrics,
    load_resting_hr_records,
    load_sleep_model_metrics,
    load_spo2_breathing_index,
    load_spo2_records,
)
from health_analysis.plotting import PlotPaths, build_plot_paths, plot_dual_series, plot_single_series
from health_analysis.processing import (
    aggregate_daily,
    aggregate_weekly,
    align_weekly_series,
    compute_lagged_correlation,
    compute_pairwise_correlations,
    compute_rolling_average,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot wearable SpO₂ trends and correlations against readiness, sleep, activity, and scale data.",
    )
    parser.add_argument(
        "--oura-app-data",
        type=Path,
        help="Directory containing the wearable (Oura-format) CSV exports (dailyspo2.csv, dailyreadiness.csv, etc.).",
    )
    parser.add_argument(
        "--body-file",
        dest="body_files",
        action="append",
        type=Path,
        default=[],
        metavar="PATH",
        help="Body composition CSV from the scale. Repeat this flag to add multiple files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where plots will be written (default: ./plots).",
    )
    parser.add_argument(
        "--cutoff-date",
        type=date.fromisoformat,
        default=DEFAULT_PRE_CUTOFF,
        help=(
            "ISO date (YYYY-MM-DD). Weeks on or before this date are used for the pre-cutoff "
            "deep sleep vs visceral fat comparison (default: 2023-01-31)."
        ),
    )
    parser.add_argument(
        "--hf-repo-id",
        help="Optional Hugging Face dataset repository ID to download data from (e.g., username/dataset).",
    )
    parser.add_argument(
        "--hf-revision",
        default=None,
        help="Optional revision (branch/tag/commit) for the Hugging Face dataset repository.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token used for private dataset access (default: read from HF_TOKEN env var).",
    )
    parser.add_argument(
        "--hf-oura-subdir",
        default="oura",
        help="Relative path inside the Hugging Face dataset containing the wearable CSV exports.",
    )
    parser.add_argument(
        "--hf-scale-files",
        action="append",
        default=["scale/body.csv"],
        help="Relative path(s) inside the Hugging Face dataset for scale CSV files. Repeat to add more.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    plot_paths: PlotPaths = build_plot_paths(output_dir)

    hf_options = HuggingFaceOptions(
        repo_id=args.hf_repo_id,
        revision=args.hf_revision,
        token=args.hf_token,
        oura_subdir=args.hf_oura_subdir,
        scale_files=tuple(args.hf_scale_files),
    )

    context = resolve_data_paths(
        oura_app_data=args.oura_app_data,
        body_files=args.body_files,
        hf_options=hf_options,
    )

    spo2_records = load_spo2_records(context.wearable.daily_spo2)
    if not spo2_records:
        raise SystemExit("No blood oxygenation records found in the export.")

    readiness_records = load_resting_hr_records(context.wearable.daily_readiness)

    daily_days, daily_spo2 = aggregate_daily(spo2_records)
    plot_single_series(
        daily_days,
        daily_spo2,
        plot_paths.daily_spo2,
        title="Daily Blood Oxygenation (SpO2)",
        y_label="Average SpO2 (%)",
    )
    print(f"Saved plot to {plot_paths.daily_spo2}")

    rolling_days, rolling_spo2 = compute_rolling_average(daily_days, daily_spo2)
    plot_single_series(
        rolling_days,
        rolling_spo2,
        plot_paths.weekly_spo2_rolling,
        title="Rolling 7-Day Average Blood Oxygenation (SpO2)",
        y_label="Rolling Average SpO2 (%)",
        color="#2ca02c",
    )
    print(f"Saved plot to {plot_paths.weekly_spo2_rolling}")

    weekly_spo2_weeks, weekly_spo2 = aggregate_weekly(daily_days, daily_spo2)
    weekly_series: dict[str, tuple[list[date], list[float]]] = {
        "SpO2": (weekly_spo2_weeks, weekly_spo2)
    }

    # Additional SpO2-derived metrics.
    bdi_records = load_spo2_breathing_index(context.wearable.daily_spo2)
    if bdi_records:
        bdi_days, bdi_values = aggregate_daily(bdi_records)
        weekly_bdi_weeks, weekly_bdi_values = aggregate_weekly(bdi_days, bdi_values)
        if weekly_bdi_weeks:
            weekly_series["Breathing Disturbance Index"] = (weekly_bdi_weeks, weekly_bdi_values)

    readiness_metrics = load_daily_readiness_metrics(context.wearable.daily_readiness)
    sleep_metrics = load_daily_sleep_metrics(context.wearable.daily_sleep)
    activity_metrics = load_daily_activity_metrics(context.wearable.daily_activity)
    sleep_model_metrics = load_sleep_model_metrics(context.wearable.sleep_model)

    for metric_map in (readiness_metrics, sleep_metrics, activity_metrics, sleep_model_metrics):
        _register_weekly_metrics(metric_map.items(), weekly_series)

    body_metrics = load_body_metrics(context.body_files)
    body_label_map = {
        "weight": "Weight",
        "fatRate": "Body Fat %",
        "visceralFat": "Visceral Fat Index",
    }
    _register_weekly_metrics(body_metrics.items(), weekly_series, rename=body_label_map)

    _plot_resting_hr(
        weekly_spo2_weeks,
        weekly_spo2,
        readiness_records,
        plot_paths.weekly_spo2_vs_resting_hr,
    )

    _plot_body_comparisons(
        weekly_spo2_weeks,
        weekly_spo2,
        body_metrics,
        plot_paths,
    )

    restfulness_summary = _plot_restfulness_comparisons(
        weekly_series,
        plot_paths,
    )

    deep_sleep_summary = _plot_deep_sleep_vs_visceral(
        weekly_series,
        plot_paths,
        args.cutoff_date,
    )

    lagged_summary = _compute_lagged_metrics(weekly_series)

    correlations = compute_pairwise_correlations(weekly_series)
    if correlations:
        correlations.sort(key=lambda item: abs(item[2]), reverse=True)
        spo2_correlations = [
            (metric_a, metric_b, corr, count)
            for metric_a, metric_b, corr, count in correlations
            if metric_a == "SpO2" or metric_b == "SpO2"
        ]
        if spo2_correlations:
            print("Top weekly correlations involving SpO2 (Pearson):")
            for metric_a, metric_b, corr, count in spo2_correlations[:20]:
                other = metric_b if metric_a == "SpO2" else metric_a
                print(f"  SpO2 vs {other}: {corr:.3f} (n={count})")
        else:
            print("No overlapping weekly data available to correlate with SpO2.")
    else:
        print("Insufficient overlapping weekly data to compute correlations.")

    if restfulness_summary:
        print("Restfulness weekly correlations:")
        for label, corr, count in restfulness_summary:
            print(f"  Restfulness vs {label}: {corr:.3f} (n={count})")

    if deep_sleep_summary:
        print("Deep sleep weekly correlations:")
        for label, corr, count in deep_sleep_summary:
            print(f"  Deep Sleep vs {label}: {corr:.3f} (n={count})")

    if lagged_summary:
        print("Lagged weekly correlations (1-week lead):")
        for label, corr, count in lagged_summary:
            print(f"  {label}: {corr:.3f} (pairs={count})")
    else:
        print("Lagged weekly correlations: insufficient overlapping data.")


def _register_weekly_metrics(
    items: Sequence[tuple[str, list[tuple[date, float]]]],
    weekly_series: dict[str, tuple[list[date], list[float]]],
    rename: dict[str, str] | None = None,
) -> None:
    rename = rename or {}

    for name, records in items:
        if not records:
            continue
        daily_days, daily_values = aggregate_daily(records)
        weekly_days, weekly_values = aggregate_weekly(daily_days, daily_values)
        if weekly_days:
            label = rename.get(name, name)
            weekly_series[label] = (weekly_days, weekly_values)


def _plot_resting_hr(
    weekly_spo2_weeks: list[date],
    weekly_spo2: list[float],
    readiness_records: list[tuple[date, float]],
    output_path: Path,
) -> None:
    rest_days, rest_values = aggregate_daily(readiness_records)
    if not rest_days:
        print("Skipping resting HR comparison plot (no readiness data).")
        return

    weekly_rest, weekly_hr = aggregate_weekly(rest_days, rest_values)
    weeks_aligned, aligned_spo2, aligned_hr = align_weekly_series(
        weekly_spo2_weeks,
        weekly_spo2,
        weekly_rest,
        weekly_hr,
    )
    if not weeks_aligned:
        print("Skipping resting HR comparison plot (no overlapping weeks).")
        return

    plot_dual_series(
        weeks_aligned,
        aligned_spo2,
        aligned_hr,
        output_path,
        primary_label="SpO2",
        primary_unit="%",
        secondary_label="Resting HR",
        secondary_unit="bpm",
    )
    print(f"Saved plot to {output_path}")


def _plot_body_comparisons(
    weekly_spo2_weeks: list[date],
    weekly_spo2: list[float],
    body_metrics: dict[str, list[tuple[date, float]]],
    plot_paths: PlotPaths,
) -> None:
    comparisons = [
        ("weight", "Weight", "kg", plot_paths.weekly_spo2_vs_weight, "#9467bd"),
        ("fatRate", "Body Fat %", "%", plot_paths.weekly_spo2_vs_body_fat, "#ff7f0e"),
        ("visceralFat", "Visceral Fat Index", "index", plot_paths.weekly_spo2_vs_visceral_fat, "#8c564b"),
    ]

    for field, label, unit, output_path, color in comparisons:
        records = body_metrics.get(field, [])
        if not records:
            print(f"Skipping {label} comparison plot (no data).")
            continue

        daily_days, daily_values = aggregate_daily(records)
        weekly_metric_weeks, weekly_metric_values = aggregate_weekly(daily_days, daily_values)
        weeks_aligned, aligned_spo2, aligned_metric = align_weekly_series(
            weekly_spo2_weeks,
            weekly_spo2,
            weekly_metric_weeks,
            weekly_metric_values,
        )
        if not weeks_aligned:
            print(f"Skipping {label} comparison plot (no overlapping weeks).")
            continue

        plot_dual_series(
            weeks_aligned,
            aligned_spo2,
            aligned_metric,
            output_path,
            primary_label="SpO2",
            primary_unit="%",
            secondary_label=label,
            secondary_unit=unit,
            secondary_color=color,
        )
        print(f"Saved plot to {output_path}")


def _plot_restfulness_comparisons(
    weekly_series: dict[str, tuple[list[date], list[float]]],
    plot_paths: PlotPaths,
) -> list[tuple[str, float, int]]:
    restfulness_key = "Sleep Contributor: Restfulness"
    restfulness_series = weekly_series.get(restfulness_key)
    if not restfulness_series:
        print("No Restfulness contributor data found; skipping restfulness comparisons.")
        return []

    restfulness_weeks, restfulness_values = restfulness_series
    results: list[tuple[str, float, int]] = []

    comparisons = [
        ("Weight", "kg", plot_paths.weekly_restfulness_vs_weight, "#9467bd"),
        ("Body Fat %", "%", plot_paths.weekly_restfulness_vs_body_fat, "#ff7f0e"),
    ]

    for target_label, unit, output_path, color in comparisons:
        target_series = weekly_series.get(target_label)
        if not target_series:
            print(f"Skipping Restfulness vs {target_label} plot (no data).")
            continue

        weeks_aligned, restfulness_aligned, target_aligned = align_weekly_series(
            restfulness_weeks,
            restfulness_values,
            target_series[0],
            target_series[1],
        )
        if not weeks_aligned:
            print(f"Skipping Restfulness vs {target_label} plot (no overlapping weeks).")
            continue

        plot_dual_series(
            weeks_aligned,
            restfulness_aligned,
            target_aligned,
            output_path,
            primary_label="Restfulness",
            primary_unit="score",
            secondary_label=target_label,
            secondary_unit=unit,
            secondary_color=color,
        )
        print(f"Saved plot to {output_path}")

        corr = float(np.corrcoef(restfulness_aligned, target_aligned)[0, 1])
        results.append((target_label, corr, len(weeks_aligned)))

    return results


def _plot_deep_sleep_vs_visceral(
    weekly_series: dict[str, tuple[list[date], list[float]]],
    plot_paths: PlotPaths,
    cutoff_date: date,
) -> list[tuple[str, float, int]]:
    deep_sleep_key = "Sleep Deep Sleep Duration (min)"
    deep_sleep_series = weekly_series.get(deep_sleep_key)
    visceral_series = weekly_series.get("Visceral Fat Index")
    if not deep_sleep_series or not visceral_series:
        print("Deep sleep duration or visceral fat data missing; skipping their comparison.")
        return []

    weeks_aligned, deep_sleep_aligned, visceral_aligned = align_weekly_series(
        deep_sleep_series[0],
        deep_sleep_series[1],
        visceral_series[0],
        visceral_series[1],
    )
    if not weeks_aligned:
        print("Skipping Deep Sleep vs Visceral Fat plot (no overlapping weeks).")
        return []

    plot_dual_series(
        weeks_aligned,
        deep_sleep_aligned,
        visceral_aligned,
        plot_paths.weekly_deep_sleep_vs_visceral_fat,
        primary_label="Sleep Deep Sleep Duration (min)",
        primary_unit="min",
        secondary_label="Visceral Fat Index",
        secondary_unit="index",
        secondary_color="#8c564b",
    )
    print(f"Saved plot to {plot_paths.weekly_deep_sleep_vs_visceral_fat}")

    results: list[tuple[str, float, int]] = []
    corr = float(np.corrcoef(deep_sleep_aligned, visceral_aligned)[0, 1])
    results.append(("Visceral Fat Index", corr, len(weeks_aligned)))

    filtered_pairs = [
        (week, ds, vf)
        for week, ds, vf in zip(weeks_aligned, deep_sleep_aligned, visceral_aligned)
        if week <= cutoff_date
    ]
    if len(filtered_pairs) >= 2:
        weeks_pre_cutoff = [week for week, _, _ in filtered_pairs]
        deep_sleep_pre = [ds for _, ds, _ in filtered_pairs]
        visceral_pre = [vf for _, _, vf in filtered_pairs]
        plot_dual_series(
            weeks_pre_cutoff,
            deep_sleep_pre,
            visceral_pre,
            plot_paths.weekly_deep_sleep_vs_visceral_fat_pre_cutoff,
            primary_label="Sleep Deep Sleep Duration (min)",
            primary_unit="min",
            secondary_label="Visceral Fat Index",
            secondary_unit="index",
            secondary_color="#8c564b",
        )
        print(f"Saved plot to {plot_paths.weekly_deep_sleep_vs_visceral_fat_pre_cutoff}")

        corr_pre = float(np.corrcoef(deep_sleep_pre, visceral_pre)[0, 1])
        label = f"Visceral Fat Index (<= {cutoff_date.isoformat()})"
        results.append((label, corr_pre, len(filtered_pairs)))

    return results


def _compute_lagged_metrics(
    weekly_series: dict[str, tuple[list[date], list[float]]],
) -> list[tuple[str, float, int]]:
    weight_series = weekly_series.get("Weight")
    if not weight_series:
        return []

    results: list[tuple[str, float, int]] = []
    restfulness_series = weekly_series.get("Sleep Contributor: Restfulness")
    if restfulness_series:
        corr, count = compute_lagged_correlation(
            weight_series[0],
            weight_series[1],
            restfulness_series[0],
            restfulness_series[1],
        )
        if corr is not None:
            results.append(("Weight (week t) vs Restfulness (week t+1)", corr, count))

    spo2_series = weekly_series.get("SpO2")
    if spo2_series:
        corr, count = compute_lagged_correlation(
            weight_series[0],
            weight_series[1],
            spo2_series[0],
            spo2_series[1],
        )
        if corr is not None:
            results.append(("Weight (week t) vs SpO2 (week t+1)", corr, count))

    return results


if __name__ == "__main__":
    main()
