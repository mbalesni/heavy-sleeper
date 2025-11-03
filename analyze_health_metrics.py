#!/usr/bin/env python3
"""Analyse wearable and scale exports and generate plots/correlations."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import re
from typing import Sequence

import numpy as np

from health_analysis.constants import (
    DEVICE_OURA,
    DEEP_SLEEP_LABEL,
    OURA_SLEEP_METRIC_UNITS,
    RESTFULNESS_LABEL,
    RESTING_HR_LABEL,
    RESTLESS_PERIODS_LABEL,
    RESTLESS_RATE_LABEL,
    TOTAL_SLEEP_DURATION_LABEL,
    SPO2_LABEL,
    WEIGHT_LABEL,
    format_metric,
)
from health_analysis.data_sources import resolve_data_paths
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
from health_analysis.plotting import (
    PlotPaths,
    build_plot_paths,
    plot_correlation_heatmap,
    plot_dual_series,
    plot_scatter_with_regression,
    plot_single_series,
)
from health_analysis.processing import (
    aggregate_daily,
    aggregate_weekly,
    align_weekly_series,
    compute_lagged_correlation,
    compute_pairwise_correlations,
    compute_rolling_average,
    pearsonr_with_p,
)

SLEEP_METRIC_UNITS: dict[str, str] = {
    format_metric(metric_name, DEVICE_OURA): unit
    for metric_name, unit in OURA_SLEEP_METRIC_UNITS.items()
}

TARGET_METRIC_UNITS: dict[str, str] = {
    SPO2_LABEL: "%",
    WEIGHT_LABEL: "kg",
}


def _format_p_value(p_value: float | None) -> str:
    if p_value is None:
        return "p=?"
    if p_value < 0.0001:
        return "p<0.0001"
    if p_value < 0.001:
        return "p<0.001"
    if p_value < 0.01:
        return f"p={p_value:.3f}"
    return f"p={p_value:.4f}"


def _significance_marker(p_value: float | None) -> str:
    if p_value is None:
        return ""
    if p_value < 0.001:
        return " ***"
    if p_value < 0.01:
        return " **"
    if p_value < 0.05:
        return " *"
    return ""


def _format_stats_text(
    r_value: float,
    p_value: float | None,
    sample_size: int,
    *,
    sample_label: str = "week",
) -> str:
    plural_label = sample_label if sample_size == 1 else f"{sample_label}s"
    stats = f"r={r_value:.3f}, {_format_p_value(p_value)}, {sample_size} {plural_label}"
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot wearable SpO₂ trends and correlations against readiness, sleep, activity, and scale data.",
    )
    parser.add_argument(
        "--oura-app-data",
        type=Path,
        default=None,
        help=(
            "Directory containing the wearable (Oura-format) CSV exports. "
            "Defaults to data/oura when omitted."
        ),
    )
    parser.add_argument(
        "--body-file",
        dest="body_files",
        action="append",
        type=Path,
        default=[],
        metavar="PATH",
        help=(
            "Body composition CSV from the scale. Repeat this flag for multiple files. "
            "Defaults to data/scale/body.csv when omitted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where plots will be written (default: ./plots).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    plot_paths: PlotPaths = build_plot_paths(output_dir)

    wearable_paths, body_files = resolve_data_paths(
        oura_app_data=args.oura_app_data,
        body_files=args.body_files or None,
    )

    spo2_records = load_spo2_records(wearable_paths["daily_spo2"])
    if not spo2_records:
        raise SystemExit("No blood oxygenation records found in the export.")

    readiness_records = load_resting_hr_records(wearable_paths["daily_readiness"])

    daily_days, daily_spo2 = aggregate_daily(spo2_records)
    plot_single_series(
        daily_days,
        daily_spo2,
        plot_paths.daily_spo2,
        title=f"Daily Blood Oxygenation ({SPO2_LABEL})",
        y_label=f"Average {SPO2_LABEL} (%)",
    )
    print(f"Saved plot to {plot_paths.daily_spo2}")

    rolling_days, rolling_spo2 = compute_rolling_average(daily_days, daily_spo2)
    plot_single_series(
        rolling_days,
        rolling_spo2,
        plot_paths.weekly_spo2_rolling,
        title=f"Rolling 7-Day Average Blood Oxygenation ({SPO2_LABEL})",
        y_label=f"Rolling Average {SPO2_LABEL} (%)",
        color="#2ca02c",
    )
    print(f"Saved plot to {plot_paths.weekly_spo2_rolling}")

    weekly_spo2_weeks, weekly_spo2 = aggregate_weekly(daily_days, daily_spo2)
    weekly_series: dict[str, tuple[list[date], list[float]]] = {
        SPO2_LABEL: (weekly_spo2_weeks, weekly_spo2)
    }

    # Additional SpO2-derived metrics.
    bdi_records = load_spo2_breathing_index(wearable_paths["daily_spo2"])
    if bdi_records:
        bdi_days, bdi_values = aggregate_daily(bdi_records)
        weekly_bdi_weeks, weekly_bdi_values = aggregate_weekly(bdi_days, bdi_values)
        if weekly_bdi_weeks:
            weekly_series[
                format_metric("Breathing Disturbance Index", DEVICE_OURA)
            ] = (weekly_bdi_weeks, weekly_bdi_values)

    readiness_metrics = load_daily_readiness_metrics(wearable_paths["daily_readiness"])
    sleep_metrics = load_daily_sleep_metrics(wearable_paths["daily_sleep"])
    activity_metrics = load_daily_activity_metrics(wearable_paths["daily_activity"])
    sleep_model_metrics = load_sleep_model_metrics(wearable_paths["sleep_model"])

    for metric_map in (readiness_metrics, sleep_metrics, activity_metrics, sleep_model_metrics):
        _register_weekly_metrics(metric_map.items(), weekly_series)

    rest_days, rest_values = aggregate_daily(readiness_records)
    if rest_days:
        weekly_rest_weeks, weekly_rest_values = aggregate_weekly(rest_days, rest_values)
        if weekly_rest_weeks:
            weekly_series[RESTING_HR_LABEL] = (weekly_rest_weeks, weekly_rest_values)

    body_metrics = load_body_metrics(body_files)
    body_label_map = {
        "weight": WEIGHT_LABEL,
    }
    _register_weekly_metrics(body_metrics.items(), weekly_series, rename=body_label_map)

    _add_restless_period_rate(weekly_series)

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

    beta_metric = _plot_weight_lagged_and_delta_relationships(
        weekly_series,
        plot_paths,
    )

    sleep_metric_summary = _plot_sleep_metric_comparisons(
        weekly_series,
        plot_paths,
    )

    deep_sleep_summary = _plot_deep_sleep_vs_weight(
        weekly_series,
        plot_paths,
    )

    lagged_summary = _compute_lagged_metrics(weekly_series)

    correlations = compute_pairwise_correlations(weekly_series)
    if correlations:
        correlations.sort(key=lambda item: abs(item[2]), reverse=True)
        spo2_correlations = [
            (metric_a, metric_b, corr, p_value, count)
            for metric_a, metric_b, corr, p_value, count in correlations
            if metric_a == SPO2_LABEL or metric_b == SPO2_LABEL
        ]
        if spo2_correlations:
            print(f"Top weekly correlations involving {SPO2_LABEL} (Pearson):")
            for metric_a, metric_b, corr, p_value, count in spo2_correlations[:20]:
                other = metric_b if metric_a == SPO2_LABEL else metric_a
                stats_text = _format_stats_text(corr, p_value, count)
                print(
                    f"  {SPO2_LABEL} vs {other}: {stats_text}{_significance_marker(p_value)}"
                )
        else:
            print(
                f"No overlapping weekly data available to correlate with {SPO2_LABEL}."
            )
    else:
        print("Insufficient overlapping weekly data to compute correlations.")

    if restfulness_summary:
        print(f"{RESTFULNESS_LABEL} weekly correlations:")
        for label, corr, p_value, count in restfulness_summary:
            stats_text = _format_stats_text(corr, p_value, count)
            print(
                f"  {RESTFULNESS_LABEL} vs {label}: {stats_text}{_significance_marker(p_value)}"
            )

    if beta_metric is not None and np.isfinite(beta_metric):
        print(f"β (Restless Periods/hr/kg): {beta_metric:.3f}")

    if sleep_metric_summary:
        print("Extended sleep metric correlations:")
        for metric_label, target_label, corr, p_value, count in sorted(
            sleep_metric_summary, key=lambda item: abs(item[2]), reverse=True
        ):
            stats_text = _format_stats_text(corr, p_value, count)
            print(
                f"  {metric_label} vs {target_label}: {stats_text}{_significance_marker(p_value)}"
            )

    if deep_sleep_summary:
        print(f"{DEEP_SLEEP_LABEL} weekly correlations:")
        for label, corr, p_value, count in deep_sleep_summary:
            stats_text = _format_stats_text(corr, p_value, count)
            print(
                f"  {DEEP_SLEEP_LABEL} vs {label}: {stats_text}{_significance_marker(p_value)}"
            )

    if lagged_summary:
        print("Lagged weekly correlations (1-week lead):")
        for label, corr, p_value, count in lagged_summary:
            stats_text = _format_stats_text(corr, p_value, count, sample_label="pair")
            print(f"  {label}: {stats_text}{_significance_marker(p_value)}")
    else:
        print("Lagged weekly correlations: insufficient overlapping data.")

    print("Significance legend: * p<0.05, ** p<0.01, *** p<0.001")


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


def _add_restless_period_rate(
    weekly_series: dict[str, tuple[list[date], list[float]]]
) -> None:
    restless_series = weekly_series.get(RESTLESS_PERIODS_LABEL)
    duration_series = weekly_series.get(TOTAL_SLEEP_DURATION_LABEL)
    if not restless_series or not duration_series:
        return

    weeks, restless_values, duration_values = align_weekly_series(
        restless_series[0],
        restless_series[1],
        duration_series[0],
        duration_series[1],
    )
    if not weeks:
        return

    rates: list[float] = []
    for restless, duration_min in zip(restless_values, duration_values):
        if duration_min is None or np.isnan(duration_min):
            rates.append(float("nan"))
            continue
        hours = float(duration_min) / 60.0
        if hours <= 0 or np.isnan(restless):
            rates.append(float("nan"))
            continue
        rates.append(float(restless) / hours)

    weekly_series[RESTLESS_RATE_LABEL] = (weeks, rates)


def _plot_resting_hr(
    weekly_spo2_weeks: list[date],
    weekly_spo2: list[float],
    readiness_records: list[tuple[date, float]],
    output_path: Path,
) -> None:
    rest_days, rest_values = aggregate_daily(readiness_records)
    if not rest_days:
        print(f"Skipping {RESTING_HR_LABEL} comparison plot (no readiness data).")
        return

    weekly_rest, weekly_hr = aggregate_weekly(rest_days, rest_values)
    weeks_aligned, aligned_spo2, aligned_hr = align_weekly_series(
        weekly_spo2_weeks,
        weekly_spo2,
        weekly_rest,
        weekly_hr,
    )
    if not weeks_aligned:
        print(f"Skipping {RESTING_HR_LABEL} comparison plot (no overlapping weeks).")
        return

    corr, p_value, sample_size = pearsonr_with_p(aligned_spo2, aligned_hr)
    stats_text = _format_stats_text(corr, p_value, sample_size)

    plot_dual_series(
        weeks_aligned,
        aligned_spo2,
        aligned_hr,
        output_path,
        primary_label=SPO2_LABEL,
        primary_unit="%",
        secondary_label=RESTING_HR_LABEL,
        secondary_unit="bpm",
        stats_text=stats_text,
    )
    print(f"Saved plot to {output_path}")
    print(
        f"{RESTING_HR_LABEL} correlation with {SPO2_LABEL}: "
        f"{stats_text}{_significance_marker(p_value)}"
    )


def _plot_body_comparisons(
    weekly_spo2_weeks: list[date],
    weekly_spo2: list[float],
    body_metrics: dict[str, list[tuple[date, float]]],
    plot_paths: PlotPaths,
) -> None:
    comparisons = [
        ("weight", WEIGHT_LABEL, "kg", plot_paths.weekly_spo2_vs_weight, "#9467bd"),
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

        corr, p_value, sample_size = pearsonr_with_p(aligned_spo2, aligned_metric)
        stats_text = _format_stats_text(corr, p_value, sample_size)

        plot_dual_series(
            weeks_aligned,
            aligned_spo2,
            aligned_metric,
            output_path,
            primary_label=SPO2_LABEL,
            primary_unit="%",
            secondary_label=label,
            secondary_unit=unit,
            secondary_color=color,
            stats_text=stats_text,
        )
        print(f"Saved plot to {output_path}")
        print(
            f"{SPO2_LABEL} vs {label}: {stats_text}{_significance_marker(p_value)}"
        )

        if label == WEIGHT_LABEL:
            all_weeks = sorted(set(weekly_spo2_weeks + weekly_metric_weeks))
            spo2_lookup = {week: value for week, value in zip(weekly_spo2_weeks, weekly_spo2)}
            weight_lookup = {
                week: value for week, value in zip(weekly_metric_weeks, weekly_metric_values)
            }
            spo2_full = [spo2_lookup.get(week, float("nan")) for week in all_weeks]
            weight_full = [weight_lookup.get(week, float("nan")) for week in all_weeks]

            plot_dual_series(
                all_weeks,
                spo2_full,
                weight_full,
                plot_paths.weekly_spo2_vs_weight_all,
                primary_label=SPO2_LABEL,
                primary_unit="%",
                secondary_label=label,
                secondary_unit=unit,
                secondary_color=color,
                stats_text=stats_text,
            )
            print(f"Saved plot to {plot_paths.weekly_spo2_vs_weight_all}")


def _plot_restfulness_comparisons(
    weekly_series: dict[str, tuple[list[date], list[float]]],
    plot_paths: PlotPaths,
) -> list[tuple[str, float, float | None, int]]:
    restfulness_key = RESTFULNESS_LABEL
    restfulness_series = weekly_series.get(restfulness_key)
    if not restfulness_series:
        print(
            f"No {RESTFULNESS_LABEL} data found; skipping restfulness comparisons."
        )
        return []

    restfulness_weeks, restfulness_values = restfulness_series
    results: list[tuple[str, float, float | None, int]] = []

    comparisons = [
        (WEIGHT_LABEL, "kg", plot_paths.weekly_restfulness_vs_weight, "#9467bd"),
    ]

    for target_label, unit, output_path, color in comparisons:
        target_series = weekly_series.get(target_label)
        if not target_series:
            print(
                f"Skipping {RESTFULNESS_LABEL} vs {target_label} plot (no data)."
            )
            continue

        weeks_aligned, restfulness_aligned, target_aligned = align_weekly_series(
            restfulness_weeks,
            restfulness_values,
            target_series[0],
            target_series[1],
        )
        if not weeks_aligned:
            print(
                f"Skipping {RESTFULNESS_LABEL} vs {target_label} plot (no overlapping weeks)."
            )
            continue

        corr, p_value, sample_size = pearsonr_with_p(restfulness_aligned, target_aligned)
        stats_text = _format_stats_text(corr, p_value, sample_size)

        plot_dual_series(
            weeks_aligned,
            restfulness_aligned,
            target_aligned,
            output_path,
            primary_label=RESTFULNESS_LABEL,
            primary_unit="score",
            secondary_label=target_label,
            secondary_unit=unit,
            secondary_color=color,
            stats_text=stats_text,
        )
        print(f"Saved plot to {output_path}")

        results.append((target_label, corr, p_value, sample_size))

    return results


def _plot_weight_lagged_and_delta_relationships(
    weekly_series: dict[str, tuple[list[date], list[float]]],
    plot_paths: PlotPaths,
) -> float | None:
    weight_series = weekly_series.get(WEIGHT_LABEL)
    if not weight_series:
        print(f"No {WEIGHT_LABEL} data found; skipping weight comparison scatter plots.")
        return None

    weight_weeks, weight_values = weight_series
    beta_value: float | None = None

    comparisons: list[tuple[str, Path, Path, str, str, bool]] = [
        (
            RESTFULNESS_LABEL,
            plot_paths.weekly_weight_vs_restfulness,
            plot_paths.weekly_delta_weight_vs_restfulness,
            "#9467bd",
            "#ff9896",
            False,
        ),
        (
            RESTLESS_RATE_LABEL,
            plot_paths.weekly_weight_vs_restless_rate,
            plot_paths.weekly_delta_weight_vs_restless_rate,
            "#1f77b4",
            "#17becf",
            True,
        ),
        (
            RESTLESS_PERIODS_LABEL,
            plot_paths.weekly_weight_vs_restless_periods,
            plot_paths.weekly_delta_weight_vs_restless,
            "#ff7f0e",
            "#c85119",
            False,
        ),
    ]

    for (
        metric_label,
        scatter_path,
        delta_path,
        point_color,
        line_color,
        capture_beta,
    ) in comparisons:
        metric_series = weekly_series.get(metric_label)
        if not metric_series:
            print(f"No {metric_label} data found; skipping scatter plots.")
            continue

        weeks_aligned, weight_aligned, metric_aligned = align_weekly_series(
            weight_weeks,
            weight_values,
            metric_series[0],
            metric_series[1],
        )
        if not weeks_aligned:
            print(f"No overlapping weeks for comparisons ({WEIGHT_LABEL} vs {metric_label}).")
            continue

        weight_arr = np.asarray(weight_aligned, dtype=float)
        metric_arr = np.asarray(metric_aligned, dtype=float)
        mask = ~(np.isnan(weight_arr) | np.isnan(metric_arr))
        weight_arr = weight_arr[mask]
        metric_arr = metric_arr[mask]

        if weight_arr.size < 2:
            print(f"Insufficient overlapping data for scatter ({WEIGHT_LABEL} vs {metric_label}).")
            continue

        corr, p_value, sample_size = pearsonr_with_p(weight_arr, metric_arr)
        stats_text = _format_stats_text(corr, p_value, sample_size)
        slope = plot_scatter_with_regression(
            weight_arr,
            metric_arr,
            scatter_path,
            x_label=f"{WEIGHT_LABEL} (kg)",
            y_label=f"{metric_label}",
            title=f"{WEIGHT_LABEL} vs {metric_label}",
            point_color=point_color,
            line_color=line_color,
            stats_text=stats_text,
        )
        if capture_beta and np.isfinite(slope):
            beta_value = float(slope)

        print(f"Saved plot to {scatter_path}")
        print(
            f"Same-week correlation ({WEIGHT_LABEL} vs {metric_label}): "
            f"{stats_text}{_significance_marker(p_value)}"
        )

        if weight_arr.size >= 3:
            delta_weight = np.diff(weight_arr)
            delta_metric = np.diff(metric_arr)
            mask_delta = ~(np.isnan(delta_weight) | np.isnan(delta_metric))
            delta_weight = delta_weight[mask_delta]
            delta_metric = delta_metric[mask_delta]

            if delta_weight.size >= 2:
                corr_delta, p_delta, sample_delta = pearsonr_with_p(
                    delta_weight, delta_metric
                )
                stats_delta = _format_stats_text(
                    corr_delta, p_delta, sample_delta, sample_label="week-pair"
                )
                plot_scatter_with_regression(
                    delta_weight,
                    delta_metric,
                    delta_path,
                    x_label=f"Δ {WEIGHT_LABEL} (kg)",
                    y_label=f"Δ {metric_label}",
                    title=f"Δ {WEIGHT_LABEL} vs Δ {metric_label}",
                    point_color="#2ca02c",
                    line_color=line_color,
                    stats_text=stats_delta,
                )
                print(f"Saved plot to {delta_path}")
                print(
                    f"Week-over-week Δ correlation ({WEIGHT_LABEL} vs {metric_label}): "
                    f"{stats_delta}{_significance_marker(p_delta)}"
                )
            else:
                print(
                    f"Insufficient week-to-week changes for Δ scatter ({WEIGHT_LABEL} vs {metric_label})."
                )
        else:
            print(
                f"Not enough aligned weeks to compute Δ scatter ({WEIGHT_LABEL} vs {metric_label})."
            )

    return beta_value


def _plot_sleep_metric_comparisons(
    weekly_series: dict[str, tuple[list[date], list[float]]],
    plot_paths: PlotPaths,
) -> list[tuple[str, str, float, int]]:
    metrics_available = sorted(
        ((label, SLEEP_METRIC_UNITS[label]) for label in SLEEP_METRIC_UNITS if label in weekly_series),
        key=lambda item: item[0].lower(),
    )
    if not metrics_available:
        print("No extended sleep metrics available for comparison plots.")
        return []

    targets_available = sorted(
        (
            (label, TARGET_METRIC_UNITS[label])
            for label in TARGET_METRIC_UNITS
            if label in weekly_series
        ),
        key=lambda item: item[0].lower(),
    )
    if not targets_available:
        print("No comparison targets available for sleep metric plots.")
        return []

    corr_matrix = np.full((len(metrics_available), len(targets_available)), np.nan)
    p_matrix = np.full((len(metrics_available), len(targets_available)), np.nan)
    summary: list[tuple[str, str, float, float | None, int]] = []

    for metric_idx, (metric_label, metric_unit) in enumerate(metrics_available):
        metric_weeks, metric_values = weekly_series[metric_label]
        for target_idx, (target_label, target_unit) in enumerate(targets_available):
            target_weeks, target_values = weekly_series[target_label]
            weeks_aligned, metric_aligned, target_aligned = align_weekly_series(
                metric_weeks,
                metric_values,
                target_weeks,
                target_values,
            )

            if len(weeks_aligned) < 2:
                continue

            # Skip self-correlations (e.g. Resting HR (bpm) [Oura] vs itself).
            if metric_label == target_label:
                continue

            output_path = plot_paths.sleep_metric_dir / (
                f"{_slugify(metric_label)}_vs_{_slugify(target_label)}.png"
            )
            corr, p_value, sample_size = pearsonr_with_p(metric_aligned, target_aligned)
            stats_text = _format_stats_text(corr, p_value, sample_size)

            corr_matrix[metric_idx, target_idx] = corr
            if p_value is not None:
                p_matrix[metric_idx, target_idx] = p_value
            summary.append((metric_label, target_label, corr, p_value, sample_size))

            if target_label == RESTING_HR_LABEL:
                # Heatmap/summary only; skip generating individual plots against resting HR.
                continue

            plot_dual_series(
                weeks_aligned,
                metric_aligned,
                target_aligned,
                output_path,
                primary_label=metric_label,
                primary_unit=metric_unit,
                secondary_label=target_label,
                secondary_unit=target_unit,
                stats_text=stats_text,
            )
            print(f"Saved plot to {output_path}")

    if summary and np.isfinite(corr_matrix).any():
        plot_correlation_heatmap(
            corr_matrix,
            [label for label, _ in metrics_available],
            [label for label, _ in targets_available],
            plot_paths.sleep_metric_heatmap,
            p_values=p_matrix,
        )
        print(f"Saved heatmap to {plot_paths.sleep_metric_heatmap}")

    return summary


def _plot_deep_sleep_vs_weight(
    weekly_series: dict[str, tuple[list[date], list[float]]],
    plot_paths: PlotPaths,
) -> list[tuple[str, float, float | None, int]]:
    deep_sleep_key = DEEP_SLEEP_LABEL
    deep_sleep_series = weekly_series.get(deep_sleep_key)
    weight_series = weekly_series.get(WEIGHT_LABEL)
    if not deep_sleep_series or not weight_series:
        print(
            f"{DEEP_SLEEP_LABEL} or {WEIGHT_LABEL} data missing; skipping their comparison."
        )
        return []

    weeks_aligned, deep_sleep_aligned, weight_aligned = align_weekly_series(
        deep_sleep_series[0],
        deep_sleep_series[1],
        weight_series[0],
        weight_series[1],
    )
    if not weeks_aligned:
        print(
            f"Skipping {DEEP_SLEEP_LABEL} vs {WEIGHT_LABEL} plot (no overlapping weeks)."
        )
        return []

    corr, p_value, sample_size = pearsonr_with_p(deep_sleep_aligned, weight_aligned)
    stats_text = _format_stats_text(corr, p_value, sample_size)

    plot_dual_series(
        weeks_aligned,
        deep_sleep_aligned,
        weight_aligned,
        plot_paths.weekly_deep_sleep_vs_weight,
        primary_label=DEEP_SLEEP_LABEL,
        primary_unit="min",
        secondary_label=WEIGHT_LABEL,
        secondary_unit="kg",
        secondary_color="#9467bd",
        stats_text=stats_text,
    )
    print(f"Saved plot to {plot_paths.weekly_deep_sleep_vs_weight}")

    results: list[tuple[str, float, float | None, int]] = []
    results.append((WEIGHT_LABEL, corr, p_value, sample_size))

    return results


def _compute_lagged_metrics(
    weekly_series: dict[str, tuple[list[date], list[float]]],
) -> list[tuple[str, float, float | None, int]]:
    weight_series = weekly_series.get(WEIGHT_LABEL)
    if not weight_series:
        return []

    results: list[tuple[str, float, float | None, int]] = []
    restfulness_series = weekly_series.get(RESTFULNESS_LABEL)
    if restfulness_series:
        corr, count, p_value = compute_lagged_correlation(
            weight_series[0],
            weight_series[1],
            restfulness_series[0],
            restfulness_series[1],
        )
        if corr is not None:
            results.append(
                (
                    f"{WEIGHT_LABEL} (week t) vs {RESTFULNESS_LABEL} (week t+1)",
                    corr,
                    p_value,
                    count,
                )
            )

    spo2_series = weekly_series.get(SPO2_LABEL)
    if spo2_series:
        corr, count, p_value = compute_lagged_correlation(
            weight_series[0],
            weight_series[1],
            spo2_series[0],
            spo2_series[1],
        )
        if corr is not None:
            results.append(
                (
                    f"{WEIGHT_LABEL} (week t) vs {SPO2_LABEL} (week t+1)",
                    corr,
                    p_value,
                    count,
                )
            )

    return results


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower())
    slug = slug.strip("_")
    return slug or "metric"


if __name__ == "__main__":
    main()
