#!/usr/bin/env python3
"""Plot blood oxygenation (SpO2) insights from an Oura export."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import os


DEFAULT_CUTOFF_DATE = date(2023, 1, 31)


@dataclass(frozen=True)
class OuraPaths:
    daily_spo2: Path
    daily_readiness: Path
    daily_sleep: Path
    daily_activity: Path
    sleep_model: Path

    @classmethod
    def from_directory(cls, directory: Path) -> "OuraPaths":
        return cls(
            daily_spo2=directory / "dailyspo2.csv",
            daily_readiness=directory / "dailyreadiness.csv",
            daily_sleep=directory / "dailysleep.csv",
            daily_activity=directory / "dailyactivity.csv",
            sleep_model=directory / "sleepmodel.csv",
        )


@dataclass(frozen=True)
class PlotPaths:
    daily_spo2: Path
    weekly_spo2_rolling: Path
    weekly_spo2_vs_resting_hr: Path
    weekly_spo2_vs_weight: Path
    weekly_spo2_vs_body_fat: Path
    weekly_spo2_vs_visceral_fat: Path
    weekly_restfulness_vs_weight: Path
    weekly_restfulness_vs_body_fat: Path
    weekly_deep_sleep_vs_visceral_fat: Path
    weekly_deep_sleep_vs_visceral_fat_pre_cutoff: Path


def build_plot_paths(output_dir: Path) -> PlotPaths:
    return PlotPaths(
        daily_spo2=output_dir / "daily_spo2.png",
        weekly_spo2_rolling=output_dir / "weekly_spo2_rolling.png",
        weekly_spo2_vs_resting_hr=output_dir / "weekly_spo2_vs_resting_hr.png",
        weekly_spo2_vs_weight=output_dir / "weekly_spo2_vs_weight.png",
        weekly_spo2_vs_body_fat=output_dir / "weekly_spo2_vs_body_fat.png",
        weekly_spo2_vs_visceral_fat=output_dir / "weekly_spo2_vs_visceral_fat.png",
        weekly_restfulness_vs_weight=output_dir / "weekly_restfulness_vs_weight.png",
        weekly_restfulness_vs_body_fat=output_dir / "weekly_restfulness_vs_body_fat.png",
        weekly_deep_sleep_vs_visceral_fat=output_dir / "weekly_deep_sleep_vs_visceral_fat.png",
        weekly_deep_sleep_vs_visceral_fat_pre_cutoff=output_dir
        / "weekly_deep_sleep_vs_visceral_fat_pre_cutoff.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Oura SpO₂ trends and correlations against readiness, sleep, activity, and body data.",
    )
    parser.add_argument(
        "--oura-app-data",
        type=Path,
        help="Directory containing Oura CSV exports (dailyspo2.csv, dailyreadiness.csv, etc.).",
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
        type=lambda value: date.fromisoformat(value),
        default=DEFAULT_CUTOFF_DATE,
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
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token used for private dataset access (default: read from HF_TOKEN env var).",
    )
    parser.add_argument(
        "--hf-oura-subdir",
        default="oura",
        help="Relative path inside the Hugging Face dataset containing the Oura CSV exports.",
    )
    parser.add_argument(
        "--hf-scale-files",
        action="append",
        default=["scale/body.csv"],
        help="Relative path(s) inside the Hugging Face dataset for scale CSV files. Repeat to add more.",
    )
    return parser.parse_args()


def load_spo2_records(path: Path) -> list[tuple[date, float]]:
    """Return (date, average_spo2) pairs parsed from the export file."""
    records: list[tuple[date, float]] = []
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        # Skip header row
        header = handle.readline()
        if not header:
            return records

        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split(";", 3)
            if len(parts) < 4:
                continue

            _, _, day_str, spo2_payload = parts
            if not day_str:
                continue

            try:
                day = datetime.strptime(day_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            spo2_payload = spo2_payload.strip()
            if not spo2_payload:
                continue

            try:
                spo2_data = json.loads(spo2_payload)
            except json.JSONDecodeError:
                continue

            average = spo2_data.get("average")
            if average is None:
                continue

            try:
                average_value = float(average)
            except (TypeError, ValueError):
                continue

            records.append((day, average_value))

    return records


def load_spo2_breathing_index(path: Path) -> list[tuple[date, float]]:
    """Return (date, breathing_disturbance_index) pairs."""
    records: list[tuple[date, float]] = []
    if not path.exists():
        return records

    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
        if not header:
            return records

        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split(";", 3)
            if len(parts) < 4:
                continue

            _, bdi_str, day_str, _ = parts
            if not day_str:
                continue

            try:
                day = datetime.strptime(day_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            value = _safe_float(bdi_str)
            if value is None:
                continue

            records.append((day, value))

    return records


def load_resting_hr_records(path: Path) -> list[tuple[date, float]]:
    """Return (date, resting_heart_rate) pairs parsed from the readiness export."""
    records: list[tuple[date, float]] = []
    if not path.exists():
        return records

    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
        if not header:
            return records

        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split(";", 6)
            if len(parts) < 3:
                continue

            _, contributors_payload, day_str, *_ = parts
            if not day_str or not contributors_payload:
                continue

            try:
                day = datetime.strptime(day_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            try:
                contributors = json.loads(contributors_payload)
            except json.JSONDecodeError:
                continue

            value = contributors.get("resting_heart_rate")
            if value is None:
                continue

            try:
                records.append((day, float(value)))
            except (TypeError, ValueError):
                continue

    return records


def load_body_metrics(paths: list[Path]) -> dict[str, list[tuple[date, float]]]:
    """Return daily records for selected body-composition metrics."""
    metrics = {
        "weight": [],
        "fatRate": [],
        "visceralFat": [],
    }

    for path in paths:
        if not path.exists():
            continue

        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                day: date | None = None

                if "time" in row and row["time"]:
                    time_str = row["time"]
                    parsed_dt: datetime | None = None
                    try:
                        parsed_dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S%z")
                    except ValueError:
                        try:
                            parsed_dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            parsed_dt = None
                    if parsed_dt is not None:
                        day = parsed_dt.date()
                elif "timestamp" in row and row["timestamp"]:
                    ts_value = _safe_float(row["timestamp"])
                    if ts_value is not None:
                        try:
                            day = datetime.fromtimestamp(ts_value, tz=timezone.utc).date()
                        except (OverflowError, OSError, ValueError):
                            day = None

                if day is None:
                    continue

                for field, records in metrics.items():
                    value_str = row.get(field)
                    if not value_str:
                        continue

                    try:
                        value = float(value_str)
                    except (TypeError, ValueError):
                        continue

                    if field != "weight" and value <= 0:
                        continue

                    records.append((day, value))

    return metrics


def _safe_float(value: str | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"nan", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _append_metric(
    container: dict[str, list[tuple[date, float]]],
    metric_name: str,
    day: date,
    value: float | None,
) -> None:
    if value is None or math.isnan(value):
        return
    container.setdefault(metric_name, []).append((day, value))


def load_daily_readiness_metrics(path: Path) -> dict[str, list[tuple[date, float]]]:
    metrics: dict[str, list[tuple[date, float]]] = {}
    if not path.exists():
        return metrics

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            day_str = row.get("day")
            if not day_str:
                continue
            try:
                day = datetime.strptime(day_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            score = _safe_float(row.get("score"))
            _append_metric(metrics, "Readiness Score", day, score)

            temp_dev = _safe_float(row.get("temperature_deviation"))
            _append_metric(metrics, "Readiness Temperature Deviation", day, temp_dev)

            temp_trend = _safe_float(row.get("temperature_trend_deviation"))
            _append_metric(metrics, "Readiness Temperature Trend Deviation", day, temp_trend)

            contributors_raw = row.get("contributors", "")
            if contributors_raw:
                try:
                    contributors = json.loads(contributors_raw)
                except json.JSONDecodeError:
                    contributors = {}
                for key, value in contributors.items():
                    numeric_value = _safe_float(value)
                    _append_metric(
                        metrics, f"Readiness Contributor: {key.replace('_', ' ').title()}", day, numeric_value
                    )

    return metrics


def load_daily_sleep_metrics(path: Path) -> dict[str, list[tuple[date, float]]]:
    metrics: dict[str, list[tuple[date, float]]] = {}
    if not path.exists():
        return metrics

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            day_str = row.get("day")
            if not day_str:
                continue
            try:
                day = datetime.strptime(day_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            score = _safe_float(row.get("score"))
            _append_metric(metrics, "Sleep Score", day, score)

            contributors_raw = row.get("contributors", "")
            if contributors_raw:
                try:
                    contributors = json.loads(contributors_raw)
                except json.JSONDecodeError:
                    contributors = {}
                for key, value in contributors.items():
                    numeric_value = _safe_float(value)
                    _append_metric(
                        metrics, f"Sleep Contributor: {key.replace('_', ' ').title()}", day, numeric_value
                    )

    return metrics


def load_daily_activity_metrics(path: Path) -> dict[str, list[tuple[date, float]]]:
    metrics: dict[str, list[tuple[date, float]]] = {}
    if not path.exists():
        return metrics

    exclude_keys = {
        "id",
        "day",
        "timestamp",
        "class_5_min",
        "contributors",
    }

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            day_str = row.get("day")
            if not day_str:
                continue
            try:
                day = datetime.strptime(day_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            for key, value in row.items():
                if key in exclude_keys:
                    continue
                numeric_value = _safe_float(value)
                _append_metric(
                    metrics, f"Activity {key.replace('_', ' ').title()}", day, numeric_value
                )

            contributors_raw = row.get("contributors", "")
            if contributors_raw:
                try:
                    contributors = json.loads(contributors_raw)
                except json.JSONDecodeError:
                    contributors = {}
                for key, value in contributors.items():
                    numeric_value = _safe_float(value)
                    _append_metric(
                        metrics,
                        f"Activity Contributor: {key.replace('_', ' ').title()}",
                        day,
                        numeric_value,
                    )

    return metrics


def load_sleep_model_metrics(path: Path) -> dict[str, list[tuple[date, float]]]:
    metrics: dict[str, list[tuple[date, float]]] = {}
    if not path.exists():
        return metrics

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            day_str = row.get("day")
            if not day_str:
                continue
            try:
                day = datetime.strptime(day_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            def add_duration(field: str, label: str) -> None:
                value_seconds = _safe_float(row.get(field))
                if value_seconds is None:
                    return
                minutes = value_seconds / 60.0
                _append_metric(metrics, label, day, minutes)

            add_duration("deep_sleep_duration", "Sleep Deep Sleep Duration (min)")
            add_duration("total_sleep_duration", "Sleep Total Duration (min)")
            add_duration("rem_sleep_duration", "Sleep REM Duration (min)")
            add_duration("light_sleep_duration", "Sleep Light Duration (min)")
            add_duration("awake_time", "Sleep Awake Time (min)")
            add_duration("time_in_bed", "Sleep Time In Bed (min)")
            add_duration("latency", "Sleep Latency (min)")

    return metrics


def aggregate_daily(records: list[tuple[date, float]]) -> tuple[list[date], list[float]]:
    """Aggregate duplicates by averaging values on the same day."""
    if not records:
        return [], []

    totals: defaultdict[date, float] = defaultdict(float)
    counts: defaultdict[date, int] = defaultdict(int)

    for day, value in records:
        totals[day] += value
        counts[day] += 1

    ordered_days = sorted(totals)
    averages = [totals[day] / counts[day] for day in ordered_days]

    return ordered_days, averages


def aggregate_weekly(days: list[date], values: list[float]) -> tuple[list[date], list[float]]:
    """Aggregate values into Monday-based ISO weeks."""
    if not days:
        return [], []

    totals: defaultdict[date, float] = defaultdict(float)
    counts: defaultdict[date, int] = defaultdict(int)

    for day, value in zip(days, values):
        week_start = day - timedelta(days=day.weekday())
        totals[week_start] += value
        counts[week_start] += 1

    ordered_weeks = sorted(totals)
    averages = [totals[week] / counts[week] for week in ordered_weeks]
    return ordered_weeks, averages


def compute_rolling_average(
    days: list[date], values: list[float], window_days: int = 7
) -> tuple[list[date], list[float]]:
    """Compute a rolling average constrained to the previous `window_days` calendar days."""
    if not days:
        return [], []

    averaged: list[float] = []
    window_values: deque[float] = deque()
    window_days_queue: deque[date] = deque()

    for day, value in zip(days, values):
        window_values.append(value)
        window_days_queue.append(day)

        while (day - window_days_queue[0]).days >= window_days:
            window_values.popleft()
            window_days_queue.popleft()

        averaged.append(sum(window_values) / len(window_values))

    return days, averaged


def plot_line(
    x_dates: list[date],
    y_values: list[float],
    output_path: Path,
    *,
    title: str,
    y_label: str,
    color: str = "#1f77b4",
    linestyle: str = "-",
    marker: str = "o",
    markersize: float = 3.0,
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(
        x_dates,
        y_values,
        color=color,
        linestyle=linestyle,
        linewidth=1.5,
        marker=marker,
        markersize=markersize,
    )
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def align_weekly_series(
    spo2_weeks: list[date],
    spo2_values: list[float],
    metric_weeks: list[date],
    metric_values: list[float],
) -> tuple[list[date], list[float], list[float]]:
    """Align weekly SpO2 data with another metric, keeping only overlapping weeks."""
    lookup_spo2 = {week: value for week, value in zip(spo2_weeks, spo2_values)}
    lookup_metric = {week: value for week, value in zip(metric_weeks, metric_values)}

    overlapping_weeks = sorted(lookup_spo2.keys() & lookup_metric.keys())
    aligned_weeks: list[date] = []
    aligned_spo2: list[float] = []
    aligned_metric: list[float] = []

    for week in overlapping_weeks:
        spo2_value = lookup_spo2.get(week)
        metric_value = lookup_metric.get(week)

        if spo2_value is None or metric_value is None:
            continue
        if math.isnan(spo2_value) or math.isnan(metric_value):
            continue

        aligned_weeks.append(week)
        aligned_spo2.append(spo2_value)
        aligned_metric.append(metric_value)

    return aligned_weeks, aligned_spo2, aligned_metric


def plot_weekly_spo2_vs_metric(
    week_starts: list[date],
    weekly_spo2: list[float],
    weekly_metric: list[float],
    output_path: Path,
    *,
    metric_label: str,
    metric_unit: str,
    metric_color: str = "#d62728",
    marker: str = "o",
    markersize: float = 3.0,
) -> None:
    plt.figure(figsize=(12, 5))
    ax1 = plt.gca()
    ax1.plot(
        week_starts,
        weekly_spo2,
        color="#1f77b4",
        linewidth=1.8,
        label="SpO2",
        marker=marker,
        markersize=markersize,
    )
    ax1.set_xlabel("Week Starting")
    ax1.set_ylabel("Average SpO2 (%)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(
        week_starts,
        weekly_metric,
        color=metric_color,
        linewidth=1.8,
        linestyle="--",
        label=metric_label,
        marker=marker,
        markersize=markersize,
    )
    ax2.set_ylabel(f"{metric_label} ({metric_unit})", color=metric_color)
    ax2.tick_params(axis="y", labelcolor=metric_color)

    plt.title(f"Weekly Averages: SpO2 vs {metric_label}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_weekly_dual_metric(
    week_starts: list[date],
    primary_values: list[float],
    secondary_values: list[float],
    output_path: Path,
    *,
    primary_label: str,
    primary_unit: str,
    secondary_label: str,
    secondary_unit: str,
    secondary_color: str = "#d62728",
    marker: str = "o",
    markersize: float = 3.0,
) -> None:
    plt.figure(figsize=(12, 5))
    ax1 = plt.gca()
    ax1.plot(
        week_starts,
        primary_values,
        color="#1f77b4",
        linewidth=1.8,
        label=primary_label,
        marker=marker,
        markersize=markersize,
    )
    ax1.set_xlabel("Week Starting")
    ax1.set_ylabel(f"{primary_label} ({primary_unit})", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(
        week_starts,
        secondary_values,
        color=secondary_color,
        linewidth=1.8,
        linestyle="--",
        label=secondary_label,
        marker=marker,
        markersize=markersize,
    )
    ax2.set_ylabel(f"{secondary_label} ({secondary_unit})", color=secondary_color)
    ax2.tick_params(axis="y", labelcolor=secondary_color)

    plt.title(f"Weekly Averages: {primary_label} vs {secondary_label}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compute_pairwise_correlations(
    series: dict[str, tuple[list[date], list[float]]]
) -> list[tuple[str, str, float, int]]:
    """Compute pairwise Pearson correlations for overlapping weeks."""
    metrics = list(series.keys())
    results: list[tuple[str, str, float, int]] = []

    for i, metric_a in enumerate(metrics):
        weeks_a, values_a = series[metric_a]
        map_a = {
            week: value
            for week, value in zip(weeks_a, values_a)
            if value is not None and not math.isnan(value)
        }

        for metric_b in metrics[i + 1 :]:
            weeks_b, values_b = series[metric_b]
            map_b = {
                week: value
                for week, value in zip(weeks_b, values_b)
                if value is not None and not math.isnan(value)
            }

            overlapping = sorted(map_a.keys() & map_b.keys())
            if len(overlapping) < 2:
                continue

            values_a_overlap = [map_a[week] for week in overlapping]
            values_b_overlap = [map_b[week] for week in overlapping]
            corr_matrix = np.corrcoef(values_a_overlap, values_b_overlap)
            corr = float(corr_matrix[0, 1])
            results.append((metric_a, metric_b, corr, len(overlapping)))

    return results


def compute_lagged_correlation(
    base_weeks: list[date],
    base_values: list[float],
    compare_weeks: list[date],
    compare_values: list[float],
    *,
    lag_weeks: int = 1,
) -> tuple[float | None, int]:
    if not base_weeks or not compare_weeks:
        return None, 0

    lag_delta = timedelta(weeks=lag_weeks)
    base_map = {
        week: value
        for week, value in zip(base_weeks, base_values)
        if value is not None and not math.isnan(value)
    }
    compare_map = {
        week: value
        for week, value in zip(compare_weeks, compare_values)
        if value is not None and not math.isnan(value)
    }

    aligned_base: list[float] = []
    aligned_compare: list[float] = []

    for week, value in base_map.items():
        target_week = week + lag_delta
        target_value = compare_map.get(target_week)
        if target_value is None or math.isnan(target_value):
            continue
        aligned_base.append(value)
        aligned_compare.append(target_value)

    if len(aligned_base) < 2:
        return None, len(aligned_base)

    corr = float(np.corrcoef(aligned_base, aligned_compare)[0, 1])
    return corr, len(aligned_base)


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = build_plot_paths(output_dir)

    oura_dir: Path | None = None
    body_files: list[Path] = []

    if args.hf_repo_id:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SystemExit(
                "huggingface_hub is required to download data from the Hub. "
                "Install it with `uv run --with huggingface_hub ...`."
            ) from exc

        repo_path = Path(
            snapshot_download(
                repo_id=args.hf_repo_id,
                repo_type="dataset",
                revision=args.hf_revision,
                token=args.hf_token,
            )
        )
        remote_oura_dir = (repo_path / args.hf_oura_subdir).resolve()
        if not remote_oura_dir.is_dir():
            raise SystemExit(
                f"Oura data directory '{args.hf_oura_subdir}' not found in Hugging Face dataset {args.hf_repo_id}."
            )
        oura_dir = remote_oura_dir

        for relative_path in args.hf_scale_files:
            scale_path = (repo_path / relative_path).resolve()
            if scale_path.exists():
                body_files.append(scale_path)
            else:
                print(
                    f"Warning: scale file '{relative_path}' not present in Hugging Face dataset {args.hf_repo_id}."
                )

    if args.oura_app_data is not None:
        local_oura_dir = args.oura_app_data.expanduser().resolve()
        if not local_oura_dir.is_dir():
            raise SystemExit(f"Oura app data directory not found: {local_oura_dir}")
        # Prefer local data if both sources provided.
        oura_dir = local_oura_dir

    if oura_dir is None:
        raise SystemExit(
            "No Oura data directory available. Provide --oura-app-data or --hf-repo-id with --hf-oura-subdir."
        )

    oura_paths = OuraPaths.from_directory(oura_dir)

    if not oura_paths.daily_spo2.exists():
        raise SystemExit(f"dailyspo2.csv not found in {oura_dir}")

    optional_oura_files = [
        ("daily readiness", oura_paths.daily_readiness),
        ("daily sleep", oura_paths.daily_sleep),
        ("daily activity", oura_paths.daily_activity),
        ("sleep model", oura_paths.sleep_model),
    ]
    for label, path in optional_oura_files:
        if not path.exists():
            print(f"Warning: {label} export not found, skipping: {path}")

    for body_path in args.body_files:
        resolved = body_path.expanduser().resolve()
        if resolved.exists():
            body_files.append(resolved)
        else:
            print(f"Warning: body composition file not found, skipping: {resolved}")
    if not body_files:
        print("Warning: no body composition files provided; skipping weight/body-fat comparisons.")

    cutoff_date: date = args.cutoff_date

    spo2_records = load_spo2_records(oura_paths.daily_spo2)
    if not spo2_records:
        raise SystemExit("No blood oxygenation records found in the export.")

    readiness_records = load_resting_hr_records(oura_paths.daily_readiness)

    daily_days, daily_spo2 = aggregate_daily(spo2_records)
    plot_line(
        daily_days,
        daily_spo2,
        plot_paths.daily_spo2,
        title="Daily Blood Oxygenation (SpO2)",
        y_label="Average SpO2 (%)",
    )
    print(f"Saved plot to {plot_paths.daily_spo2}")

    rolling_days, rolling_spo2 = compute_rolling_average(daily_days, daily_spo2)
    plot_line(
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
    bdi_records = load_spo2_breathing_index(oura_paths.daily_spo2)
    if bdi_records:
        bdi_days, bdi_values = aggregate_daily(bdi_records)
        weekly_bdi_weeks, weekly_bdi_values = aggregate_weekly(bdi_days, bdi_values)
        if weekly_bdi_weeks:
            weekly_series["Breathing Disturbance Index"] = (weekly_bdi_weeks, weekly_bdi_values)
    resting_daily_days, resting_daily_hr = aggregate_daily(readiness_records)
    weekly_hr_weeks, weekly_resting_hr = aggregate_weekly(resting_daily_days, resting_daily_hr)

    weeks_hr, aligned_spo2_hr, aligned_hr = align_weekly_series(
        weekly_spo2_weeks, weekly_spo2, weekly_hr_weeks, weekly_resting_hr
    )
    if weeks_hr:
        plot_weekly_spo2_vs_metric(
            weeks_hr,
            aligned_spo2_hr,
            aligned_hr,
            plot_paths.weekly_spo2_vs_resting_hr,
            metric_label="Resting HR",
            metric_unit="bpm",
        )
        print(f"Saved plot to {plot_paths.weekly_spo2_vs_resting_hr}")
    else:
        print("Skipping resting HR comparison plot (no overlapping weeks).")

    readiness_metrics = load_daily_readiness_metrics(oura_paths.daily_readiness)
    sleep_metrics = load_daily_sleep_metrics(oura_paths.daily_sleep)
    activity_metrics = load_daily_activity_metrics(oura_paths.daily_activity)
    sleep_model_metrics = load_sleep_model_metrics(oura_paths.sleep_model)

    for metric_map in (readiness_metrics, sleep_metrics, activity_metrics, sleep_model_metrics):
        for metric_name, records in metric_map.items():
            daily_metric_days, daily_metric_values = aggregate_daily(records)
            weekly_metric_weeks, weekly_metric_values = aggregate_weekly(
                daily_metric_days, daily_metric_values
            )
            if not weekly_metric_weeks:
                continue
            weekly_series[metric_name] = (weekly_metric_weeks, weekly_metric_values)

    unique_body_files = list(dict.fromkeys(body_files))
    body_metrics = load_body_metrics(unique_body_files) if unique_body_files else {}
    body_metric_configs = [
        ("weight", "Weight", "kg", plot_paths.weekly_spo2_vs_weight, "#9467bd"),
        ("fatRate", "Body Fat %", "%", plot_paths.weekly_spo2_vs_body_fat, "#ff7f0e"),
        ("visceralFat", "Visceral Fat Index", "index", plot_paths.weekly_spo2_vs_visceral_fat, "#8c564b"),
    ]

    for field, label, unit, output_path, color in body_metric_configs:
        records = body_metrics.get(field, [])
        if not records:
            print(f"Skipping {label} comparison plot (no data).")
            continue

        daily_metric_days, daily_metric_values = aggregate_daily(records)
        weekly_metric_weeks, weekly_metric_values = aggregate_weekly(
            daily_metric_days, daily_metric_values
        )
        if weekly_metric_weeks:
            weekly_series[label] = (weekly_metric_weeks, weekly_metric_values)

        weeks_metric, aligned_spo2_metric, aligned_metric = align_weekly_series(
            weekly_spo2_weeks, weekly_spo2, weekly_metric_weeks, weekly_metric_values
        )
        if not weeks_metric:
            print(f"Skipping {label} comparison plot (no overlapping weeks).")
            continue

        plot_weekly_spo2_vs_metric(
            weeks_metric,
            aligned_spo2_metric,
            aligned_metric,
            output_path,
            metric_label=label,
            metric_unit=unit,
            metric_color=color,
        )
        print(f"Saved plot to {output_path}")

    restfulness_analysis: list[tuple[str, float, int]] = []
    restfulness_key = "Sleep Contributor: Restfulness"
    restfulness_series = weekly_series.get(restfulness_key)
    if restfulness_series:
        restfulness_weeks, restfulness_values = restfulness_series

        def handle_restfulness_comparison(
            target_label: str,
            target_unit: str,
            output_path: Path,
            color: str,
        ) -> None:
            target_series = weekly_series.get(target_label)
            if not target_series:
                print(f"Skipping Restfulness vs {target_label} plot (no data).")
                return

            weeks_aligned, restfulness_aligned, target_aligned = align_weekly_series(
                restfulness_weeks,
                restfulness_values,
                target_series[0],
                target_series[1],
            )

            if weeks_aligned:
                plot_weekly_dual_metric(
                    weeks_aligned,
                    restfulness_aligned,
                    target_aligned,
                    output_path,
                    primary_label="Sleep Contributor: Restfulness",
                    primary_unit="score",
                    secondary_label=target_label,
                    secondary_unit=target_unit,
                    secondary_color=color,
                )
                print(f"Saved plot to {output_path.resolve()}")
                corr = float(np.corrcoef(restfulness_aligned, target_aligned)[0, 1])
                restfulness_analysis.append((target_label, corr, len(weeks_aligned)))
            else:
                print(f"Skipping Restfulness vs {target_label} plot (no overlapping weeks).")

        handle_restfulness_comparison(
            "Weight",
            "kg",
            plot_paths.weekly_restfulness_vs_weight,
            "#9467bd",
        )
        handle_restfulness_comparison(
            "Body Fat %",
            "%",
            plot_paths.weekly_restfulness_vs_body_fat,
            "#ff7f0e",
        )
    else:
        print("No Restfulness contributor data found; skipping restfulness comparisons.")

    deep_sleep_analysis: list[tuple[str, float, int]] = []
    deep_sleep_key = "Sleep Deep Sleep Duration (min)"
    deep_sleep_series = weekly_series.get(deep_sleep_key)
    visceral_series = weekly_series.get("Visceral Fat Index")
    if deep_sleep_series and visceral_series:
        weeks_aligned, deep_sleep_aligned, visceral_aligned = align_weekly_series(
            deep_sleep_series[0],
            deep_sleep_series[1],
            visceral_series[0],
            visceral_series[1],
        )
        if weeks_aligned:
            plot_weekly_dual_metric(
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
            corr = float(np.corrcoef(deep_sleep_aligned, visceral_aligned)[0, 1])
            deep_sleep_analysis.append(("Visceral Fat Index", corr, len(weeks_aligned)))

            filtered_pairs = [
                (week, ds, vf)
                for week, ds, vf in zip(weeks_aligned, deep_sleep_aligned, visceral_aligned)
                if week <= cutoff_date
            ]
            if len(filtered_pairs) >= 2:
                weeks_pre2023 = [week for week, _, _ in filtered_pairs]
                deep_sleep_pre2023 = [ds for _, ds, _ in filtered_pairs]
                visceral_pre2023 = [vf for _, _, vf in filtered_pairs]
                plot_weekly_dual_metric(
                    weeks_pre2023,
                    deep_sleep_pre2023,
                    visceral_pre2023,
                    plot_paths.weekly_deep_sleep_vs_visceral_fat_pre_cutoff,
                    primary_label="Sleep Deep Sleep Duration (min)",
                    primary_unit="min",
                    secondary_label="Visceral Fat Index",
                    secondary_unit="index",
                    secondary_color="#8c564b",
                )
                print(f"Saved plot to {plot_paths.weekly_deep_sleep_vs_visceral_fat_pre_cutoff}")
                corr_pre2023 = float(
                    np.corrcoef(deep_sleep_pre2023, visceral_pre2023)[0, 1]
                )
                deep_sleep_analysis.append(
                    (f"Visceral Fat Index (<= {cutoff_date.isoformat()})", corr_pre2023, len(filtered_pairs))
                )
        else:
            print("Skipping Deep Sleep vs Visceral Fat plot (no overlapping weeks).")
    else:
        print("Deep sleep duration or visceral fat data missing; skipping their comparison.")

    lagged_analysis: list[tuple[str, float, int]] = []
    weight_series = weekly_series.get("Weight")
    if weight_series:
        if restfulness_series:
            lag_corr, lag_count = compute_lagged_correlation(
                weight_series[0], weight_series[1], restfulness_series[0], restfulness_series[1]
            )
            if lag_corr is not None:
                lagged_analysis.append(("Weight (week t) vs Restfulness (week t+1)", lag_corr, lag_count))
        spo2_series = weekly_series.get("SpO2")
        if spo2_series:
            lag_corr, lag_count = compute_lagged_correlation(
                weight_series[0], weight_series[1], spo2_series[0], spo2_series[1]
            )
            if lag_corr is not None:
                lagged_analysis.append(("Weight (week t) vs SpO2 (week t+1)", lag_corr, lag_count))

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
                other_metric = metric_b if metric_a == "SpO2" else metric_a
                print(f"  SpO2 vs {other_metric}: {corr:.3f} (n={count})")
        else:
            print("No overlapping weekly data available to correlate with SpO2.")
        if restfulness_analysis:
            print("Restfulness weekly correlations:")
            for label, corr, count in restfulness_analysis:
                print(f"  Restfulness vs {label}: {corr:.3f} (n={count})")
        if deep_sleep_analysis:
            print("Deep sleep weekly correlations:")
            for label, corr, count in deep_sleep_analysis:
                print(f"  Deep Sleep vs {label}: {corr:.3f} (n={count})")
        if lagged_analysis:
            print("Lagged weekly correlations (1-week lead):")
            for label, corr, count in lagged_analysis:
                print(f"  {label}: {corr:.3f} (pairs={count})")
        else:
            print("Lagged weekly correlations: insufficient overlapping data.")
    else:
        print("Insufficient overlapping weekly data to compute correlations.")


if __name__ == "__main__":
    main()
