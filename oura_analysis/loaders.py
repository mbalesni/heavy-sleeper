"""CSV loading utilities for Oura exports and scale data."""

from __future__ import annotations

import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable
import math


def load_spo2_records(path: Path) -> list[tuple[date, float]]:
    """Return (date, average_spo2) pairs parsed from the export file."""
    records: list[tuple[date, float]] = []
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
        if not header:
            return records

        for raw_line in handle:
            parts = raw_line.strip().split(";", 3)
            if len(parts) < 4:
                continue
            _, _, day_str, payload = parts
            if not day_str:
                continue
            try:
                day = datetime.strptime(day_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            payload = payload.strip()
            if not payload:
                continue

            try:
                spo2_data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            average = spo2_data.get("average")
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
            parts = raw_line.strip().split(";", 3)
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
            parts = raw_line.strip().split(";", 6)
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
            try:
                records.append((day, float(value)))
            except (TypeError, ValueError):
                continue

    return records


def load_body_metrics(paths: Iterable[Path]) -> dict[str, list[tuple[date, float]]]:
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
                day = _resolve_body_row_date(row)
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


def _resolve_body_row_date(row: dict[str, str]) -> date | None:
    if "time" in row and row["time"]:
        time_str = row["time"]
        try:
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S%z").date()
        except ValueError:
            try:
                return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                ).date()
            except ValueError:
                return None

    if "timestamp" in row and row["timestamp"]:
        ts_value = _safe_float(row["timestamp"])
        if ts_value is None:
            return None
        try:
            return datetime.fromtimestamp(ts_value, tz=timezone.utc).date()
        except (OverflowError, OSError, ValueError):
            return None

    return None


def _safe_float(value: str | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text or text.lower() in {"nan", "null"}:
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
                        metrics,
                        f"Readiness Contributor: {key.replace('_', ' ').title()}",
                        day,
                        numeric_value,
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
                        metrics,
                        f"Sleep Contributor: {key.replace('_', ' ').title()}",
                        day,
                        numeric_value,
                    )

    return metrics


def load_daily_activity_metrics(path: Path) -> dict[str, list[tuple[date, float]]]:
    metrics: dict[str, list[tuple[date, float]]] = {}
    if not path.exists():
        return metrics

    exclude_keys = {"id", "day", "timestamp", "class_5_min", "contributors"}

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
                    metrics,
                    f"Activity {key.replace('_', ' ').title()}",
                    day,
                    numeric_value,
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
