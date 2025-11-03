"""CSV loading utilities for Oura exports and scale data (pandas-based)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

BODY_FIELDS = ("weight", "fatRate", "visceralFat")
WEIGHT_MAX_KG = 68.0
BODY_MIN_TIMESTAMP = pd.Timestamp("2020-11-11 19:06:26+00:00")


def load_spo2_records(path: Path) -> list[tuple[date, float]]:
    """Return (date, average_spo2) pairs parsed from the Oura export."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, sep=";", encoding="utf-8")
    if df.empty or "day" not in df.columns or "spo2_percentage" not in df.columns:
        return []

    df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.date
    payloads = df["spo2_percentage"]
    df["average"] = payloads.apply(_parse_spo2_payload)
    df = df.dropna(subset=["day", "average"])

    return [(day, float(value)) for day, value in df[["day", "average"]].itertuples(index=False, name=None)]


def load_spo2_breathing_index(path: Path) -> list[tuple[date, float]]:
    """Return (date, breathing_disturbance_index) pairs."""
    if not path.exists():
        return []

    df = pd.read_csv(path, sep=";", encoding="utf-8")
    if df.empty or "day" not in df.columns or "breathing_disturbance_index" not in df.columns:
        return []

    df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.date
    df["bdi"] = pd.to_numeric(df["breathing_disturbance_index"], errors="coerce")
    df = df.dropna(subset=["day", "bdi"])

    return [(day, float(value)) for day, value in df[["day", "bdi"]].itertuples(index=False, name=None)]


def load_resting_hr_records(path: Path) -> list[tuple[date, float]]:
    """Return (date, resting_heart_rate) pairs parsed from the readiness export."""
    df = _read_oura_csv(path)
    if df.empty or "contributors" not in df.columns:
        return []

    df["resting_hr"] = df["contributors"].apply(_extract_resting_hr)
    df = df.dropna(subset=["resting_hr"])

    return [(day, float(value)) for day, value in df[["day", "resting_hr"]].itertuples(index=False, name=None)]


def load_body_metrics(paths: Iterable[Path]) -> dict[str, list[tuple[date, float]]]:
    """Return daily records for key body-composition metrics."""
    metrics = {field: [] for field in BODY_FIELDS}

    for path in paths:
        df = _read_body_csv(path)
        if df is None or df.empty:
            continue

        if "weight" in df.columns:
            weight_values = pd.to_numeric(df["weight"], errors="coerce")
            # Ignore scale readings that are obviously not the primary user.
            keep_mask = weight_values.lt(WEIGHT_MAX_KG) | weight_values.isna()
            df = df.loc[keep_mask].copy()
            if df.empty:
                continue

        for field in BODY_FIELDS:
            if field not in df.columns:
                continue
            values = pd.to_numeric(df[field], errors="coerce")
            if field != "weight":
                values = values.where(values > 0)
            valid = df["day"].notna() & values.notna()
            for day, value in zip(df.loc[valid, "day"], values[valid]):
                metrics[field].append((day, float(value)))

    return metrics


def load_daily_readiness_metrics(path: Path) -> dict[str, list[tuple[date, float]]]:
    metrics: dict[str, list[tuple[date, float]]] = {}
    df = _read_oura_csv(path)
    if df.empty:
        return metrics

    readiness_fields = [
        ("score", "Readiness Score"),
        ("temperature_deviation", "Readiness Temperature Deviation"),
        ("temperature_trend_deviation", "Readiness Temperature Trend Deviation"),
    ]
    for column, label in readiness_fields:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        for day, value in zip(df["day"], values):
            _append_metric(metrics, label, day, value)

    if "contributors" in df.columns:
        for day, payload in zip(df["day"], df["contributors"]):
            contributors = _load_json_dict(payload)
            for key, value in contributors.items():
                numeric = _coerce_float(value)
                label = f"Readiness Contributor: {key.replace('_', ' ').title()}"
                _append_metric(metrics, label, day, numeric)

    return metrics


def load_daily_sleep_metrics(path: Path) -> dict[str, list[tuple[date, float]]]:
    metrics: dict[str, list[tuple[date, float]]] = {}
    df = _read_oura_csv(path)
    if df.empty:
        return metrics

    if "score" in df.columns:
        values = pd.to_numeric(df["score"], errors="coerce")
        for day, value in zip(df["day"], values):
            _append_metric(metrics, "Sleep Score", day, value)

    if "contributors" in df.columns:
        for day, payload in zip(df["day"], df["contributors"]):
            contributors = _load_json_dict(payload)
            for key, value in contributors.items():
                numeric = _coerce_float(value)
                label = f"Sleep Contributor: {key.replace('_', ' ').title()}"
                _append_metric(metrics, label, day, numeric)

    return metrics


def load_daily_activity_metrics(path: Path) -> dict[str, list[tuple[date, float]]]:
    metrics: dict[str, list[tuple[date, float]]] = {}
    df = _read_oura_csv(path)
    if df.empty:
        return metrics

    exclude = {"id", "day", "timestamp", "class_5_min", "contributors"}
    for column in df.columns:
        if column in exclude:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        label = f"Activity {column.replace('_', ' ').title()}"
        for day, value in zip(df["day"], values):
            _append_metric(metrics, label, day, value)

    if "contributors" in df.columns:
        for day, payload in zip(df["day"], df["contributors"]):
            contributors = _load_json_dict(payload)
            for key, value in contributors.items():
                numeric = _coerce_float(value)
                label = f"Activity Contributor: {key.replace('_', ' ').title()}"
                _append_metric(metrics, label, day, numeric)

    return metrics


def load_sleep_model_metrics(path: Path) -> dict[str, list[tuple[date, float]]]:
    metrics: dict[str, list[tuple[date, float]]] = {}
    df = _read_oura_csv(path)
    if df.empty:
        return metrics

    duration_fields = {
        "deep_sleep_duration": "Sleep Deep Sleep Duration (min)",
        "total_sleep_duration": "Sleep Total Duration (min)",
        "rem_sleep_duration": "Sleep REM Duration (min)",
        "light_sleep_duration": "Sleep Light Duration (min)",
        "awake_time": "Sleep Awake Time (min)",
        "time_in_bed": "Sleep Time In Bed (min)",
        "latency": "Sleep Latency (min)",
    }

    for column, label in duration_fields.items():
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce") / 60.0
        for day, value in zip(df["day"], values):
            _append_metric(metrics, label, day, value)

    value_fields = {
        "efficiency": ("Sleep Efficiency (%)", 1.0),
        "restless_periods": ("Sleep Restless Periods", 1.0),
        "average_breath": ("Sleep Average Breath (rpm)", 1.0),
    }
    for column, (label, scale) in value_fields.items():
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce") * scale
        for day, value in zip(df["day"], values):
            _append_metric(metrics, label, day, value)

    if "movement_30_sec" in df.columns:
        df["movement_index"] = df["movement_30_sec"].apply(_parse_movement_series)
        for day, value in zip(df["day"], df["movement_index"]):
            _append_metric(metrics, "Sleep Movement Index", day, value)

    return metrics


def _parse_spo2_payload(payload: object) -> float | None:
    if not isinstance(payload, str):
        return None
    text = payload.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return _coerce_float(data.get("average"))


def _extract_resting_hr(payload: object) -> float | None:
    data = _load_json_dict(payload)
    return _coerce_float(data.get("resting_heart_rate"))


def _read_oura_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, sep=";", encoding="utf-8")
    if "day" not in df.columns:
        return pd.DataFrame()

    df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.date
    df = df.dropna(subset=["day"])
    return df


def _read_body_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    df = pd.read_csv(path, encoding="utf-8-sig")
    day_series = None
    if "time" in df.columns:
        day_series = pd.to_datetime(df["time"], errors="coerce", utc=True)
    elif "timestamp" in df.columns:
        day_series = pd.to_datetime(df["timestamp"], errors="coerce", unit="s", utc=True)

    if day_series is None:
        return None

    # Ignore legacy records that pre-date the current user.
    recent_mask = day_series >= BODY_MIN_TIMESTAMP
    df = df.loc[recent_mask].copy()
    day_series = day_series.loc[recent_mask]

    df["day"] = day_series.dt.tz_localize(None).dt.date
    df = df.dropna(subset=["day"])
    return df


def _load_json_dict(payload: object) -> dict[str, object]:
    if not isinstance(payload, str):
        return {}
    text = payload.strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        try:
            numeric = float(str(value).strip())
        except (TypeError, ValueError):
            return None
    if pd.isna(numeric):
        return None
    return numeric


def _parse_movement_series(payload: object) -> float | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        digits: list[int] = []
        for char in payload:
            if char.isdigit():
                digits.append(int(char))
        if not digits:
            return None
        return float(sum(digits)) / len(digits)
    return None


def _append_metric(
    container: dict[str, list[tuple[date, float]]],
    metric_name: str,
    day: date,
    value: float | None,
) -> None:
    if value is None or pd.isna(value):
        return
    container.setdefault(metric_name, []).append((day, float(value)))
