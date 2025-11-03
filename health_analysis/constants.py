"""Shared constants and helpers for wearable and scale data analysis."""

from __future__ import annotations

from datetime import date

# Default cutoff for generating the "pre-cutoff" deep sleep plot.
DEFAULT_PRE_CUTOFF = date(2022, 1, 1)

# Device labels used throughout the analysis.
DEVICE_OURA = "Oura"
DEVICE_XIAOMI_SCALE = "Xiaomi Scale"


def format_metric(metric_name: str, device: str) -> str:
    """Return a display label that includes the originating device."""
    return f"{metric_name} [{device}]"


# Canonical sleep metric names (before device tagging) and their units.
OURA_SLEEP_METRIC_UNITS: dict[str, str] = {
    "Efficiency (%)": "%",
    "Restless Periods": "count",
    "Restless Period Rate (per hour)": "count/hr",
    "REM Sleep Duration (min)": "min",
    "Total Sleep Duration (min)": "min",
    "Movement Index": "index",
    "Light Sleep Duration (min)": "min",
    "Awake Time (min)": "min",
    "Average Breath Rate (rpm)": "rpm",
    "Deep Sleep Duration (min)": "min",
    "Restfulness Score": "score",
    "Resting HR (bpm)": "bpm",
}


# Commonly referenced metric labels with device tags.
SPO2_LABEL = format_metric("SpO2", DEVICE_OURA)
WEIGHT_LABEL = format_metric("Weight", DEVICE_XIAOMI_SCALE)
RESTING_HR_LABEL = format_metric("Resting HR (bpm)", DEVICE_OURA)
RESTFULNESS_LABEL = format_metric("Restfulness Score", DEVICE_OURA)
RESTLESS_PERIODS_LABEL = format_metric("Restless Periods", DEVICE_OURA)
RESTLESS_RATE_LABEL = format_metric("Restless Period Rate (per hour)", DEVICE_OURA)
TOTAL_SLEEP_DURATION_LABEL = format_metric("Total Sleep Duration (min)", DEVICE_OURA)
DEEP_SLEEP_LABEL = format_metric("Deep Sleep Duration (min)", DEVICE_OURA)
