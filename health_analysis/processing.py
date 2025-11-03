"""Aggregation and statistical helpers."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import date, timedelta
import math
from statistics import NormalDist
from typing import Iterable

import numpy as np

try:  # Optional dependency for exact p-values
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover - SciPy is optional
    _scipy_stats = None

_NORMAL_DIST = NormalDist()


def aggregate_daily(records: Iterable[tuple[date, float]]) -> tuple[list[date], list[float]]:
    totals: defaultdict[date, float] = defaultdict(float)
    counts: defaultdict[date, int] = defaultdict(int)

    for day, value in records:
        totals[day] += value
        counts[day] += 1

    ordered_days = sorted(totals)
    averages = [totals[day] / counts[day] for day in ordered_days]
    return ordered_days, averages


def aggregate_weekly(days: list[date], values: list[float]) -> tuple[list[date], list[float]]:
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
    averaged: list[float] = []
    window_values: deque[float] = deque()
    window_days_queue: deque[date] = deque()

    for day, value in zip(days, values):
        window_values.append(value)
        window_days_queue.append(day)

        while window_days_queue and (day - window_days_queue[0]).days >= window_days:
            window_values.popleft()
            window_days_queue.popleft()

        averaged.append(sum(window_values) / len(window_values))

    return days, averaged


def align_weekly_series(
    base_weeks: list[date],
    base_values: list[float],
    comparison_weeks: list[date],
    comparison_values: list[float],
) -> tuple[list[date], list[float], list[float]]:
    base_lookup = {week: value for week, value in zip(base_weeks, base_values)}
    comparison_lookup = {week: value for week, value in zip(comparison_weeks, comparison_values)}

    overlapping = sorted(base_lookup.keys() & comparison_lookup.keys())
    aligned_weeks: list[date] = []
    aligned_base: list[float] = []
    aligned_comparison: list[float] = []

    for week in overlapping:
        base_value = base_lookup.get(week)
        comparison_value = comparison_lookup.get(week)
        if base_value is None or comparison_value is None:
            continue
        if math.isnan(base_value) or math.isnan(comparison_value):
            continue
        aligned_weeks.append(week)
        aligned_base.append(base_value)
        aligned_comparison.append(comparison_value)

    return aligned_weeks, aligned_base, aligned_comparison


def pearsonr_with_p(
    values_a: Iterable[float], values_b: Iterable[float]
) -> tuple[float, float | None, int]:
    """Return Pearson r, two-tailed p-value, and sample size."""

    arr_a = np.asarray(list(values_a), dtype=float)
    arr_b = np.asarray(list(values_b), dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError("values_a and values_b must have the same length")

    # Filter out NaNs to avoid propagating them into the correlation
    mask = ~(np.isnan(arr_a) | np.isnan(arr_b))
    arr_a = arr_a[mask]
    arr_b = arr_b[mask]

    n = int(arr_a.size)
    if n == 0:
        return float("nan"), None, 0

    corr_matrix = np.corrcoef(arr_a, arr_b)
    corr = float(corr_matrix[0, 1])

    if n < 3 or not math.isfinite(corr):
        return corr, None, n

    if abs(corr) >= 1.0:
        # Perfect correlation (typically due to identical series)
        return math.copysign(1.0, corr), 0.0, n

    if _scipy_stats is not None:  # pragma: no cover - SciPy optional
        _, p_value = _scipy_stats.pearsonr(arr_a, arr_b)
        return corr, float(p_value), n

    # Fisher z-transform approximation (Normal) for the p-value
    fisher_z = math.atanh(corr)
    z_score = abs(fisher_z) * math.sqrt(max(n - 3, 1))
    p_value = 2.0 * (1.0 - _NORMAL_DIST.cdf(z_score))
    return corr, float(p_value), n


def compute_pairwise_correlations(
    series: dict[str, tuple[list[date], list[float]]]
) -> list[tuple[str, str, float, float | None, int]]:
    metrics = list(series.keys())
    results: list[tuple[str, str, float, float | None, int]] = []

    for index, metric_a in enumerate(metrics):
        weeks_a, values_a = series[metric_a]
        map_a = {
            week: value
            for week, value in zip(weeks_a, values_a)
            if value is not None and not math.isnan(value)
        }

        for metric_b in metrics[index + 1 :]:
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
            corr, p_value, sample_size = pearsonr_with_p(values_a_overlap, values_b_overlap)
            if sample_size < 2:
                continue
            results.append((metric_a, metric_b, corr, p_value, sample_size))

    return results


def compute_lagged_correlation(
    base_weeks: list[date],
    base_values: list[float],
    compare_weeks: list[date],
    compare_values: list[float],
    *,
    lag_weeks: int = 1,
) -> tuple[float | None, int, float | None]:
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
        return None, len(aligned_base), None

    corr, p_value, sample_size = pearsonr_with_p(aligned_base, aligned_compare)
    return corr, sample_size, p_value
