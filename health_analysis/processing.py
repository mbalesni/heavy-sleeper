"""Aggregation and statistical helpers."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import date, timedelta
import math
from typing import Iterable

import numpy as np


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


def compute_pairwise_correlations(
    series: dict[str, tuple[list[date], list[float]]]
) -> list[tuple[str, str, float, int]]:
    metrics = list(series.keys())
    results: list[tuple[str, str, float, int]] = []

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
