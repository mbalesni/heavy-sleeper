"""Plotting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from statistics import NormalDist

try:  # Optional SciPy for precise confidence intervals
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover - SciPy optional
    _scipy_stats = None

_NORMAL_DIST = NormalDist()


@dataclass(frozen=True)
class PlotPaths:
    daily_spo2: Path
    weekly_spo2_rolling: Path
    weekly_spo2_vs_resting_hr: Path
    weekly_spo2_vs_weight: Path
    weekly_spo2_vs_weight_all: Path
    weekly_restfulness_vs_weight: Path
    weekly_deep_sleep_vs_weight: Path
    weekly_weight_vs_restfulness: Path
    weekly_weight_vs_restless_rate: Path
    weekly_weight_vs_restless_periods: Path
    weekly_delta_weight_vs_restfulness: Path
    weekly_delta_weight_vs_restless_rate: Path
    weekly_delta_weight_vs_restless: Path
    sleep_metric_dir: Path
    sleep_metric_heatmap: Path


def build_plot_paths(output_dir: Path) -> PlotPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    sleep_metric_dir = output_dir / "sleep_metrics"
    sleep_metric_dir.mkdir(exist_ok=True)
    return PlotPaths(
        daily_spo2=output_dir / "daily_spo2.png",
        weekly_spo2_rolling=output_dir / "weekly_spo2_rolling.png",
        weekly_spo2_vs_resting_hr=output_dir / "weekly_spo2_vs_resting_hr.png",
        weekly_spo2_vs_weight=output_dir / "weekly_spo2_vs_weight.png",
        weekly_spo2_vs_weight_all=output_dir / "weekly_spo2_vs_weight_all.png",
        weekly_restfulness_vs_weight=output_dir / "weekly_restfulness_vs_weight.png",
        weekly_deep_sleep_vs_weight=output_dir / "weekly_deep_sleep_vs_weight.png",
        weekly_weight_vs_restfulness=output_dir / "weekly_weight_vs_restfulness.png",
        weekly_weight_vs_restless_rate=output_dir / "weekly_weight_vs_restless_rate.png",
        weekly_weight_vs_restless_periods=output_dir / "weekly_weight_vs_restless_periods.png",
        weekly_delta_weight_vs_restfulness=output_dir / "weekly_delta_weight_vs_delta_restfulness.png",
        weekly_delta_weight_vs_restless_rate=output_dir / "weekly_delta_weight_vs_delta_restless_rate.png",
        weekly_delta_weight_vs_restless=output_dir / "weekly_delta_weight_vs_delta_restless_periods.png",
        sleep_metric_dir=sleep_metric_dir,
        sleep_metric_heatmap=output_dir / "sleep_metric_correlations.png",
    )


def plot_single_series(
    dates: list[date],
    values: list[float],
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
        dates,
        values,
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


def plot_dual_series(
    week_starts: list[date],
    primary_values: list[float],
    secondary_values: list[float],
    output_path: Path,
    *,
    primary_label: str,
    primary_unit: str,
    secondary_label: str,
    secondary_unit: str,
    primary_color: str = "#1f77b4",
    secondary_color: str = "#d62728",
    marker: str = "o",
    markersize: float = 3.0,
    stats_text: str | None = None,
) -> None:
    plt.figure(figsize=(12, 5))
    ax1 = plt.gca()
    ax1.plot(
        week_starts,
        primary_values,
        color=primary_color,
        linewidth=1.8,
        label=primary_label,
        marker=marker,
        markersize=markersize,
    )
    ax1.set_xlabel("Week Starting")
    ax1.set_ylabel(f"{primary_label} ({primary_unit})", color=primary_color)
    ax1.tick_params(axis="y", labelcolor=primary_color)
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

    title = f"Weekly Averages: {primary_label} vs {secondary_label}"
    if stats_text:
        title = f"{title}\n{stats_text}"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_correlation_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    output_path: Path,
    *,
    p_values: np.ndarray | None = None,
) -> None:
    fig_width = max(6, len(column_labels) * 1.5)
    fig_height = max(4, len(row_labels) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    masked = np.ma.masked_invalid(matrix)
    cmap = plt.colormaps.get("coolwarm")
    im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Pearson r", fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for row_idx, row_label in enumerate(row_labels):
        for col_idx, col_label in enumerate(column_labels):
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                continue
            text = f"r={value:.2f}"
            if p_values is not None:
                p_value = p_values[row_idx, col_idx]
                if np.isnan(p_value):
                    text += "\np=?"
                elif p_value < 0.0001:
                    text += "\np<0.0001"
                elif p_value < 0.001:
                    text += "\np<0.001"
                elif p_value < 0.01:
                    text += f"\np={p_value:.3f}"
                else:
                    text += f"\np={p_value:.4f}"
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    max_row_len = max((len(label) for label in row_labels), default=0)
    max_col_len = max((len(label) for label in column_labels), default=0)
    left_margin = max(0.08, min(0.26, 0.08 + 0.004 * max_row_len))
    bottom_margin = max(0.10, min(0.28, 0.10 + 0.006 * max_col_len))
    right_margin = 0.985
    top_margin = 0.97
    fig.subplots_adjust(left=left_margin, bottom=bottom_margin, right=right_margin, top=top_margin)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_with_regression(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
    output_path: Path,
    *,
    x_label: str,
    y_label: str,
    title: str,
    point_color: str = "#1f77b4",
    line_color: str = "#d62728",
    stats_text: str | None = None,
) -> float:
    plt.figure(figsize=(8, 6))

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        plt.close()
        return float("nan")

    ax = plt.gca()
    ax.scatter(x, y, color=point_color, alpha=0.75, edgecolor="none", s=45)

    slope = float("nan")
    intercept = float("nan")

    if x.size >= 2 and np.any(x != x[0]):
        slope, intercept = np.polyfit(x, y, 1)
        x_sorted = np.linspace(np.min(x), np.max(x), 200)
        y_pred = slope * x_sorted + intercept

        ax.plot(x_sorted, y_pred, color=line_color, linewidth=2, label="OLS fit")

        if x.size > 2:
            residuals = y - (slope * x + intercept)
            dof = x.size - 2
            if dof > 0:
                s_err = np.sqrt(np.sum(residuals**2) / dof)
                x_mean = np.mean(x)
                s_xx = np.sum((x - x_mean) ** 2)
                if s_xx > 0 and s_err > 0:
                    if _scipy_stats is not None:
                        t_value = _scipy_stats.t.ppf(0.975, dof)
                    else:
                        t_value = _NORMAL_DIST.inv_cdf(0.975)
                    band = t_value * s_err * np.sqrt(1 / x.size + (x_sorted - x_mean) ** 2 / s_xx)
                    ax.fill_between(
                        x_sorted,
                        y_pred - band,
                        y_pred + band,
                        color=line_color,
                        alpha=0.18,
                        label="95% CI",
                    )

    if np.isfinite(slope):
        beta_text = f"β={slope:.2f}"
        if stats_text:
            stats_text = f"{stats_text}\n{beta_text}"
        else:
            stats_text = beta_text

    if stats_text:
        title = f"{title}\n{stats_text}"

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    if "kg" in x_label:
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return float(slope)
