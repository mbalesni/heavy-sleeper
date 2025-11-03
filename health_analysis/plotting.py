"""Plotting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt


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
    output_dir.mkdir(parents=True, exist_ok=True)
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

    plt.title(f"Weekly Averages: {primary_label} vs {secondary_label}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
