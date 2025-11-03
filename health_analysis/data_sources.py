"""Simple helpers for locating wearable and scale CSV files on disk."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

DEFAULT_OURA_DIR = Path("data/oura")
DEFAULT_BODY_FILES = (Path("data/scale/body.csv"),)

# Expected CSV filenames from the Oura export directory.
OURA_FILES = {
    "daily_spo2": "dailyspo2.csv",
    "daily_readiness": "dailyreadiness.csv",
    "daily_sleep": "dailysleep.csv",
    "daily_activity": "dailyactivity.csv",
    "sleep_model": "sleepmodel.csv",
}


def resolve_wearable_paths(base_dir: Path) -> dict[str, Path]:
    """Expand the provided directory and ensure all Oura CSVs exist."""
    directory = base_dir.expanduser().resolve()
    if not directory.is_dir():
        raise SystemExit(f"Oura export directory not found: {directory}")

    resolved: dict[str, Path] = {}
    for key, filename in OURA_FILES.items():
        candidate = directory / filename
        if not candidate.exists():
            raise SystemExit(
                f"Expected '{filename}' in {directory}, but it is missing. "
                "Run another export from the Oura app."
            )
        resolved[key] = candidate
    return resolved


def resolve_body_files(body_files: Sequence[Path]) -> list[Path]:
    """Expand and deduplicate scale CSV paths, warning on missing files."""
    resolved: list[Path] = []
    for path in body_files:
        candidate = path.expanduser().resolve()
        if candidate.exists():
            resolved.append(candidate)
        else:
            print(f"Warning: body composition file not found, skipping: {candidate}")

    unique_paths: list[Path] = []
    for candidate in resolved:
        if candidate not in unique_paths:
            unique_paths.append(candidate)
    return unique_paths


def resolve_data_paths(
    *,
    oura_app_data: Path | None,
    body_files: Sequence[Path] | None = None,
) -> tuple[dict[str, Path], list[Path]]:
    """Return absolute paths for the wearable CSVs and any scale CSVs."""
    wearable_paths = resolve_wearable_paths(oura_app_data or DEFAULT_OURA_DIR)

    if body_files:
        resolved_body = resolve_body_files(body_files)
    else:
        resolved_body = resolve_body_files(DEFAULT_BODY_FILES)

    if not resolved_body:
        print("Warning: no body composition files available; body plots will be skipped.")

    return wearable_paths, resolved_body
