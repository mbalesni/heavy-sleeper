"""Helpers for resolving where to load Oura and scale data from."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class OuraPaths:
    """Absolute paths to the Oura CSV exports we rely on."""

    daily_spo2: Path
    daily_readiness: Path
    daily_sleep: Path
    daily_activity: Path
    sleep_model: Path

    @classmethod
    def from_directory(cls, directory: Path) -> "OuraPaths":
        base = directory.expanduser().resolve()
        return cls(
            daily_spo2=base / "dailyspo2.csv",
            daily_readiness=base / "dailyreadiness.csv",
            daily_sleep=base / "dailysleep.csv",
            daily_activity=base / "dailyactivity.csv",
            sleep_model=base / "sleepmodel.csv",
        )


@dataclass(frozen=True)
class DataContext:
    """Resolved locations for Oura exports and optional scale files."""

    oura: OuraPaths
    body_files: list[Path]


@dataclass(frozen=True)
class HuggingFaceOptions:
    """Configuration for optionally pulling data from the Hugging Face Hub."""

    repo_id: str | None
    revision: str | None
    token: str | None
    oura_subdir: str
    scale_files: tuple[str, ...]

    @property
    def enabled(self) -> bool:
        return bool(self.repo_id)


def _snapshot_hf_dataset(options: HuggingFaceOptions) -> Path:
    """Download a dataset snapshot and return its local path."""
    if not options.repo_id:
        raise ValueError("Cannot snapshot Hugging Face dataset without a repo_id.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "huggingface_hub is required for --hf-repo-id support. "
            "Install it via `uv run --with huggingface_hub ...`."
        ) from exc

    return Path(
        snapshot_download(
            repo_id=options.repo_id,
            repo_type="dataset",
            revision=options.revision,
            token=options.token,
        )
    )


def resolve_data_paths(
    *,
    oura_app_data: Path | None,
    body_files: Sequence[Path],
    hf_options: HuggingFaceOptions,
) -> DataContext:
    """Resolve data sources based on CLI inputs."""

    resolved_body_files: list[Path] = []
    for body_path in body_files:
        expanded = body_path.expanduser().resolve()
        if expanded.exists():
            resolved_body_files.append(expanded)
        else:
            print(f"Warning: body composition file not found, skipping: {expanded}")

    oura_dir: Path | None = None

    if hf_options.enabled:
        repo_path = _snapshot_hf_dataset(hf_options)
        remote_oura_dir = (repo_path / hf_options.oura_subdir).resolve()
        if not remote_oura_dir.is_dir():
            raise SystemExit(
                f"Oura data directory '{hf_options.oura_subdir}' not found in "
                f"Hugging Face dataset {hf_options.repo_id}."
            )
        oura_dir = remote_oura_dir

        for relative_path in hf_options.scale_files:
            candidate = (repo_path / relative_path).resolve()
            if candidate.exists():
                resolved_body_files.append(candidate)
            else:
                print(
                    f"Warning: scale file '{relative_path}' not found in "
                    f"Hugging Face dataset {hf_options.repo_id}."
                )

    if oura_app_data is not None:
        local_dir = oura_app_data.expanduser().resolve()
        if not local_dir.is_dir():
            raise SystemExit(f"Oura app data directory not found: {local_dir}")
        # Local data takes precedence when both sources are provided.
        oura_dir = local_dir

    if oura_dir is None:
        raise SystemExit(
            "No Oura data directory available. Provide --oura-app-data or "
            "--hf-repo-id with --hf-oura-subdir."
        )

    oura_paths = OuraPaths.from_directory(oura_dir)
    if not oura_paths.daily_spo2.exists():
        raise SystemExit(
            f"dailyspo2.csv not found in {oura_dir}. Check that the export is complete."
        )

    unique_body_files = list(dict.fromkeys(resolved_body_files))
    if not unique_body_files:
        print("Warning: no body composition files provided; body-weight plots will be skipped.")

    return DataContext(oura=oura_paths, body_files=unique_body_files)
