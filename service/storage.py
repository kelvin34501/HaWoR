from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


SUPPORTED_VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}
SUPPORTED_VIS_MODES = {"off", "cam", "world"}


@dataclass(frozen=True)
class JobPathPlan:
    input_dir: Path
    output_dir: Path
    video_paths: tuple[Path, ...]


def build_job_path_plan(
    input_dir: str | Path,
    output_dir: Optional[str | Path],
    *,
    overwrite: bool,
) -> JobPathPlan:
    resolved_input_dir = Path(input_dir).expanduser().resolve()
    if not resolved_input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {resolved_input_dir}")
    if not resolved_input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {resolved_input_dir}")
    if not os.access(resolved_input_dir, os.R_OK | os.X_OK):
        raise PermissionError(f"input_dir is not readable: {resolved_input_dir}")

    video_paths = tuple(scan_videos(resolved_input_dir))
    if not video_paths:
        raise ValueError(f"No supported video files found in directory: {resolved_input_dir}")

    _validate_unique_video_stems(video_paths)

    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else default_output_dir_for_input(resolved_input_dir)
    )
    _ensure_output_root_ready(resolved_output_dir)
    _validate_existing_results(video_paths, resolved_output_dir, overwrite=overwrite)

    return JobPathPlan(
        input_dir=resolved_input_dir,
        output_dir=resolved_output_dir,
        video_paths=video_paths,
    )


def validate_vis_mode(vis_mode: str) -> str:
    normalized = vis_mode.strip().lower()
    if normalized not in SUPPORTED_VIS_MODES:
        supported = ", ".join(sorted(SUPPORTED_VIS_MODES))
        raise ValueError(f"Unsupported vis_mode: {vis_mode}. Supported values: {supported}")
    return normalized


def default_output_dir_for_input(input_dir: Path) -> Path:
    return (input_dir.parent / f"{input_dir.name}_output").resolve()


def scan_videos(input_dir: Path) -> Iterable[Path]:
    for candidate in sorted(input_dir.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            yield candidate.resolve()


def video_result_dir(output_dir: Path, video_path: Path) -> Path:
    return output_dir / video_path.stem


def _ensure_output_root_ready(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output_dir is not a directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(output_dir, os.W_OK | os.X_OK):
        raise PermissionError(f"output_dir is not writable: {output_dir}")


def _validate_unique_video_stems(video_paths: tuple[Path, ...]) -> None:
    seen: dict[str, Path] = {}
    for video_path in video_paths:
        existing = seen.get(video_path.stem)
        if existing is not None:
            raise ValueError(
                "Duplicate video stem detected in input_dir: "
                f"{existing.name} and {video_path.name} both map to {video_path.stem}"
            )
        seen[video_path.stem] = video_path


def _validate_existing_results(
    video_paths: tuple[Path, ...],
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return

    conflicts = [str(video_result_dir(output_dir, video_path)) for video_path in video_paths if video_result_dir(output_dir, video_path).exists()]
    if conflicts:
        preview = ", ".join(conflicts[:3])
        if len(conflicts) > 3:
            preview = f"{preview}, ..."
        raise FileExistsError(
            "Result directories already exist and overwrite=false: "
            f"{preview}"
        )