from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

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
DEFAULT_S3MOUNT_PREFIXES = (Path("/mnt/oss"),)
S3_URL_SCHEME = "s3://"


@dataclass(frozen=True)
class JobPathPlan:
    input_dir: Path
    output_dir: Path
    video_paths: tuple[Path, ...]


@dataclass(frozen=True)
class S3Url:
    """A parsed ``s3://bucket/key`` location.

    ``prefix`` is the bucket-relative key path with surrounding slashes stripped
    and may be empty when the URL points at the bucket root.
    """

    bucket: str
    prefix: str

    @classmethod
    def from_url(cls, url: str) -> "S3Url":
        """Parse an ``s3://bucket/path/to/videos`` URL into bucket and prefix.

        Rejects empty values, non-``s3://`` schemes, missing buckets, and any
        prefix that uses ``..`` to traverse out of the bucket.
        """
        raw = (url or "").strip()
        if not raw:
            raise ValueError("input_url must not be empty")
        if not raw.lower().startswith(S3_URL_SCHEME):
            raise ValueError(f"input_url must be an {S3_URL_SCHEME} URL, got: {url}")

        remainder = raw[len(S3_URL_SCHEME):]
        bucket, _, key = remainder.partition("/")
        bucket = bucket.strip()
        if not bucket:
            raise ValueError(f"input_url is missing a bucket name: {url}")

        prefix = key.strip("/")
        if prefix and any(segment == ".." for segment in prefix.split("/")):
            raise ValueError(f"input_url prefix must not contain '..' segments: {url}")
        return cls(bucket=bucket, prefix=prefix)


def build_job_path_plan(
    input_dir: str | Path,
    output_dir: Optional[str | Path],
    *,
    overwrite: bool,
    s3mount_prefixes: Optional[Sequence[str | Path]] = None,
) -> JobPathPlan:
    resolved_input_dir = _resolve_accessible_directory(input_dir, field_name="input_dir")
    input_access = classify_storage_path(
        resolved_input_dir,
        s3mount_prefixes=s3mount_prefixes,
    )

    if _is_directory_empty(resolved_input_dir):
        raise ValueError(f"input_dir is empty: {resolved_input_dir}")

    video_paths = tuple(scan_videos(resolved_input_dir))
    if not video_paths:
        raise ValueError(f"No supported video files found in directory: {resolved_input_dir}")

    _validate_unique_video_stems(video_paths)

    resolved_output_dir = (_resolve_output_directory(output_dir)
                           if output_dir else default_output_dir_for_input(resolved_input_dir))
    output_access = classify_storage_path(
        resolved_output_dir,
        s3mount_prefixes=s3mount_prefixes,
    )
    if input_access != output_access:
        raise ValueError("input_dir and output_dir must use the same storage access model "
                         f"(got input={input_access}, output={output_access})")

    _ensure_output_root_ready(resolved_output_dir)
    _validate_existing_results(video_paths, resolved_output_dir, overwrite=overwrite)

    return JobPathPlan(
        input_dir=resolved_input_dir,
        output_dir=resolved_output_dir,
        video_paths=video_paths,
    )


def resolve_within_mount(mount_dir: str | Path, subpath: str) -> Path:
    """Resolve a bucket-relative subpath to an absolute path inside ``mount_dir``.

    Rejects absolute subpaths and any value that escapes the mount via ``..`` so a
    request can only ever reach data inside its own per-job mount.
    """
    resolved_mount = Path(mount_dir).resolve()
    raw = (subpath or "").strip()
    if not raw:
        raise ValueError("subpath must not be empty")
    if raw.lower().startswith("s3://"):
        raise ValueError("Native s3:// paths are not supported; provide a bucket-relative subpath")

    candidate = Path(raw)
    if candidate.is_absolute():
        raise ValueError(f"subpath must be relative to the bucket, got absolute path: {raw}")

    resolved = (resolved_mount / candidate).resolve()
    if not _is_relative_to(resolved, resolved_mount):
        raise ValueError(f"subpath escapes the bucket mount: {raw}")
    return resolved


def default_output_subdir(input_subdir: str) -> str:
    normalized = (input_subdir or "").strip().strip("/")
    if not normalized:
        raise ValueError("input_subdir must not be empty")
    return f"{normalized}_output"


def validate_vis_mode(vis_mode: str) -> str:
    normalized = vis_mode.strip().lower()
    if normalized not in SUPPORTED_VIS_MODES:
        supported = ", ".join(sorted(SUPPORTED_VIS_MODES))
        raise ValueError(f"Unsupported vis_mode: {vis_mode}. Supported values: {supported}")
    return normalized


def default_output_dir_for_input(input_dir: Path) -> Path:
    return (input_dir.parent / f"{input_dir.name}_output").resolve()


def classify_storage_path(
    path: str | Path,
    *,
    s3mount_prefixes: Optional[Sequence[str | Path]] = None,
) -> str:
    if isinstance(path, str) and path.strip().lower().startswith("s3://"):
        raise ValueError("Native s3:// paths are not supported in this service; use a local or s3mount path")

    resolved_path = Path(path).expanduser().resolve()
    prefixes = _normalize_s3mount_prefixes(s3mount_prefixes)
    if any(_is_relative_to(resolved_path, prefix) for prefix in prefixes):
        return "s3mount"
    return "local"


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
            raise ValueError("Duplicate video stem detected in input_dir: "
                             f"{existing.name} and {video_path.name} both map to {video_path.stem}")
        seen[video_path.stem] = video_path


def _validate_existing_results(
    video_paths: tuple[Path, ...],
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return

    conflicts = [
        str(video_result_dir(output_dir, video_path))
        for video_path in video_paths
        if video_result_dir(output_dir, video_path).exists()
    ]
    if conflicts:
        preview = ", ".join(conflicts[:3])
        if len(conflicts) > 3:
            preview = f"{preview}, ..."
        raise FileExistsError("Result directories already exist and overwrite=false: "
                              f"{preview}")


def _resolve_accessible_directory(raw_path: str | Path, *, field_name: str) -> Path:
    if isinstance(raw_path, str) and raw_path.strip().lower().startswith("s3://"):
        raise ValueError(f"{field_name} does not support native s3:// paths; provide a local or s3mount path")

    resolved_path = Path(raw_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {resolved_path}")
    if not resolved_path.is_dir():
        raise NotADirectoryError(f"{field_name} is not a directory: {resolved_path}")
    if not os.access(resolved_path, os.R_OK | os.X_OK):
        raise PermissionError(f"{field_name} is not accessible: {resolved_path}")
    return resolved_path


def _resolve_output_directory(raw_path: str | Path) -> Path:
    if isinstance(raw_path, str) and raw_path.strip().lower().startswith("s3://"):
        raise ValueError("output_dir does not support native s3:// paths; provide a local or s3mount path")
    return Path(raw_path).expanduser().resolve()


def _normalize_s3mount_prefixes(prefixes: Optional[Sequence[str | Path]],) -> tuple[Path, ...]:
    raw_prefixes = prefixes or DEFAULT_S3MOUNT_PREFIXES
    normalized: list[Path] = []
    for prefix in raw_prefixes:
        normalized.append(Path(prefix).expanduser().resolve())
    return tuple(normalized)


def _is_directory_empty(path: Path) -> bool:
    return next(path.iterdir(), None) is None


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
    except ValueError:
        return False
    return True
