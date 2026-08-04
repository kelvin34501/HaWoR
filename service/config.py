from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

GpuIds = Union[str, Sequence[Union[int, str]]]


@dataclass(frozen=True)
class ServiceConfig:
    cache_dir: Path
    gpu_ids: GpuIds
    cleanup_intermediate: bool
    cleanup_failed_cache: bool
    host: str
    port: int
    s3mount_bin: str
    mount_root: Path
    mount_ready_timeout: float
    max_workers: Optional[int] = None
    jobs_per_gpu: int = 1
    copy_extracted_images: bool = True
    run_visualizations: bool = False
    decord_num_threads: int = 1
    decord_recycle_after: int = 4096
    window_context_frames: int = 16
    temporal_block_frames: int = 16
    normalize_input_timestamps: bool = True


def load_service_config(
    *,
    cache_dir: Optional[Union[str, Path]] = None,
    gpu_ids: Optional[GpuIds] = None,
    cleanup_intermediate: Optional[bool] = None,
    cleanup_failed_cache: Optional[bool] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    max_workers: Optional[int] = None,
    jobs_per_gpu: Optional[int] = None,
    copy_extracted_images: Optional[bool] = None,
    run_visualizations: Optional[bool] = None,
    s3mount_bin: Optional[str] = None,
    mount_root: Optional[Union[str, Path]] = None,
    mount_ready_timeout: Optional[float] = None,
    decord_num_threads: Optional[int] = None,
    decord_recycle_after: Optional[int] = None,
    window_context_frames: Optional[int] = None,
    temporal_block_frames: Optional[int] = None,
    normalize_input_timestamps: Optional[bool] = None,
) -> ServiceConfig:
    resolved_cache_dir = Path(cache_dir or os.getenv("HAWOR_CACHE_DIR", ".hawor_cache")).expanduser().resolve()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    _assert_writable_directory(resolved_cache_dir)

    resolved_mount_root = Path(mount_root or os.getenv("HAWOR_MOUNT_ROOT", "/mnt/oss")).expanduser().resolve()
    resolved_mount_root.mkdir(parents=True, exist_ok=True)
    _assert_writable_directory(resolved_mount_root)

    resolved_mount_ready_timeout = mount_ready_timeout
    if resolved_mount_ready_timeout is None:
        resolved_mount_ready_timeout = float(os.getenv("HAWOR_MOUNT_READY_TIMEOUT", "30"))
    if resolved_mount_ready_timeout <= 0:
        raise ValueError("mount_ready_timeout must be positive")

    resolved_port = port
    if resolved_port is None:
        resolved_port = int(os.getenv("HAWOR_PORT", "8000"))
    if resolved_port <= 0:
        raise ValueError("port must be positive")

    resolved_max_workers = max_workers
    if resolved_max_workers is None:
        raw_max_workers = os.getenv("HAWOR_MAX_WORKERS")
        if raw_max_workers:
            resolved_max_workers = int(raw_max_workers)
    if resolved_max_workers is not None and resolved_max_workers <= 0:
        raise ValueError("max_workers must be positive when provided")

    resolved_jobs_per_gpu = jobs_per_gpu
    if resolved_jobs_per_gpu is None:
        raw_jobs_per_gpu = os.getenv("HAWOR_JOBS_PER_GPU")
        resolved_jobs_per_gpu = int(raw_jobs_per_gpu) if raw_jobs_per_gpu else 1
    if resolved_jobs_per_gpu < 1:
        raise ValueError("jobs_per_gpu must be >= 1")

    resolved_decord_num_threads = _positive_int_setting(
        decord_num_threads,
        env_name="HAWOR_DECORD_NUM_THREADS",
        default=1,
        setting_name="decord_num_threads",
    )
    resolved_decord_recycle_after = _positive_int_setting(
        decord_recycle_after,
        env_name="HAWOR_DECORD_RECYCLE_AFTER",
        default=4096,
        setting_name="decord_recycle_after",
    )
    resolved_window_context_frames = _nonnegative_int_setting(
        window_context_frames,
        env_name="HAWOR_WINDOW_CONTEXT_FRAMES",
        default=16,
        setting_name="window_context_frames",
    )
    resolved_temporal_block_frames = _positive_int_setting(
        temporal_block_frames,
        env_name="HAWOR_TEMPORAL_BLOCK_FRAMES",
        default=16,
        setting_name="temporal_block_frames",
    )
    available_owned_frames = 3001 - 2 * resolved_window_context_frames
    if available_owned_frames < 1:
        raise ValueError(
            "window_context_frames is too large for the 3001-frame process cap"
        )
    if resolved_temporal_block_frames > available_owned_frames:
        raise ValueError(
            "temporal_block_frames does not fit with window_context_frames "
            "under the 3001-frame process cap"
        )

    return ServiceConfig(
        cache_dir=resolved_cache_dir,
        gpu_ids=gpu_ids or os.getenv("HAWOR_GPU_IDS", "0"),
        cleanup_intermediate=(cleanup_intermediate if cleanup_intermediate is not None else _parse_bool_env(
            "HAWOR_CLEANUP_INTERMEDIATE", default=True)),
        cleanup_failed_cache=(cleanup_failed_cache if cleanup_failed_cache is not None else _parse_bool_env(
            "HAWOR_CLEANUP_FAILED_CACHE", default=False)),
        host=host or os.getenv("HAWOR_HOST", "0.0.0.0"),
        port=resolved_port,
        s3mount_bin=s3mount_bin or os.getenv("HAWOR_S3MOUNT_BIN", "s3mount"),
        mount_root=resolved_mount_root,
        mount_ready_timeout=resolved_mount_ready_timeout,
        max_workers=resolved_max_workers,
        jobs_per_gpu=resolved_jobs_per_gpu,
        copy_extracted_images=(copy_extracted_images if copy_extracted_images is not None else _parse_bool_env(
            "HAWOR_COPY_EXTRACTED_IMAGES", default=True)),
        run_visualizations=(run_visualizations if run_visualizations is not None else _parse_bool_env(
            "HAWOR_RUN_VISUALIZATIONS", default=False)),
        decord_num_threads=resolved_decord_num_threads,
        decord_recycle_after=resolved_decord_recycle_after,
        window_context_frames=resolved_window_context_frames,
        temporal_block_frames=resolved_temporal_block_frames,
        normalize_input_timestamps=(
            normalize_input_timestamps
            if normalize_input_timestamps is not None
            else _parse_bool_env("HAWOR_NORMALIZE_INPUT_TIMESTAMPS", default=True)
        ),
    )


def _parse_bool_env(name: str, *, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw_value}")


def _positive_int_setting(
    explicit_value: Optional[int],
    *,
    env_name: str,
    default: int,
    setting_name: str,
) -> int:
    raw_value = explicit_value
    if raw_value is None:
        raw_value = os.getenv(env_name, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{setting_name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{setting_name} must be a positive integer")
    return value


def _nonnegative_int_setting(
    explicit_value: Optional[int],
    *,
    env_name: str,
    default: int,
    setting_name: str,
) -> int:
    raw_value = explicit_value
    if raw_value is None:
        raw_value = os.getenv(env_name, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{setting_name} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{setting_name} must be a non-negative integer")
    return value


def _assert_writable_directory(path: Path) -> None:
    if not path.is_dir():
        raise NotADirectoryError(f"cache_dir is not a directory: {path}")

    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".hawor_write_check_", delete=True):
            pass
    except OSError as exc:
        raise PermissionError(f"cache_dir is not writable: {path}") from exc
