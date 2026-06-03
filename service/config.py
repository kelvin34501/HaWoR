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


def load_service_config(
    *,
    cache_dir: Optional[Union[str, Path]] = None,
    gpu_ids: Optional[GpuIds] = None,
    cleanup_intermediate: Optional[bool] = None,
    cleanup_failed_cache: Optional[bool] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    max_workers: Optional[int] = None,
    s3mount_bin: Optional[str] = None,
    mount_root: Optional[Union[str, Path]] = None,
    mount_ready_timeout: Optional[float] = None,
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


def _assert_writable_directory(path: Path) -> None:
    if not path.is_dir():
        raise NotADirectoryError(f"cache_dir is not a directory: {path}")

    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".hawor_write_check_", delete=True):
            pass
    except OSError as exc:
        raise PermissionError(f"cache_dir is not writable: {path}") from exc
