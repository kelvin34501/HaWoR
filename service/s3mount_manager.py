"""On-demand s3mount management for the HaWoR annotation service.

Each annotation job mounts exactly one object-storage bucket at a per-job
directory ``<mount_root>/<job_id>/`` using the ``s3mount`` binary, runs against
bucket-relative paths, and unmounts when the job reaches a terminal state.

Security notes:
- Access/secret keys are written to a private ``0600`` credentials file that is
  referenced by the s3mount child process through ``AWS_SHARED_CREDENTIALS_FILE``.
- Secrets are never stored on the returned handle, logged, or echoed back.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

#: Per-job mount point layout: ``<mount_root>/<job_id>/<JOB_INPUT_MOUNT_SUBDIR>``.
JOB_INPUT_MOUNT_SUBDIR = "video_in"


class MountError(RuntimeError):
    """Raised when a bucket cannot be mounted or becomes ready in time."""


class BucketBusyError(RuntimeError):
    """Raised when the requested bucket is already mounted by an active job."""


_PROXY_ENV_VARS = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)


@dataclass(frozen=True)
class MountSpec:
    """Connection details for a single bucket mount.

    ``access_key`` and ``secret_key`` are sensitive and must never be logged.
    """

    bucket: str
    endpoint: str
    access_key: str
    secret_key: str
    prefix: Optional[str] = None
    region: Optional[str] = None
    force_path_style: bool = False
    use_listobject_v2: bool = False
    read_only: bool = False

    def bucket_key(self) -> str:
        """Stable, secret-free identity used to detect duplicate bucket use."""
        normalized_prefix = (self.prefix or "").strip("/")
        return f"{self.endpoint.rstrip('/')}|{self.bucket}|{normalized_prefix}"


@dataclass
class MountHandle:
    job_id: str
    mount_dir: Path
    bucket_key: str
    _credentials_file: Path
    _process: Optional[subprocess.Popen] = field(default=None, repr=False)


class S3MountManager:
    """Mounts and unmounts object-storage buckets on demand via ``s3mount``."""

    def __init__(
        self,
        *,
        s3mount_bin: str,
        mount_root: Path,
        ready_timeout: float,
        cache_root: Optional[Path] = None,
        cache_ttl_seconds: Optional[int] = None,
        poll_interval: float = 0.5,
    ) -> None:
        self._s3mount_bin = s3mount_bin
        self._mount_root = Path(mount_root)
        self._ready_timeout = ready_timeout
        self._cache_root = Path(cache_root) if cache_root else None
        self._cache_ttl_seconds = cache_ttl_seconds
        self._poll_interval = max(0.05, poll_interval)
        self._lock = threading.Lock()
        self._active: dict[str, MountHandle] = {}

    def mount(self, job_id: str, spec: MountSpec) -> MountHandle:
        bucket_key = spec.bucket_key()
        mount_dir = (self._mount_root / job_id / JOB_INPUT_MOUNT_SUBDIR).resolve()

        with self._lock:
            if bucket_key in self._active:
                raise BucketBusyError(f"Bucket is already in use by an active job: {bucket_key}")
            handle = MountHandle(
                job_id=job_id,
                mount_dir=mount_dir,
                bucket_key=bucket_key,
                _credentials_file=Path(),
            )
            self._active[bucket_key] = handle

        try:
            mount_dir.mkdir(parents=True, exist_ok=True)
            credentials_file = self._write_credentials_file(spec)
            handle._credentials_file = credentials_file
            handle._process = self._spawn(spec, mount_dir, credentials_file)
            self._wait_until_ready(handle)
            return handle
        except BaseException:
            self._teardown(handle)
            with self._lock:
                self._active.pop(bucket_key, None)
            raise

    def unmount(self, handle: MountHandle) -> None:
        try:
            self._teardown(handle)
        finally:
            with self._lock:
                self._active.pop(handle.bucket_key, None)

    def _spawn(
        self,
        spec: MountSpec,
        mount_dir: Path,
        credentials_file: Path,
    ) -> subprocess.Popen:
        argv = self._build_argv(spec, mount_dir)
        env = self._build_env(credentials_file)
        try:
            return subprocess.Popen(
                argv,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise MountError(f"s3mount binary not found: {self._s3mount_bin}") from exc

    def _build_argv(self, spec: MountSpec, mount_dir: Path) -> list[str]:
        argv = [self._s3mount_bin, spec.bucket, str(mount_dir), "--endpoint-url", spec.endpoint]
        if spec.prefix:
            prefix = spec.prefix.strip("/")
            if prefix:
                argv += ["--prefix", f"{prefix}/"]
        if spec.read_only:
            argv += ["--read-only"]
        else:
            argv += ["--allow-delete", "--allow-overwrite"]
        if spec.force_path_style:
            argv += ["--force-path-style"]
        if spec.use_listobject_v2:
            argv += ["--use-listobject-v2"]
        if spec.region:
            argv += ["--region", spec.region]
        if self._cache_root is not None:
            cache_dir = (self._cache_root / mount_dir.name)
            cache_dir.mkdir(parents=True, exist_ok=True)
            argv += ["--cache", str(cache_dir)]
            if self._cache_ttl_seconds is not None:
                argv += ["--metadata-ttl", str(self._cache_ttl_seconds)]
        return argv

    def _build_env(self, credentials_file: Path) -> dict[str, str]:
        env = dict(os.environ)
        # Proxies break access to in-cluster object-storage endpoints.
        for name in _PROXY_ENV_VARS:
            env.pop(name, None)
        # Force the child to read only the per-job credentials file and ignore
        # any ambient AWS credentials in the environment.
        env.pop("AWS_ACCESS_KEY_ID", None)
        env.pop("AWS_SECRET_ACCESS_KEY", None)
        env.pop("AWS_SESSION_TOKEN", None)
        env["AWS_SHARED_CREDENTIALS_FILE"] = str(credentials_file)
        env["AWS_PROFILE"] = "default"
        return env

    def _write_credentials_file(self, spec: MountSpec) -> Path:
        fd, raw_path = tempfile.mkstemp(prefix="hawor_s3mount_", suffix=".credentials")
        path = Path(raw_path)
        try:
            content = ("[default]\n"
                       f"aws_access_key_id = {spec.access_key}\n"
                       f"aws_secret_access_key = {spec.secret_key}\n")
            with os.fdopen(fd, "w") as handle:
                handle.write(content)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        os.chmod(path, 0o600)
        return path

    def _wait_until_ready(self, handle: MountHandle) -> None:
        deadline = time.monotonic() + self._ready_timeout
        process = handle._process
        assert process is not None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise MountError(self._mount_failure_message(handle))
            if os.path.ismount(handle.mount_dir):
                return
            time.sleep(self._poll_interval)
        raise MountError(f"Timed out after {self._ready_timeout:g}s waiting for mount: "
                         f"{handle.mount_dir}")

    def _mount_failure_message(self, handle: MountHandle) -> str:
        process = handle._process
        detail = ""
        if process is not None and process.stderr is not None:
            try:
                detail = process.stderr.read().decode("utf-8", "replace").strip()
            except (OSError, ValueError):
                detail = ""
        message = f"s3mount process exited before mounting {handle.mount_dir}"
        if detail:
            message = f"{message}: {detail}"
        return message

    def _teardown(self, handle: MountHandle) -> None:
        if os.path.ismount(handle.mount_dir):
            self._run_unmount_command(handle.mount_dir)

        process = handle._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        handle._process = None

        if handle._credentials_file != Path():
            handle._credentials_file.unlink(missing_ok=True)

        self._remove_empty_dir(handle.mount_dir)

    def _run_unmount_command(self, mount_dir: Path) -> None:
        commands = (
            ["fusermount", "-u", str(mount_dir)],
            ["umount", "-l", "-f", str(mount_dir)],
        )
        for command in commands:
            try:
                result = subprocess.run(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except FileNotFoundError:
                continue
            if result.returncode == 0 or not os.path.ismount(mount_dir):
                return

    def _remove_empty_dir(self, mount_dir: Path) -> None:
        if os.path.ismount(mount_dir):
            return
        try:
            mount_dir.rmdir()
        except OSError:
            return
        # Remove the now-empty per-job parent directory (<mount_root>/<job_id>),
        # but never the shared mount root itself.
        parent = mount_dir.parent
        if parent != self._mount_root.resolve():
            try:
                parent.rmdir()
            except OSError:
                pass
