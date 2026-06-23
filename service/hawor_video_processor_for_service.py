#!/usr/bin/env python3
"""Service-specific HaWoR processor with cache-rooted intermediate outputs.

This file is copied from scripts/hawor_video_processor.py and adapted for
folder-level service execution. The original script remains unchanged.
"""

from __future__ import annotations

import contextlib
import os
import queue
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

from service.storage import PROCESS_DONE_FILENAME

GpuIds = Union[str, Sequence[Union[int, str]]]


@dataclass(frozen=True)
class HaWoRProcessResult:
    video_path: Path
    output_dir: Path
    work_dir: Path
    log_path: Path
    gpu_id: int
    cam_space_dir: Path
    slam_dir: Path
    extracted_images_dir: Path
    extracted_images_50fps_dir: Optional[Path]


@dataclass(frozen=True)
class HaWoRProcessorConfig:
    project_dir: Path = Path(__file__).resolve().parents[1]
    segment_seconds: int = 100
    min_last_segment_seconds: int = 60
    split_mode: str = "reencode"
    overlap_policy: str = "keep_last"
    vis_mode: str = "off"
    run_post_steps: bool = True
    force_interpolate: bool = False
    cleanup_intermediate: bool = True
    overwrite_chunks: bool = False
    copy_extracted_images: bool = True


class GpuPool:
    """Simple blocking GPU pool for worker threads."""

    def __init__(self, gpu_ids: GpuIds = (0,), jobs_per_gpu: int = 1) -> None:
        parsed = _parse_gpu_ids(gpu_ids)
        if not parsed:
            raise ValueError("gpu_ids must contain at least one GPU id")
        if jobs_per_gpu < 1:
            raise ValueError("jobs_per_gpu must be >= 1")

        self.gpu_ids = tuple(parsed)
        self._jobs_per_gpu = jobs_per_gpu
        self._available: "queue.Queue[int]" = queue.Queue()
        for gpu_id in self.gpu_ids:
            for _ in range(jobs_per_gpu):
                self._available.put(gpu_id)

    @contextlib.contextmanager
    def acquire(self) -> Iterator[int]:
        gpu_id = self._available.get(block=True)
        try:
            yield gpu_id
        finally:
            self._available.put(gpu_id)

    @property
    def size(self) -> int:
        return len(self.gpu_ids) * self._jobs_per_gpu


class HaWoRVideoProcessorForService:
    """Run HaWoR annotation for one video with cache-rooted work directories."""

    def __init__(
        self,
        gpu_ids: GpuIds = (0,),
        config: Optional[HaWoRProcessorConfig] = None,
        gpu_pool: Optional[GpuPool] = None,
    ) -> None:
        self.config = config or HaWoRProcessorConfig()
        self.gpu_pool = gpu_pool or GpuPool(gpu_ids)
        self._validate_config()
        self._log_lock = threading.Lock()

    def process_video(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        *,
        scratch_dir: Union[str, Path],
        overwrite_output: bool = False,
        clear_on_overwrite: bool = True,
    ) -> HaWoRProcessResult:
        """Process a single video.

        Args:
            clear_on_overwrite: When *overwrite_output* is True, pre-clear known
                output artifacts before the run.  Set to ``False`` for s3mount
                output directories where delete operations are unsupported.
        """
        video = Path(video_path).expanduser().resolve()
        out_dir = Path(output_dir).expanduser().resolve()
        scratch_root = Path(scratch_dir).expanduser().resolve()

        if not video.is_file():
            raise FileNotFoundError(f"Video file not found: {video}")

        scratch_root.mkdir(parents=True, exist_ok=True)

        work_dir = scratch_root / "_segmented_work"
        staged_output_dir = scratch_root / "_final_output"
        # The live log is append-written and used as subprocess stdout, which is
        # unsupported on s3mount output targets. Keep it on local scratch and
        # publish a single sequential copy to out_dir at the end.
        live_log_path = scratch_root / "process.log"
        log_path = out_dir / "process.log"
        if overwrite_output and clear_on_overwrite:
            self._clear_known_outputs(out_dir, scratch_root, log_path)

        with self.gpu_pool.acquire() as gpu_id:
            env = self._build_env(gpu_id)
            try:
                self._run_segmented_pipeline(video, work_dir, live_log_path, gpu_id, env)
                self._copy_directory_contents(work_dir / "merged", staged_output_dir, overwrite_output=True)

                extracted_images_50fps_dir: Optional[Path] = None
                if self.config.run_post_steps:
                    extracted_images_50fps_dir = staged_output_dir / "extracted_images_50fps"
                    self._run_extract_50fps(video, extracted_images_50fps_dir, live_log_path, env)
                    self._run_interpolation(staged_output_dir, live_log_path, env)
                elif self.config.force_interpolate:
                    self._run_interpolation(staged_output_dir, live_log_path, env)

                exclude: frozenset[str] = frozenset()
                if not self.config.copy_extracted_images:
                    exclude = frozenset({"extracted_images", "extracted_images_50fps"})
                out_dir.mkdir(parents=True, exist_ok=True)
                self._copy_directory_contents(staged_output_dir,
                                              out_dir,
                                              overwrite_output=overwrite_output,
                                              exclude=exclude)
                self._write_done_sentinel(out_dir, scratch_root)
            finally:
                self._publish_log(live_log_path, log_path)

            if self.config.cleanup_intermediate and scratch_root.exists():
                shutil.rmtree(scratch_root)

            return HaWoRProcessResult(
                video_path=video,
                output_dir=out_dir,
                work_dir=work_dir,
                log_path=log_path,
                gpu_id=gpu_id,
                cam_space_dir=out_dir / "cam_space",
                slam_dir=out_dir / "SLAM",
                extracted_images_dir=out_dir / "extracted_images",
                extracted_images_50fps_dir=(out_dir / "extracted_images_50fps"
                                            if extracted_images_50fps_dir is not None else None),
            )

    def _validate_config(self) -> None:
        if self.config.segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        if self.config.min_last_segment_seconds < 0:
            raise ValueError("min_last_segment_seconds must be non-negative")
        if self.config.split_mode not in {"copy", "reencode"}:
            raise ValueError("split_mode must be 'copy' or 'reencode'")
        if self.config.overlap_policy not in {"keep_last", "keep_first"}:
            raise ValueError("overlap_policy must be 'keep_last' or 'keep_first'")
        if shutil.which(sys.executable) is None:
            raise FileNotFoundError(f"Python executable not found: {sys.executable}")
        if shutil.which("ffmpeg") is None:
            raise FileNotFoundError("ffmpeg not found in PATH")

    def _build_env(self, gpu_id: int) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return env

    def _run_segmented_pipeline(
        self,
        video: Path,
        work_dir: Path,
        log_path: Path,
        gpu_id: int,
        env: dict[str, str],
    ) -> None:
        script = self.config.project_dir / "scripts" / "segmented_demo_pipeline.py"
        cmd = [
            sys.executable,
            str(script),
            "--video_path",
            str(video),
            "--segment_seconds",
            str(self.config.segment_seconds),
            "--min_last_segment_seconds",
            str(self.config.min_last_segment_seconds),
            "--overlap_policy",
            self.config.overlap_policy,
            "--vis_mode",
            self.config.vis_mode,
            "--gpu_id",
            str(gpu_id),
            "--work_dir",
            str(work_dir),
        ]
        if self.config.split_mode == "reencode":
            cmd.append("--reencode")
        if self.config.overwrite_chunks:
            cmd.append("--overwrite_chunks")
        cmd.append("--skip_copy_back")

        self._run_command(cmd, log_path, env)

    def _copy_directory_contents(
            self,
            source_dir: Path,
            destination_dir: Path,
            *,
            overwrite_output: bool,
            exclude: frozenset[str] = frozenset(),
    ) -> None:
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {source_dir}")

        destination_dir.mkdir(parents=True, exist_ok=True)
        for source_path in sorted(source_dir.iterdir()):
            if source_path.name in exclude:
                continue
            destination_path = destination_dir / source_path.name
            if destination_path.exists() and not overwrite_output:
                raise FileExistsError(f"Output artifact already exists: {destination_path}. "
                                      "Use overwrite_output=True or a fresh output_dir.")

            # No pre-delete: shutil.copyfile overwrites on local filesystems and on
            # s3mount targets with --allow-overwrite. _copy_tree_data_only uses
            # exist_ok=True and per-file copyfile so it is safe for both.
            if source_path.is_dir():
                self._copy_tree_data_only(source_path, destination_path)
            else:
                shutil.copyfile(source_path, destination_path)

    def _copy_tree_data_only(self, source_dir: Path, destination_dir: Path) -> None:
        """Recursively copy file data only.

        Avoids ``shutil.copytree``/``copy2`` because they call ``copystat`` (chmod,
        utimes, xattr), which is unsupported on s3mount targets and raises
        ``PermissionError [Errno 1]``. Only directory creation and sequential file
        writes are used here, which s3mount supports.
        """
        destination_dir.mkdir(parents=True, exist_ok=True)
        for entry in sorted(source_dir.iterdir()):
            target = destination_dir / entry.name
            if entry.is_dir():
                self._copy_tree_data_only(entry, target)
            else:
                shutil.copyfile(entry, target)

    def _publish_log(self, live_log_path: Path, final_log_path: Path) -> None:
        """Copy the local scratch log to the (possibly s3mount) output directory.

        Best-effort and metadata-free: a single sequential write, no append and no
        ``copystat``. Failures here must not mask the original processing outcome.
        """
        if not live_log_path.is_file():
            return
        try:
            final_log_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(live_log_path, final_log_path)
        except OSError:
            pass

    def _clear_known_outputs(self, output_dir: Path, scratch_root: Path, log_path: Path) -> None:
        # Clear local scratch unconditionally; this also removes any live log inside it.
        if scratch_root.exists():
            shutil.rmtree(scratch_root)
        scratch_root.mkdir(parents=True, exist_ok=True)

        # Best-effort removal of previous out_dir artifacts. Silently skips paths
        # that cannot be deleted (e.g. s3mount targets without --allow-delete).
        out_paths = [
            log_path,
            output_dir / "cam_space",
            output_dir / "cam_space_50fps",
            output_dir / "SLAM",
            output_dir / "extracted_images",
            output_dir / "extracted_images_50fps",
            output_dir / "world_space_res.pth",
            output_dir / "world_space_res_50fps.pth",
        ]
        for path in out_paths:
            if not path.exists():
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except OSError:
                pass

    def _write_done_sentinel(self, out_dir: Path, scratch_root: Path) -> None:
        """Write a process.done sentinel to *out_dir* (best-effort, s3mount-safe).

        Mirrors _publish_log: create a local file on scratch first, then copy
        sequentially to the (possibly s3mount) output directory.  Failures are
        silently swallowed so the sentinel never masks the actual processing result.

        The file is written with non-empty content (``"x"``) because s3mount's
        object-storage backend may not persist zero-byte objects.
        """
        local_sentinel = scratch_root / PROCESS_DONE_FILENAME
        remote_sentinel = out_dir / PROCESS_DONE_FILENAME
        try:
            local_sentinel.write_text("x")
            shutil.copyfile(local_sentinel, remote_sentinel)
        except OSError:
            pass

    def _run_extract_50fps(
        self,
        video: Path,
        output_folder: Path,
        log_path: Path,
        env: dict[str, str],
    ) -> None:
        script = self.config.project_dir / "scripts" / "extract_image_50fps.py"
        cmd = [
            sys.executable,
            str(script),
            "--video_path",
            str(video),
            "--output_folder",
            str(output_folder),
        ]
        self._run_command(cmd, log_path, env)

    def _run_interpolation(
        self,
        output_dir: Path,
        log_path: Path,
        env: dict[str, str],
    ) -> None:
        script = self.config.project_dir / "scripts" / "interpolation.py"
        cmd = [
            sys.executable,
            str(script),
            "--folder_path",
            str(output_dir),
        ]
        self._run_command(cmd, log_path, env)

    def _run_command(
        self,
        cmd: Sequence[str],
        log_path: Path,
        env: dict[str, str],
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_lock:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n$ {' '.join(cmd)}\n")
                log.flush()

        with log_path.open("a", encoding="utf-8") as log:
            proc = subprocess.run(
                list(cmd),
                cwd=self.config.project_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )

        if proc.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {proc.returncode}. See log: {log_path}")


def _parse_gpu_ids(gpu_ids: GpuIds) -> list[int]:
    if isinstance(gpu_ids, str):
        tokens = gpu_ids.replace(",", " ").split()
    else:
        tokens = [str(item) for item in gpu_ids]

    parsed: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        if not token.isdigit():
            raise ValueError(f"Invalid GPU id: {token}")
        gpu_id = int(token)
        if gpu_id in seen:
            raise ValueError(f"Duplicate GPU id: {gpu_id}")
        seen.add(gpu_id)
        parsed.append(gpu_id)
    return parsed
