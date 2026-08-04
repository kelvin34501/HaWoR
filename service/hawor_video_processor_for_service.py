#!/usr/bin/env python3
"""Service-specific HaWoR processor with cache-rooted intermediate outputs.

This file is copied from scripts/hawor_video_processor.py and adapted for
folder-level service execution. The original script remains unchanged.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence, Union

from service.storage import PROCESS_DONE_FILENAME

GpuIds = Union[str, Sequence[Union[int, str]]]
PEAK_RSS_ACCEPTANCE_BYTES = 20 * 1024 ** 3
# Interpolated disparity exports are transient inputs/diagnostics. Publishing
# them can roughly double the per-video result size, so the server API keeps
# them in local scratch only.
COPY_BACK_DISPARITY_ARTIFACTS = False
_RSS_SAMPLE_INTERVAL_SECONDS = 0.05
_PAGE_SIZE_BYTES = os.sysconf("SC_PAGE_SIZE")


class _PeakRssTracker:

    def __init__(self, callback: Optional[Callable[[int], None]] = None) -> None:
        self.peak_rss_bytes = 0
        self._callback = callback

    def observe(self, rss_bytes: int) -> None:
        rss_bytes = max(0, int(rss_bytes))
        if rss_bytes <= self.peak_rss_bytes:
            return
        self.peak_rss_bytes = rss_bytes
        if self._callback is not None:
            try:
                self._callback(rss_bytes)
            except Exception:
                # Telemetry delivery must never terminate an annotation process.
                pass


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
    world_space_res_path: Optional[Path]
    world_space_res_50fps_path: Optional[Path]
    cam_space_vis_path: Optional[Path]
    world_space_vis_path: Optional[Path]
    peak_rss_bytes: int = 0


@dataclass(frozen=True)
class VideoTimingInfo:
    codec_name: str
    fps: Fraction
    is_cfr: bool
    has_b_frames: bool
    min_pts_seconds: float
    min_dts_seconds: float

    @property
    def has_negative_presentation_timestamps(self) -> bool:
        return self.min_pts_seconds < -1e-6


@dataclass(frozen=True)
class HaWoRProcessorConfig:
    project_dir: Path = Path(__file__).resolve().parents[1]
    segment_seconds: int = 100
    min_last_segment_seconds: int = 50
    max_chunk_frames: int = 3001
    # HaWoR uses non-overlapping 16-frame temporal blocks. Outer windows retain
    # real context, align their owned boundaries, and crossfade shared-context
    # predictions so an API seam cannot hard-cut between restarted track phases.
    window_context_frames: int = 16
    temporal_block_frames: int = 16
    overlap_policy: str = "keep_last"
    vis_mode: str = "off"
    target_fps: float = 30
    interp_target_fps: float = 50
    run_post_steps: bool = True
    force_interpolate: bool = False
    cleanup_intermediate: bool = True
    # Stitch the merged multi-window SLAM and run the infiller over the full
    # timeline to produce a world-space result (world_space_res.pth).
    run_world_space: bool = True
    infiller_weight: str = "./weights/hawor/checkpoints/infiller.pt"
    run_visualizations: bool = True
    # Frames are decoded on demand from the video; nothing is extracted to disk, so
    # this flag is accepted for API compatibility but has no effect.
    copy_extracted_images: bool = True
    decord_num_threads: int = 1
    decord_recycle_after: int = 4096
    normalize_input_timestamps: bool = True


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
        peak_rss_callback: Optional[Callable[[int], None]] = None,
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
        peak_rss = _PeakRssTracker(peak_rss_callback)

        if not video.is_file():
            raise FileNotFoundError(f"Video file not found: {video}")

        work_dir = scratch_root / "_segmented_work"
        staged_output_dir = scratch_root / "_final_output"
        # The live log is append-written and used as subprocess stdout, which is
        # unsupported on s3mount output targets. Keep it on local scratch and
        # publish a single sequential copy to out_dir at the end.
        live_log_path = scratch_root / "process.log"
        log_path = out_dir / "process.log"
        if overwrite_output and clear_on_overwrite:
            self._clear_known_outputs(out_dir, scratch_root, log_path)
        else:
            self._clear_scratch_outputs(scratch_root)

        with self.gpu_pool.acquire() as gpu_id:
            env = self._build_env(gpu_id)
            try:
                processing_video = self._prepare_video_for_processing(
                    video,
                    scratch_root,
                    live_log_path,
                    env,
                    peak_rss,
                )
                self._run_segmented_pipeline(
                    processing_video,
                    work_dir,
                    live_log_path,
                    gpu_id,
                    env,
                    peak_rss,
                )
                self._copy_directory_contents(work_dir / "merged", staged_output_dir, overwrite_output=True)
                if self.config.cleanup_intermediate and work_dir.exists():
                    shutil.rmtree(work_dir)

                # Stitch merged SLAM + infill the full sequence -> world_space_res.pth.
                # Runs on the staged dir before interpolation so it flows to out_dir
                # with the staged copy and the 50fps step can interpolate it too.
                if self.config.run_world_space:
                    self._build_world_space_res(
                        staged_output_dir,
                        processing_video,
                        live_log_path,
                        env,
                        peak_rss,
                    )

                # Frames are never extracted to disk; the 50fps interpolation derives
                # its frame count directly from the video.
                extracted_images_50fps_dir: Optional[Path] = None
                if self.config.run_post_steps or self.config.force_interpolate:
                    self._run_interpolation(
                        staged_output_dir,
                        processing_video,
                        live_log_path,
                        env,
                        peak_rss,
                    )

                if not COPY_BACK_DISPARITY_ARTIFACTS:
                    self._delete_disparity_artifacts(staged_output_dir)

                cam_space_vis_path: Optional[Path] = None
                world_space_vis_path: Optional[Path] = None
                if self.config.run_visualizations:
                    staged_cam_space_vis_path = self._run_cam_space_visualization(
                        staged_output_dir,
                        processing_video,
                        live_log_path,
                        env,
                        peak_rss,
                    )
                    cam_space_vis_path = out_dir / staged_cam_space_vis_path.name
                    if self.config.run_world_space:
                        staged_world_space_vis_path = self._run_world_space_visualization(
                            staged_output_dir,
                            processing_video,
                            live_log_path,
                            env,
                            peak_rss,
                        )
                        world_space_vis_path = out_dir / staged_world_space_vis_path.name

                if not self.config.cleanup_intermediate:
                    self._copy_tree_data_only(work_dir, staged_output_dir / "_segmented_work")

                out_dir.mkdir(parents=True, exist_ok=True)
                self._copy_directory_contents(staged_output_dir,
                                              out_dir,
                                              overwrite_output=overwrite_output)
                self._write_done_sentinel(out_dir, scratch_root)
            finally:
                self._write_peak_rss_summary(live_log_path, peak_rss.peak_rss_bytes)
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
                world_space_res_path=(out_dir / "world_space_res.pth"
                                      if self.config.run_world_space else None),
                world_space_res_50fps_path=(out_dir / "world_space_res_50fps.pth"
                                            if self.config.run_world_space else None),
                cam_space_vis_path=cam_space_vis_path,
                world_space_vis_path=world_space_vis_path,
                peak_rss_bytes=peak_rss.peak_rss_bytes,
            )

    def _validate_config(self) -> None:
        if self.config.segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        if self.config.min_last_segment_seconds < 0:
            raise ValueError("min_last_segment_seconds must be non-negative")
        if self.config.max_chunk_frames < 2:
            raise ValueError("max_chunk_frames must be >= 2")
        if self.config.window_context_frames < 0:
            raise ValueError("window_context_frames must be non-negative")
        if self.config.temporal_block_frames < 1:
            raise ValueError("temporal_block_frames must be positive")
        if (
            self.config.max_chunk_frames
            <= 2 * self.config.window_context_frames
        ):
            raise ValueError(
                "max_chunk_frames must exceed twice window_context_frames"
            )
        if self.config.decord_num_threads < 1:
            raise ValueError("decord_num_threads must be >= 1")
        if self.config.decord_recycle_after < 1:
            raise ValueError("decord_recycle_after must be >= 1")
        if self.config.overlap_policy not in {"keep_last", "keep_first"}:
            raise ValueError("overlap_policy must be 'keep_last' or 'keep_first'")
        if shutil.which(sys.executable) is None:
            raise FileNotFoundError(f"Python executable not found: {sys.executable}")
        if shutil.which("ffmpeg") is None:
            raise FileNotFoundError("ffmpeg not found in PATH")
        if shutil.which("ffprobe") is None:
            raise FileNotFoundError("ffprobe not found in PATH")

    def _prepare_video_for_processing(
        self,
        video: Path,
        scratch_root: Path,
        log_path: Path,
        env: dict[str, str],
        peak_rss: _PeakRssTracker,
    ) -> Path:
        """Return a decoder-safe input path without modifying the source file.

        Negative presentation timestamps can make a fresh Decord accurate seek
        return pre-roll pixels from an older logical frame.  CFR inputs without
        B-frames are rewritten onto an exact frame-index timeline. Other codecs
        retain their packet spacing/order and are shifted just enough to remove
        negative timestamps. Both paths are stream copies.
        """
        if not self.config.normalize_input_timestamps:
            self._append_log(log_path, "[input] timestamp normalization disabled")
            return video

        timing = self._probe_video_timing(video)
        if not timing.has_negative_presentation_timestamps:
            self._append_log(
                log_path,
                "[input] timestamps already decoder-safe "
                f"(min_pts={timing.min_pts_seconds:.6f}s, "
                f"fps={float(timing.fps):.6f})",
            )
            return video

        suffix = video.suffix.lower() or ".mp4"
        normalized = scratch_root / f"_normalized_input{suffix}"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-map_metadata",
            "-1",
            "-c:v",
            "copy",
            "-an",
            "-sn",
            "-dn",
        ]
        if timing.is_cfr and not timing.has_b_frames:
            numerator = timing.fps.numerator
            denominator = timing.fps.denominator
            timestamp_filter = (
                f"setts=pts=N*{denominator}/({numerator}*TB):"
                f"dts=N*{denominator}/({numerator}*TB):"
                f"duration={denominator}/({numerator}*TB)"
            )
            cmd.extend(["-bsf:v", timestamp_filter])
            normalization_mode = "exact-cfr"
        else:
            cmd.extend(["-avoid_negative_ts", "make_zero"])
            normalization_mode = "offset-preserving"
        if suffix in {".m4v", ".mov", ".mp4"}:
            cmd.extend(["-movflags", "+faststart"])
        cmd.append(str(normalized))

        self._append_log(
            log_path,
            "[input] normalizing negative timestamps with a stream copy "
            f"(mode={normalization_mode}, "
            f"min_pts={timing.min_pts_seconds:.6f}s)",
        )
        self._run_command(cmd, log_path, env, peak_rss)
        if not normalized.is_file() or normalized.stat().st_size == 0:
            raise RuntimeError(
                f"Timestamp normalization did not produce a video: {normalized}"
            )

        verified = self._probe_video_timing(normalized)
        if verified.has_negative_presentation_timestamps:
            raise RuntimeError(
                "Timestamp normalization failed: output still has negative "
                f"presentation timestamps ({verified.min_pts_seconds:.6f}s)"
            )
        if abs(float(verified.fps) - float(timing.fps)) > 1e-6:
            raise RuntimeError(
                "Timestamp normalization changed the nominal frame rate: "
                f"{float(timing.fps):.6f} -> {float(verified.fps):.6f}"
            )
        self._append_log(
            log_path,
            "[input] normalized decoder timeline verified "
            f"(min_pts={verified.min_pts_seconds:.6f}s, "
            f"path={normalized})",
        )
        return normalized

    def _probe_video_timing(self, video: Path) -> VideoTimingInfo:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=codec_name,r_frame_rate,avg_frame_rate,has_b_frames:"
                "packet=pts_time,dts_time,duration_time"
            ),
            "-read_intervals",
            "%+#16",
            "-of",
            "json",
            str(video),
        ]
        completed = subprocess.run(
            cmd,
            cwd=self.config.project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "unknown ffprobe error"
            raise RuntimeError(f"Unable to inspect video timestamps: {detail}")
        try:
            payload = json.loads(completed.stdout)
            stream = payload["streams"][0]
            packets = payload["packets"]
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unable to parse video timing metadata for {video}"
            ) from exc
        if not packets:
            raise RuntimeError(f"Video contains no readable packets: {video}")

        avg_rate = _parse_frame_rate(stream.get("avg_frame_rate"))
        real_rate = _parse_frame_rate(stream.get("r_frame_rate"))
        fps = avg_rate or real_rate
        if fps is None or fps <= 0:
            raise RuntimeError(f"Video has no valid nominal frame rate: {video}")

        pts = _packet_times(packets, "pts_time")
        dts = _packet_times(packets, "dts_time")
        if not pts:
            raise RuntimeError(f"Video packets have no presentation timestamps: {video}")
        if not dts:
            dts = pts

        nominal_duration = 1.0 / float(fps)
        tolerance = max(1e-6, nominal_duration * 0.02)
        durations = _packet_times(packets, "duration_time")
        pts_steps = [b - a for a, b in zip(pts, pts[1:]) if b > a]
        cadence_is_constant = all(
            abs(value - nominal_duration) <= tolerance
            for value in durations + pts_steps
        )
        rates_match = (
            avg_rate is not None
            and real_rate is not None
            and abs(float(avg_rate) - float(real_rate)) <= 1e-6
        )
        return VideoTimingInfo(
            codec_name=str(stream.get("codec_name") or "unknown"),
            fps=fps,
            is_cfr=bool(rates_match and cadence_is_constant),
            has_b_frames=int(stream.get("has_b_frames") or 0) > 0,
            min_pts_seconds=min(pts),
            min_dts_seconds=min(dts),
        )

    def _append_log(self, log_path: Path, message: str) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_lock:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"{message}\n")

    def _build_env(self, gpu_id: int) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Forty concurrent jobs do not need eight CPU inference pools apiece.
        # One thread also keeps PyTorch/OpenCV allocator overhead below the 20 GiB
        # process-tree gate. Preserve an explicit operator override.
        env.setdefault("HAWOR_NUM_THREADS", "1")
        env["HAWOR_DECORD_NUM_THREADS"] = str(self.config.decord_num_threads)
        env["HAWOR_DECORD_RECYCLE_AFTER"] = str(self.config.decord_recycle_after)
        return env

    def _run_segmented_pipeline(
        self,
        video: Path,
        work_dir: Path,
        log_path: Path,
        gpu_id: int,
        env: dict[str, str],
        peak_rss: _PeakRssTracker,
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
            "--max_chunk_frames",
            str(self.config.max_chunk_frames),
            "--window_context_frames",
            str(self.config.window_context_frames),
            "--temporal_block_frames",
            str(self.config.temporal_block_frames),
            "--overlap_policy",
            self.config.overlap_policy,
            "--vis_mode",
            self.config.vis_mode,
            "--target_fps",
            str(self.config.target_fps),
            "--gpu_id",
            str(gpu_id),
            "--work_dir",
            str(work_dir),
        ]
        cmd.append("--skip_copy_back")

        self._run_command(cmd, log_path, env, peak_rss)

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

    def _delete_disparity_artifacts(self, output_dir: Path) -> None:
        """Delete transient interpolated disparity NPZ/MKV files before publish."""
        slam_dir = output_dir / "SLAM"
        if not slam_dir.is_dir():
            return

        artifacts = set(slam_dir.glob("*_disps_*.npz"))
        artifacts.update(slam_dir.glob("*_disps_*_uint16.mkv"))
        for artifact in sorted(artifacts):
            artifact.unlink()

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

    def _clear_scratch_outputs(self, scratch_root: Path) -> None:
        """Clear local per-video scratch, including staged final output."""
        if scratch_root.exists():
            shutil.rmtree(scratch_root)
        scratch_root.mkdir(parents=True, exist_ok=True)

    def _clear_known_outputs(self, output_dir: Path, scratch_root: Path, log_path: Path) -> None:
        self._clear_scratch_outputs(scratch_root)

        # Best-effort removal of previous out_dir artifacts. Silently skips paths
        # that cannot be deleted (e.g. s3mount targets without --allow-delete).
        out_paths = [
            log_path,
            output_dir / PROCESS_DONE_FILENAME,
            output_dir / "cam_space",
            output_dir / "cam_space_50fps",
            output_dir / "SLAM",
            output_dir / "extracted_images",
            output_dir / "extracted_images_50fps",
            output_dir / "world_space_res.pth",
            output_dir / "world_space_res_50fps.pth",
            output_dir / "cam_space_visualization.mp4",
            output_dir / "cam_space_visualization_50fps.mp4",
            output_dir / "world_space_visualization.mp4",
            output_dir / "world_space_visualization_50fps.mp4",
            output_dir / "_segmented_work",
        ]
        out_paths.extend(sorted(output_dir.glob("cam_space_visualization_*fps.mp4")))
        out_paths.extend(sorted(output_dir.glob("world_space_visualization_*fps.mp4")))
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

    def _build_world_space_res(
        self,
        seq_dir: Path,
        video: Path,
        log_path: Path,
        env: dict[str, str],
        peak_rss: _PeakRssTracker,
    ) -> None:
        script = self.config.project_dir / "scripts" / "build_world_space_res.py"
        cmd = [
            sys.executable,
            str(script),
            "--seq_dir",
            str(seq_dir),
            "--video_path",
            str(video),
            "--infiller_weight",
            str(self.config.infiller_weight),
            "--target_fps",
            str(self.config.target_fps),
        ]
        self._run_command(cmd, log_path, env, peak_rss)

    def _run_cam_space_visualization(
        self,
        seq_dir: Path,
        video: Path,
        log_path: Path,
        env: dict[str, str],
        peak_rss: _PeakRssTracker,
    ) -> Path:
        script = self.config.project_dir / "scripts" / "visualize_reconstructed_video.py"
        cam_space_dir = seq_dir / "cam_space_50fps"
        render_fps = int(round(self.config.interp_target_fps))
        if not cam_space_dir.is_dir():
            cam_space_dir = seq_dir / "cam_space"
            render_fps = int(round(self.config.target_fps))
        if not cam_space_dir.is_dir():
            raise FileNotFoundError(f"Camera-space directory not found: {cam_space_dir}")
        output_path = seq_dir / f"cam_space_visualization_{render_fps}fps.mp4"
        cmd = [
            sys.executable,
            str(script),
            "--video_path",
            str(video),
            "--cam_space_dir",
            str(cam_space_dir),
            "--output",
            str(output_path),
            "--fps",
            str(render_fps),
        ]
        self._run_command(cmd, log_path, env, peak_rss)
        return output_path

    def _run_world_space_visualization(
        self,
        seq_dir: Path,
        video: Path,
        log_path: Path,
        env: dict[str, str],
        peak_rss: _PeakRssTracker,
    ) -> Path:
        script = self.config.project_dir / "scripts" / "visualize_world_reconstructed_video.py"
        world_space_res_50fps = seq_dir / "world_space_res_50fps.pth"
        slam_npz: Optional[Path] = None
        if world_space_res_50fps.is_file():
            try:
                slam_npz = self._find_visualization_slam_npz(seq_dir, prefer_50fps=True)
            except FileNotFoundError:
                slam_npz = None

        if slam_npz is not None:
            world_space_res = world_space_res_50fps
            render_fps = int(round(self.config.interp_target_fps))
        else:
            world_space_res = seq_dir / "world_space_res.pth"
            render_fps = int(round(self.config.target_fps))
            if not world_space_res.is_file():
                raise FileNotFoundError(f"World-space result not found: {world_space_res}")
            slam_npz = self._find_visualization_slam_npz(seq_dir, prefer_50fps=False)
        output_path = seq_dir / f"world_space_visualization_{render_fps}fps.mp4"
        cmd = [
            sys.executable,
            str(script),
            "--video_path",
            str(video),
            "--seq_dir",
            str(seq_dir),
            "--world_space_res",
            str(world_space_res),
            "--slam_npz",
            str(slam_npz),
            "--output",
            str(output_path),
            "--fps",
            str(render_fps),
        ]
        self._run_command(cmd, log_path, env, peak_rss)
        return output_path

    def _find_visualization_slam_npz(self, seq_dir: Path, *, prefer_50fps: bool) -> Path:
        slam_dir = seq_dir / "SLAM"
        suffix = "_50fps" if prefer_50fps else ""
        pattern = re.compile(rf"^hawor_slam_w_scale_\d+_\d+{suffix}\.npz$")
        candidates = sorted(
            path for path in slam_dir.glob("hawor_slam_w_scale_*.npz")
            if pattern.match(path.name) and "_disps_" not in path.name
        )
        if not candidates:
            raise FileNotFoundError(f"No matching SLAM npz found under {slam_dir}")
        if len(candidates) > 1:
            listing = "\n  ".join(str(path) for path in candidates)
            raise RuntimeError(f"Multiple SLAM npz files found for visualization:\n  {listing}")
        return candidates[0]

    def _run_interpolation(
        self,
        output_dir: Path,
        video: Path,
        log_path: Path,
        env: dict[str, str],
        peak_rss: _PeakRssTracker,
    ) -> None:
        script = self.config.project_dir / "scripts" / "interpolation.py"
        cmd = [
            sys.executable,
            str(script),
            "--folder_path",
            str(output_dir),
            "--video_path",
            str(video),
            "--source_fps",
            str(self.config.target_fps),
            "--target_fps",
            str(self.config.interp_target_fps),
        ]
        self._run_command(cmd, log_path, env, peak_rss)

    def _run_command(
        self,
        cmd: Sequence[str],
        log_path: Path,
        env: dict[str, str],
        peak_rss: _PeakRssTracker,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_lock:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n$ {' '.join(cmd)}\n")
                log.flush()

        command_peak_rss_bytes = 0
        with log_path.open("a", encoding="utf-8") as log:
            proc = subprocess.Popen(
                list(cmd),
                cwd=self.config.project_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while True:
                rss_bytes = _read_process_tree_rss_bytes(proc.pid)
                command_peak_rss_bytes = max(command_peak_rss_bytes, rss_bytes)
                peak_rss.observe(rss_bytes)
                try:
                    returncode = proc.wait(timeout=_RSS_SAMPLE_INTERVAL_SECONDS)
                except subprocess.TimeoutExpired:
                    continue
                # Capture one final sample in case the root is still represented
                # in /proc while wait() reaps it.
                rss_bytes = _read_process_tree_rss_bytes(proc.pid)
                command_peak_rss_bytes = max(command_peak_rss_bytes, rss_bytes)
                peak_rss.observe(rss_bytes)
                break

        with self._log_lock:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    "[memory] command_peak_rss_bytes="
                    f"{command_peak_rss_bytes} video_peak_rss_bytes="
                    f"{peak_rss.peak_rss_bytes}\n"
                )

        if returncode != 0:
            raise RuntimeError(f"Command failed with exit code {returncode}. See log: {log_path}")

    def _write_peak_rss_summary(self, log_path: Path, peak_rss_bytes: int) -> None:
        within_limit = peak_rss_bytes < PEAK_RSS_ACCEPTANCE_BYTES
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_lock:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    "[memory] peak_rss_bytes="
                    f"{peak_rss_bytes} peak_rss_gib={peak_rss_bytes / 1024 ** 3:.3f} "
                    f"acceptance_threshold_bytes={PEAK_RSS_ACCEPTANCE_BYTES} "
                    f"acceptance={'PASS' if within_limit else 'FAIL'}\n"
                )


def _parse_frame_rate(value: object) -> Optional[Fraction]:
    if value in (None, "", "0/0"):
        return None
    try:
        rate = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def _packet_times(packets: Sequence[object], key: str) -> list[float]:
    values: list[float] = []
    for packet in packets:
        if not isinstance(packet, dict) or packet.get(key) in (None, "N/A"):
            continue
        try:
            values.append(float(packet[key]))
        except (TypeError, ValueError):
            continue
    return values


def _read_process_tree_rss_bytes(root_pid: int) -> int:
    """Return conservative summed RSS for a Linux process and its descendants.

    The service runs on Linux GPU hosts, where procfs provides both resident-page
    counts and per-thread child lists. Processes can appear or exit while the tree
    is sampled; those races are expected and the next 50 ms sample catches the
    surviving tree.
    """
    total_rss_bytes = 0
    pending = [int(root_pid)]
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)

        proc_dir = Path("/proc") / str(pid)
        try:
            statm_fields = (proc_dir / "statm").read_text().split()
            if len(statm_fields) >= 2:
                total_rss_bytes += int(statm_fields[1]) * _PAGE_SIZE_BYTES
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, OSError):
            continue

        # A multithreaded process may fork from any thread, so aggregate every
        # task's children file rather than inspecting only the thread-group leader.
        try:
            child_files = tuple((proc_dir / "task").glob("*/children"))
        except (FileNotFoundError, PermissionError, OSError):
            child_files = ()
        for child_file in child_files:
            try:
                pending.extend(int(value) for value in child_file.read_text().split())
            except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, OSError):
                continue
    return total_rss_bytes


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
