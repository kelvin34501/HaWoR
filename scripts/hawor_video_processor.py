#!/usr/bin/env python3
"""Python API for single-video HaWoR annotation jobs.

This module wraps the existing segmented pipeline in a server-friendly class:
callers submit one video path and one output directory, while the class owns a
thread-safe GPU pool.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union


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
    world_space_res_path: Optional[Path]
    world_space_res_50fps_path: Optional[Path]
    cam_space_vis_path: Optional[Path]
    world_space_vis_path: Optional[Path]


@dataclass(frozen=True)
class HaWoRProcessorConfig:
    project_dir: Path = Path(__file__).resolve().parents[1]
    python_bin: str = sys.executable
    segment_seconds: int = 100
    min_last_segment_seconds: int = 60
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


class GpuPool:
    """Simple blocking GPU pool for worker threads."""

    def __init__(self, gpu_ids: GpuIds = (0,)) -> None:
        parsed = _parse_gpu_ids(gpu_ids)
        if not parsed:
            raise ValueError("gpu_ids must contain at least one GPU id")

        self.gpu_ids = tuple(parsed)
        self._available: "queue.Queue[int]" = queue.Queue()
        for gpu_id in self.gpu_ids:
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
        return len(self.gpu_ids)


class HaWoRVideoProcessor:
    """Run HaWoR annotation for one video at a time per acquired GPU."""

    def __init__(
        self,
        gpu_ids: GpuIds = (0,),
        config: Optional[HaWoRProcessorConfig] = None,
    ) -> None:
        self.config = config or HaWoRProcessorConfig()
        self.gpu_pool = GpuPool(gpu_ids)
        self._validate_config()
        self._log_lock = threading.Lock()

    def process_video(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        *,
        overwrite_output: bool = False,
    ) -> HaWoRProcessResult:
        """Process one video and write annotation artifacts into output_dir.

        The returned output directory contains the merged annotation artifacts:
        cam_space/, SLAM/, visualizations, and optionally world/50fps outputs.
        This method blocks until a GPU is available.
        """

        video = Path(video_path).expanduser().resolve()
        out_dir = Path(output_dir).expanduser().resolve()

        if not video.is_file():
            raise FileNotFoundError(f"Video file not found: {video}")

        out_dir.mkdir(parents=True, exist_ok=True)

        work_dir = out_dir / "_segmented_work"
        log_path = out_dir / "process.log"
        if overwrite_output:
            self._clear_known_outputs(out_dir, work_dir, log_path)

        with self.gpu_pool.acquire() as gpu_id:
            env = self._build_env(gpu_id)
            self._run_segmented_pipeline(video, work_dir, log_path, gpu_id, env)
            self._copy_merged_outputs(work_dir, out_dir, overwrite_output)
            if self.config.cleanup_intermediate and work_dir.exists():
                shutil.rmtree(work_dir)

            # Stitch merged SLAM + infill the full sequence -> world_space_res.pth.
            # Runs before interpolation so the 50fps step can interpolate it too.
            if self.config.run_world_space:
                self._build_world_space_res(out_dir, video, log_path, env)

            # Frames are never extracted to disk; the 50fps interpolation derives
            # its frame count directly from the video.
            extracted_images_50fps_dir: Optional[Path] = None
            if self.config.run_post_steps or self.config.force_interpolate:
                self._run_interpolation(out_dir, video, log_path, env)

            cam_space_vis_path: Optional[Path] = None
            world_space_vis_path: Optional[Path] = None
            if self.config.run_visualizations:
                cam_space_vis_path = self._run_cam_space_visualization(out_dir, video, log_path, env)
                if self.config.run_world_space:
                    world_space_vis_path = self._run_world_space_visualization(out_dir, video, log_path, env)

            return HaWoRProcessResult(
                video_path=video,
                output_dir=out_dir,
                work_dir=work_dir,
                log_path=log_path,
                gpu_id=gpu_id,
                cam_space_dir=out_dir / "cam_space",
                slam_dir=out_dir / "SLAM",
                extracted_images_dir=out_dir / "extracted_images",
                extracted_images_50fps_dir=extracted_images_50fps_dir,
                world_space_res_path=(out_dir / "world_space_res.pth"
                                      if self.config.run_world_space else None),
                world_space_res_50fps_path=(out_dir / "world_space_res_50fps.pth"
                                            if self.config.run_world_space else None),
                cam_space_vis_path=cam_space_vis_path,
                world_space_vis_path=world_space_vis_path,
            )

    def _validate_config(self) -> None:
        if self.config.segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        if self.config.min_last_segment_seconds < 0:
            raise ValueError("min_last_segment_seconds must be non-negative")
        if self.config.overlap_policy not in {"keep_last", "keep_first"}:
            raise ValueError("overlap_policy must be 'keep_last' or 'keep_first'")
        if shutil.which(self.config.python_bin) is None:
            raise FileNotFoundError(f"Python executable not found: {self.config.python_bin}")
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
            self.config.python_bin,
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
            "--target_fps",
            str(self.config.target_fps),
            "--gpu_id",
            str(gpu_id),
            "--work_dir",
            str(work_dir),
            "--skip_copy_back",
        ]

        self._run_command(cmd, log_path, env)

    def _copy_merged_outputs(
        self,
        work_dir: Path,
        output_dir: Path,
        overwrite_output: bool,
    ) -> None:
        merged_dir = work_dir / "merged"
        if not merged_dir.is_dir():
            raise FileNotFoundError(f"Merged output directory not found: {merged_dir}")

        for src in sorted(merged_dir.iterdir()):
            dst = output_dir / src.name
            if dst.exists():
                if overwrite_output:
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                else:
                    raise FileExistsError(
                        f"Output artifact already exists: {dst}. "
                        "Use overwrite_output=True or a fresh output_dir."
                    )

            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    def _clear_known_outputs(self, output_dir: Path, work_dir: Path, log_path: Path) -> None:
        paths = [
            work_dir,
            log_path,
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
        ]
        paths.extend(sorted(output_dir.glob("cam_space_visualization_*fps.mp4")))
        paths.extend(sorted(output_dir.glob("world_space_visualization_*fps.mp4")))

        for path in paths:
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    def _build_world_space_res(
        self,
        seq_dir: Path,
        video: Path,
        log_path: Path,
        env: dict[str, str],
    ) -> None:
        script = self.config.project_dir / "scripts" / "build_world_space_res.py"
        cmd = [
            self.config.python_bin,
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
        self._run_command(cmd, log_path, env)

    def _run_cam_space_visualization(
        self,
        seq_dir: Path,
        video: Path,
        log_path: Path,
        env: dict[str, str],
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
            self.config.python_bin,
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
        self._run_command(cmd, log_path, env)
        return output_path

    def _run_world_space_visualization(
        self,
        seq_dir: Path,
        video: Path,
        log_path: Path,
        env: dict[str, str],
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
            self.config.python_bin,
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
        self._run_command(cmd, log_path, env)
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
    ) -> None:
        script = self.config.project_dir / "scripts" / "interpolation.py"
        cmd = [
            self.config.python_bin,
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
            raise RuntimeError(
                f"Command failed with exit code {proc.returncode}. See log: {log_path}"
            )


def process_video(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    gpu_ids: GpuIds = (0,),
    config: Optional[HaWoRProcessorConfig] = None,
    overwrite_output: bool = False,
) -> HaWoRProcessResult:
    """Convenience function for a one-off annotation job."""

    return HaWoRVideoProcessor(gpu_ids=gpu_ids, config=config).process_video(
        video_path,
        output_dir,
        overwrite_output=overwrite_output,
    )


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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one HaWoR video annotation job")
    parser.add_argument("--video_path", required=True, help="Input video path")
    parser.add_argument("--output_dir", required=True, help="Directory for annotation outputs")
    parser.add_argument("--gpu_ids", default="0", help='GPU pool, e.g. "0" or "0,1"')
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--segment_seconds", type=int, default=100)
    parser.add_argument("--min_last_segment_seconds", type=int, default=60)
    parser.add_argument("--target_fps", type=float, default=30)
    parser.add_argument("--overlap_policy", choices=["keep_last", "keep_first"], default="keep_last")
    parser.add_argument("--vis_mode", default="off")
    parser.add_argument("--no_world_space", action="store_true",
                        help="skip stitching + full-sequence infiller world-space reconstruction")
    parser.add_argument("--infiller_weight", default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--no_visualizations", action="store_true",
                        help="skip cam-space and world-space overlay video rendering")
    parser.add_argument("--no_post_steps", action="store_true")
    parser.add_argument("--force_interpolate", action="store_true")
    parser.add_argument("--keep_intermediate", action="store_true")
    parser.add_argument("--overwrite_output", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = HaWoRProcessorConfig(
        python_bin=args.python_bin,
        segment_seconds=args.segment_seconds,
        min_last_segment_seconds=args.min_last_segment_seconds,
        target_fps=args.target_fps,
        overlap_policy=args.overlap_policy,
        vis_mode=args.vis_mode,
        run_world_space=not args.no_world_space,
        infiller_weight=args.infiller_weight,
        run_visualizations=not args.no_visualizations,
        run_post_steps=not args.no_post_steps,
        force_interpolate=args.force_interpolate,
        cleanup_intermediate=not args.keep_intermediate,
    )
    result = process_video(
        args.video_path,
        args.output_dir,
        gpu_ids=args.gpu_ids,
        config=config,
        overwrite_output=args.overwrite_output,
    )
    print(f"Done: {result.output_dir}")
    print(f"GPU: {result.gpu_id}")
    print(f"Log: {result.log_path}")


if __name__ == "__main__":
    main()
