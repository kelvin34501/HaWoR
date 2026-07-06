#!/usr/bin/env python3
"""Visualize an existing HaWoR world_space_res.pth as a video overlay.

This is intentionally separate from visualize_reconstructed_video.py, which
renders camera-space chunks directly. This script requires an already-built
world_space_res.pth and a matching merged SLAM result; it never runs the
infiller or creates world-space results.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hawor.utils.rotation import angle_axis_to_rotation_matrix
from lib.eval_utils.custom_utils import load_slam_cam
from lib.pipeline.frame_source import FrameSource
from scripts.slam_world_utils import find_slam_npz, stitch_slam_npz
from scripts.visualize_reconstructed_video import (
    PyrenderHandRenderer,
    create_mano_model,
    drain_stderr_pipe,
    draw_hand_skeleton,
    mano_forward,
    project_points_cam,
)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _detect_focal(seq_dir: Path, slam_npz: Path, default: float = 600.0) -> float:
    focal_path = seq_dir / "est_focal.txt"
    if focal_path.exists():
        return float(focal_path.read_text().strip())

    data = np.load(slam_npz, allow_pickle=True)
    if "img_focal" in data:
        return float(np.asarray(data["img_focal"]).reshape(-1)[0])

    print(f"Warning: no focal source found, using default focal={default}", flush=True)
    return default


def _world_hand_matrices(
    pred_rot: np.ndarray,
    pred_hand_pose: np.ndarray,
    hand_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert world-space angle-axis params for one hand to rot matrices."""
    root_aa = torch.from_numpy(pred_rot[hand_idx]).float()
    hand_aa = torch.from_numpy(pred_hand_pose[hand_idx].reshape(pred_hand_pose.shape[1], 15, 3)).float()
    root = angle_axis_to_rotation_matrix(root_aa).detach().cpu().numpy()
    hand = angle_axis_to_rotation_matrix(hand_aa.reshape(-1, 3)).reshape(-1, 15, 3, 3)
    return root.astype(np.float32), hand.detach().cpu().numpy().astype(np.float32)


def load_world_space_track_data(
    seq_dir: Path,
    slam_npz: Path,
    world_space_res: Path,
    track_id: int,
    is_left: bool,
    focal: float,
    cx: float,
    cy: float,
    device: str = "cpu",
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    """Project existing world-space MANO results through merged/stitched SLAM."""
    stitched_slam, _ = stitch_slam_npz(str(seq_dir), str(slam_npz))
    slam_start = find_slam_npz(str(seq_dir), str(slam_npz))[1]
    R_w2c, t_w2c, _, _ = load_slam_cam(stitched_slam)
    R_w2c = R_w2c.float()
    t_w2c = t_w2c.float()

    pred_trans, pred_rot, pred_hand_pose, pred_betas, _ = joblib.load(world_space_res)
    pred_trans = _to_numpy(pred_trans).astype(np.float32)
    pred_rot = _to_numpy(pred_rot).astype(np.float32)
    pred_hand_pose = _to_numpy(pred_hand_pose).astype(np.float32)
    pred_betas = _to_numpy(pred_betas).astype(np.float32)

    if pred_trans.ndim != 3 or pred_rot.ndim != 3 or pred_hand_pose.ndim != 3:
        raise ValueError(f"Unexpected tensor shape in {world_space_res}")
    if track_id >= pred_trans.shape[0]:
        return {}, {}, np.array([])

    root, hand = _world_hand_matrices(pred_rot, pred_hand_pose, track_id)
    mano_model = create_mano_model(is_left=is_left, device=device)
    faces_raw = mano_model.faces
    if hasattr(faces_raw, "detach"):
        faces_raw = faces_raw.detach().cpu().numpy()
    faces = np.asarray(faces_raw, dtype=np.int64)

    joints_world, vertices_world = mano_forward(
        mano_model,
        root,
        hand,
        pred_trans[track_id],
        pred_betas[track_id],
        device=device,
    )

    first_frame = max(0, slam_start)
    last_frame = min(pred_trans.shape[1], slam_start + R_w2c.shape[0])
    joints_data: dict[int, np.ndarray] = {}
    vertices_data: dict[int, np.ndarray] = {}
    if last_frame <= first_frame:
        return joints_data, vertices_data, faces

    for frame_idx in range(first_frame, last_frame):
        cam_idx = frame_idx - slam_start
        R = R_w2c[cam_idx].cpu().numpy()
        t = t_w2c[cam_idx].cpu().numpy()
        joints_cam = joints_world[frame_idx] @ R.T + t
        vertices_cam = vertices_world[frame_idx] @ R.T + t
        joints_2d, depth = project_points_cam(joints_cam, focal, focal, cx, cy)
        joints_2d[depth <= 1e-6] = np.nan
        joints_data[frame_idx] = joints_2d
        vertices_data[frame_idx] = vertices_cam.astype(np.float32)

    return joints_data, vertices_data, faces


def visualize_world_to_video(
    video_path: Path,
    seq_dir: Path,
    world_space_res: Path,
    output_path: Path,
    slam_npz: Optional[Path] = None,
    fps: int = 30,
    focal: Optional[float] = None,
    device: str = "cpu",
    render_mode: str = "both",
    mesh_alpha: float = 0.5,
) -> None:
    video_path = Path(video_path)
    seq_dir = Path(seq_dir)
    world_space_res = Path(world_space_res)
    output_path = Path(output_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"seq_dir not found: {seq_dir}")
    if not world_space_res.is_file():
        raise FileNotFoundError(f"world_space_res not found: {world_space_res}")

    if slam_npz is None:
        slam_path, _, _ = find_slam_npz(str(seq_dir))
        slam_npz = Path(slam_path)
    else:
        slam_npz = Path(slam_npz)
    if not slam_npz.is_file():
        raise FileNotFoundError(f"SLAM npz not found: {slam_npz}")

    if focal is None:
        focal = _detect_focal(seq_dir, slam_npz)

    frames = FrameSource(str(video_path), target_fps=fps, color="bgr")
    n_frames = len(frames)
    if n_frames == 0:
        raise RuntimeError("FrameSource produced no frames")

    h, w = frames.frame_shape
    cx, cy = w / 2, h / 2
    print(
        f"Processing {n_frames} frames at {w}x{h}, focal={focal:.1f}, "
        f"fps={fps}, mode={render_mode}, world={world_space_res}",
        flush=True,
    )

    renderer_left: Optional[PyrenderHandRenderer] = None
    renderer_right: Optional[PyrenderHandRenderer] = None
    if render_mode in ("mesh", "both"):
        renderer_left = PyrenderHandRenderer(w, h, focal, focal, cx, cy)
        renderer_right = PyrenderHandRenderer(w, h, focal, focal, cx, cy)

    print("Loading left hand (track 0) from world_space_res...", flush=True)
    left_joints, left_vertices, left_faces = load_world_space_track_data(
        seq_dir, slam_npz, world_space_res, 0, True, focal, cx, cy, device=device)
    print("Loading right hand (track 1) from world_space_res...", flush=True)
    right_joints, right_vertices, right_faces = load_world_space_track_data(
        seq_dir, slam_npz, world_space_res, 1, False, focal, cx, cy, device=device)
    print(f"Left hand: {len(left_joints)} frames, Right hand: {len(right_joints)} frames", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-stats",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "medium",
        str(output_path),
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr_output: list[str] = []
    stderr_thread = threading.Thread(target=drain_stderr_pipe, args=(proc.stderr, stderr_output), daemon=True)
    stderr_thread.start()

    color_left = (0, 0, 180)
    color_right = (0, 150, 0)

    try:
        for frame_idx in tqdm(range(n_frames), desc="Encoding video"):
            if frame_idx % 100 == 0 and proc.poll() is not None:
                raise RuntimeError(f"FFmpeg process died unexpectedly with return code {proc.returncode}")

            img = np.ascontiguousarray(frames[frame_idx])

            if frame_idx in left_joints:
                if render_mode in ("mesh", "both") and renderer_left is not None:
                    img = renderer_left.render_hand(img, left_vertices[frame_idx], left_faces, color_left, mesh_alpha)
                if render_mode in ("skeleton", "both"):
                    draw_hand_skeleton(img, left_joints[frame_idx], color_left, thickness=4)
            if frame_idx in right_joints:
                if render_mode in ("mesh", "both") and renderer_right is not None:
                    img = renderer_right.render_hand(img, right_vertices[frame_idx], right_faces, color_right, mesh_alpha)
                if render_mode in ("skeleton", "both"):
                    draw_hand_skeleton(img, right_joints[frame_idx], color_right, thickness=4)

            proc.stdin.write(img.tobytes())

        proc.stdin.close()
        proc.wait()
        stderr_thread.join(timeout=10)

        if proc.returncode == 0:
            print(f"Video saved: {output_path}", flush=True)
        else:
            stderr = "".join(stderr_output)
            raise RuntimeError(f"FFmpeg error ({proc.returncode}):\n{stderr}")
    except Exception:
        proc.kill()
        raise
    finally:
        if proc.stdin:
            proc.stdin.close()
        if proc.stderr:
            proc.stderr.close()
        if renderer_left is not None:
            renderer_left.delete()
        if renderer_right is not None:
            renderer_right.delete()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize an existing HaWoR world_space_res.pth as an overlay video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video_path", required=True, help="Input video path")
    parser.add_argument("--seq_dir", required=True, help="Processed seq dir containing SLAM/")
    parser.add_argument("--world_space_res",
                        default=None,
                        help="Existing world_space_res.pth. Defaults to <seq_dir>/world_space_res.pth")
    parser.add_argument("--slam_npz", default=None, help="Explicit merged SLAM npz if seq_dir/SLAM has multiple")
    parser.add_argument("--output", required=True, help="Output MP4 video path")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate for decoding and output")
    parser.add_argument("--focal", type=float, default=None, help="Camera focal length")
    parser.add_argument("--device", default="cpu", help="Device for MANO rendering, e.g. cpu or cuda")
    parser.add_argument("--render_mode",
                        choices=["skeleton", "mesh", "both"],
                        default="both",
                        help="Hand rendering style")
    parser.add_argument("--mesh_alpha", type=float, default=0.5, help="Opacity of hand mesh")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    seq_dir = Path(args.seq_dir)
    world_space_res = Path(args.world_space_res) if args.world_space_res else seq_dir / "world_space_res.pth"

    visualize_world_to_video(
        video_path=Path(args.video_path),
        seq_dir=seq_dir,
        world_space_res=world_space_res,
        slam_npz=Path(args.slam_npz) if args.slam_npz else None,
        output_path=Path(args.output),
        fps=args.fps,
        focal=args.focal,
        device=args.device,
        render_mode=args.render_mode,
        mesh_alpha=args.mesh_alpha,
    )
    print(f"Done: {args.output}", flush=True)


if __name__ == "__main__":
    main()
