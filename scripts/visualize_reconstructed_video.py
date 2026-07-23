#!/usr/bin/env python3
"""Visualize HaWoR hand annotations as overlay on a video.

Given a video and pre-computed cam_space annotations, this script:
1. Extracts frames from the video (cached under the output directory)
2. Loads MANO parameters from cam_space
3. Renders MANO hand meshes and skeletons overlaid on the frames
4. Encodes the result as an MP4 video
"""

from __future__ import annotations

import sys
import os

# ---------------------------------------------------------------------------
# MUST set before any OpenGL import - enables headless offscreen rendering
# ---------------------------------------------------------------------------
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from natsort import natsorted
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.pipeline.frame_source import FrameSource


def patch_pyopengl_osmesa() -> None:
    """Patch PyOpenGL 3.1.0 gaps required by pyrender's OSMesa platform."""
    if os.environ.get("PYOPENGL_PLATFORM") != "osmesa":
        return

    import ctypes

    from OpenGL import platform as gl_platform
    from OpenGL.constant import Constant
    from OpenGL.raw.GL._types import GLint
    from OpenGL.raw.osmesa._types import OSMesaContext
    import OpenGL.osmesa as osmesa
    import OpenGL.raw.osmesa.mesa as raw_osmesa

    def ensure_constant(name: str, value: int) -> None:
        if getattr(osmesa, name, None) is not None:
            return
        const = Constant(name, value)
        setattr(osmesa, name, const)
        setattr(raw_osmesa, name, const)

    ensure_constant("OSMESA_DEPTH_BITS", 0x30)
    ensure_constant("OSMESA_PROFILE", 0x33)
    ensure_constant("OSMESA_CORE_PROFILE", 0x34)
    ensure_constant("OSMESA_CONTEXT_MAJOR_VERSION", 0x36)
    ensure_constant("OSMESA_CONTEXT_MINOR_VERSION", 0x37)

    if getattr(osmesa, "OSMesaCreateContextAttribs", None) is None:

        def OSMesaCreateContextAttribs(attrib_list, sharelist):
            pass

        typed_func = gl_platform.types(
            OSMesaContext,
            ctypes.POINTER(GLint),
            OSMesaContext,
        )(OSMesaCreateContextAttribs)
        create_context_attribs = gl_platform.createFunction(
            typed_func,
            gl_platform.PLATFORM.OSMesa,
            None,
            error_checker=None,
        )
        osmesa.OSMesaCreateContextAttribs = create_context_attribs
        raw_osmesa.OSMesaCreateContextAttribs = create_context_attribs

    original_get_current_context = gl_platform.PLATFORM.GetCurrentContext

    def get_current_context_address():
        ctx = original_get_current_context()
        if isinstance(ctx, int):
            return ctx
        return ctypes.cast(ctx, ctypes.c_void_p).value or 0

    gl_platform.PLATFORM.GetCurrentContext = get_current_context_address
    gl_platform.PLATFORM.CurrentContextIsValid = get_current_context_address
    gl_platform.GetCurrentContext = get_current_context_address
    gl_platform.CurrentContextIsValid = get_current_context_address
    osmesa.OSMesaGetCurrentContext = get_current_context_address
    raw_osmesa.OSMesaGetCurrentContext = get_current_context_address


patch_pyopengl_osmesa()

import pyrender
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from lib.models.mano_wrapper import MANO


def drain_stderr_pipe(pipe, output_list: list[str]) -> None:
    """Drain stderr pipe in background thread to prevent buffer deadlock."""
    try:
        for line in iter(pipe.readline, b""):
            output_list.append(line.decode("utf-8", errors="replace"))
    except Exception:
        pass
    finally:
        pipe.close()


def load_cam_space_chunk(json_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MANO parameters from one cam_space JSON chunk."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    R_root = np.squeeze(np.array(data["init_root_orient"]))
    R_hand = np.squeeze(np.array(data["init_hand_pose"]))
    t_root = np.squeeze(np.array(data["init_trans"]))
    betas = np.squeeze(np.array(data["init_betas"]))

    if R_root.ndim == 2 and R_root.shape == (3, 3):
        R_root = R_root[None, ...]
    if R_hand.ndim == 3 and R_hand.shape == (15, 3, 3):
        R_hand = R_hand[None, ...]
    if t_root.ndim == 1 and t_root.shape == (3,):
        t_root = t_root[None, ...]
    if betas.ndim == 1:
        betas = betas[None, ...]

    T = R_root.shape[0]
    if betas.shape[0] == 1:
        betas = np.repeat(betas, T, axis=0)

    return R_root, R_hand, t_root, betas


def create_mano_model(is_left: bool = False, device: str = "cpu") -> MANO:
    """Create a MANO model for one hand side."""
    if is_left:
        mano_cfg = {
            "data_dir": str(PROJECT_DIR / "_DATA/data_left/"),
            "model_path": str(PROJECT_DIR / "_DATA/data_left/mano_left"),
            "gender": "neutral",
            "num_hand_joints": 15,
            "num_betas": 10,
            "create_body_pose": False,
            "is_rhand": False,
        }
    else:
        mano_cfg = {
            "data_dir": str(PROJECT_DIR / "_DATA/data/"),
            "model_path": str(PROJECT_DIR / "_DATA/data/mano"),
            "gender": "neutral",
            "num_hand_joints": 15,
            "num_betas": 10,
            "create_body_pose": False,
        }

    mano = MANO(**mano_cfg).to(device)

    if is_left and hasattr(mano, "shapedirs"):
        with torch.no_grad():
            mano.shapedirs[:, 0, :] *= -1

    return mano


def mano_forward(
    mano_model: MANO,
    R_root: np.ndarray,
    R_hand: np.ndarray,
    t_root: np.ndarray,
    betas: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run MANO forward pass and return 3D joints and vertices."""
    global_orient = torch.from_numpy(R_root).float().to(device).unsqueeze(1)
    hand_pose = torch.from_numpy(R_hand).float().to(device)
    transl = torch.from_numpy(t_root).float().to(device)
    betas_t = torch.from_numpy(betas).float().to(device)

    with torch.no_grad():
        out = mano_model(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas_t,
            transl=transl,
            pose2rot=False,
        )

    return out.joints.detach().cpu().numpy(), out.vertices.detach().cpu().numpy()


def project_points_cam(
    points: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D camera-space points to 2D image coordinates."""
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    Zc = np.clip(Z, 1e-6, None)
    u = fx * (X / Zc) + cx
    v = fy * (Y / Zc) + cy
    return np.stack([u, v], axis=-1), Z


def draw_hand_skeleton(
    img: np.ndarray,
    keypoints_2d: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 4,
) -> np.ndarray:
    """Draw OpenPose-style 21-point hand skeleton on an image."""
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]

    for i, j in edges:
        pt1 = keypoints_2d[i]
        pt2 = keypoints_2d[j]
        if np.isfinite(pt1).all() and np.isfinite(pt2).all():
            cv2.line(img, tuple(pt1.astype(int)), tuple(pt2.astype(int)), color, thickness, cv2.LINE_AA)

    for pt in keypoints_2d:
        if np.isfinite(pt).all():
            cv2.circle(img, tuple(pt.astype(int)), 6, color, -1, cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
#  pyrender-based mesh renderer
# ---------------------------------------------------------------------------
class PyrenderHandRenderer:
    """Off-screen pyrender renderer for overlaying hand meshes onto frames.

    Creates a persistent OpenGL context once and reuses it across frames,
    updating only the mesh geometry per hand.  Provides proper Phong-style
    shading with ambient + directional lighting.

    Usage per frame::

        img = renderer.render_hand(img, vertices_cam, faces, color_bgr, alpha)
    """

    def __init__(
        self,
        img_w: int,
        img_h: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> None:
        self._img_w = img_w
        self._img_h = img_h

        # Scene with transparent background
        self._scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0],
            ambient_light=[0.25, 0.25, 0.25],
        )

        # Camera — IntrinsicsCamera handles OpenCV-style intrinsics natively.
        # Input vertices must be in OpenGL convention: X→right, Y→up, Z→backward.
        self._camera = pyrender.IntrinsicsCamera(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            znear=0.001,
            zfar=100.0,
        )
        self._scene.add(self._camera, pose=np.eye(4), name="_camera")

        # Key light — directional, shining into the scene (+Z in OpenGL)
        key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        # The light's local -Z is the emission direction.
        # Rotate 180° around Y so local -Z → world +Z.
        key_pose = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self._scene.add(key_light, pose=key_pose, name="_key_light")

        # Fill light — softer, from below-front
        fill_light = pyrender.DirectionalLight(color=[0.7, 0.7, 0.8], intensity=1.0)
        fill_pose = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self._scene.add(fill_light, pose=fill_pose, name="_fill_light")

        self._renderer = pyrender.OffscreenRenderer(img_w, img_h)
        self._mesh_node = None  # pyrender node handle

    def render_hand(
        self,
        bg_img: np.ndarray,
        vertices_cam: np.ndarray,
        faces: np.ndarray,
        color_bgr: tuple[int, int, int],
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Render one hand mesh over *bg_img* and return the composited image.

        Parameters
        ----------
        bg_img : np.ndarray
            Background image (H, W, 3) in BGR8.
        vertices_cam : np.ndarray
            (V, 3) – mesh vertices in OpenCV camera space (+Z forward, Y down).
        faces : np.ndarray
            (F, 3) – triangle indices.
        color_bgr : tuple[int, int, int]
            Hand colour in BGR order, 0-255.
        alpha : float
            Blend opacity, 0.0-1.0.

        Returns
        -------
        np.ndarray
            *bg_img* with the hand mesh overlaid.
        """
        # ---- remove previous mesh -------------------------------------------------
        if self._mesh_node is not None:
            self._scene.remove_node(self._mesh_node)
            self._mesh_node = None

        # ---- convert vertices: OpenCV → OpenGL convention -------------------------
        # OpenCV cam:  X→right  Y→down   Z→forward
        # OpenGL view: X→right  Y→up     Z→backward  (camera looks along -Z)
        verts_gl = vertices_cam.copy()
        verts_gl[:, 1] *= -1.0  # flip Y
        verts_gl[:, 2] *= -1.0  # flip Z

        # ---- colour -----------------------------------------------------------------
        # BGR → RGB, normalised to [0, 1]
        rgb = np.array([color_bgr[2], color_bgr[1], color_bgr[0]], dtype=np.float64) / 255.0

        # ---- build pyrender mesh ---------------------------------------------------
        tri = trimesh.Trimesh(vertices=verts_gl, faces=faces, process=False)

        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(rgb[0], rgb[1], rgb[2], 1.0),
            metallicFactor=0.0,
            roughnessFactor=0.55,
        )
        py_mesh = pyrender.Mesh.from_trimesh(tri, material=material)
        self._mesh_node = self._scene.add(py_mesh, name="_hand_mesh")

        # ---- render to off-screen buffer -------------------------------------------
        colour_rgb, depth = self._renderer.render(self._scene)

        # ---- composite onto background ---------------------------------------------
        mask = depth > 0
        if not mask.any():
            return bg_img

        rendered_bgr = cv2.cvtColor(colour_rgb, cv2.COLOR_RGB2BGR)
        result = bg_img.copy()
        result[mask] = (bg_img[mask] * (1.0 - alpha) + rendered_bgr[mask] * alpha).astype(np.uint8)
        return result

    def delete(self) -> None:
        """Release the off-screen renderer and its OpenGL context."""
        if self._renderer is not None:
            self._renderer.delete()
            self._renderer = None


def load_track_data(
    cam_space_dir: Path,
    track_id: int,
    is_left: bool,
    focal: float,
    cx: float,
    cy: float,
    device: str = "cpu",
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    """Load cam_space chunks, returning 2D joints, 3D vertices, and faces.

    Returns:
        joints_2d: dict[frame_idx, (21, 2)] projected 2D joint positions.
        vertices_3d: dict[frame_idx, (778, 3)] 3D mesh vertices in camera space.
        faces: (1538, 3) int64 triangle face indices.
    """
    track_dir = cam_space_dir / str(track_id)
    empty: dict[int, np.ndarray] = {}
    if not track_dir.exists():
        return empty, empty, np.array([])

    mano_model = create_mano_model(is_left=is_left, device=device)
    _faces = mano_model.faces
    if hasattr(_faces, "detach"):
        _faces = _faces.detach().cpu().numpy()
    faces = np.asarray(_faces, dtype=np.int64)
    joints_data: dict[int, np.ndarray] = {}
    vertices_data: dict[int, np.ndarray] = {}

    for json_file in natsorted(glob(str(track_dir / "*.json"))):
        json_path = Path(json_file)
        try:
            R_root, R_hand, t_root, betas = load_cam_space_chunk(json_path)
            joints_3d, vertices_3d = mano_forward(mano_model, R_root, R_hand, t_root, betas, device=device)

            stem = json_path.stem.split("_50fps")[0]
            s_idx, _ = map(int, stem.split("_"))

            for i in range(len(joints_3d)):
                frame_idx = s_idx + i
                joints_2d, depth = project_points_cam(joints_3d[i], focal, focal, cx, cy)
                joints_2d[depth <= 1e-6] = np.nan
                joints_data[frame_idx] = joints_2d
                vertices_data[frame_idx] = vertices_3d[i]
        except Exception as e:
            print(f"Warning: Failed to process {json_path}: {e}", flush=True)

    return joints_data, vertices_data, faces


def _detect_focal(cam_space_dir: Path, default: float = 600.0) -> float:
    """Detect focal length from cam_space_dir or its parent directory."""
    for base in (cam_space_dir, cam_space_dir.parent):
        focal_path = base / "est_focal.txt"
        if focal_path.exists():
            return float(focal_path.read_text().strip())
    print(f"Warning: est_focal.txt not found near {cam_space_dir}, using default focal={default}", flush=True)
    return default


def visualize_to_video(
    video_path: Path,
    cam_space_dir: Path,
    output_path: Path,
    fps: int = 30,
    focal: Optional[float] = None,
    device: str = "cpu",
    render_mode: str = "both",
    mesh_alpha: float = 0.5,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
) -> None:
    """Render HaWoR 2D hand projections into an MP4 overlay video.

    Frames are decoded on demand from the video via FrameSource -- nothing is
    extracted or cached to disk.

    Args:
        video_path: Path to input video.
        cam_space_dir: Directory containing track_0/ and track_1/ MANO JSONs.
        output_path: Path for output MP4 file.
        fps: Frame rate for decoding and output video (must match how the
            annotations were computed).
        focal: Camera focal length. Auto-detected from cam_space_dir if None.
        device: 'cpu' or 'cuda' for MANO forward pass.
        render_mode: 'skeleton', 'mesh', or 'both'.
        mesh_alpha: Opacity of hand mesh (0.0 - 1.0).
        start_frame: Zero-based index of the first frame to render.
        max_frames: Render only the first N frames when provided.
    """
    video_path = Path(video_path)
    cam_space_dir = Path(cam_space_dir)
    output_path = Path(output_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not cam_space_dir.is_dir():
        raise FileNotFoundError(f"cam_space_dir not found: {cam_space_dir}")

    if focal is None:
        focal = _detect_focal(cam_space_dir)

    frames = FrameSource(str(video_path), target_fps=fps, color="bgr")
    _render_overlay(
        frames,
        cam_space_dir,
        output_path,
        fps,
        focal,
        device,
        render_mode,
        mesh_alpha,
        start_frame,
        max_frames,
    )


def _render_overlay(
    frames: FrameSource,
    cam_space_dir: Path,
    output_path: Path,
    fps: int,
    focal: float,
    device: str,
    render_mode: str = "both",
    mesh_alpha: float = 0.5,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
) -> None:
    """Internal: render hand overlay from on-demand video frames and cam_space data."""
    total_frames = len(frames)
    if total_frames == 0:
        raise RuntimeError("FrameSource produced no frames")
    if start_frame < 0:
        raise ValueError("start_frame must be zero or greater")
    if start_frame >= total_frames:
        raise ValueError(
            f"start_frame {start_frame} is outside the video ({total_frames} frames)"
        )

    end_frame = total_frames
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be greater than zero")
        end_frame = min(total_frames, start_frame + max_frames)
    n_frames = end_frame - start_frame

    h, w = frames.frame_shape
    cx, cy = w / 2, h / 2
    print(
        f"Processing {n_frames} frames [{start_frame}, {end_frame}) at "
        f"{w}x{h}, focal={focal:.1f}, fps={fps}, mode={render_mode}",
        flush=True,
    )

    # ---- pyrender renderers (one per hand side) ---------------------------------
    renderer_left: Optional[PyrenderHandRenderer] = None
    renderer_right: Optional[PyrenderHandRenderer] = None
    if render_mode in ("mesh", "both"):
        renderer_left = PyrenderHandRenderer(w, h, focal, focal, cx, cy)
        renderer_right = PyrenderHandRenderer(w, h, focal, focal, cx, cy)

    print("Loading left hand (track 0)...", flush=True)
    left_joints, left_vertices, left_faces = load_track_data(
        cam_space_dir,
        0,
        is_left=True,
        focal=focal,
        cx=cx,
        cy=cy,
        device=device,
    )
    print("Loading right hand (track 1)...", flush=True)
    right_joints, right_vertices, right_faces = load_track_data(
        cam_space_dir,
        1,
        is_left=False,
        focal=focal,
        cx=cx,
        cy=cy,
        device=device,
    )
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
        for frame_idx in tqdm(range(start_frame, end_frame), desc="Encoding video"):
            if frame_idx % 100 == 0 and proc.poll() is not None:
                raise RuntimeError(f"FFmpeg process died unexpectedly with return code {proc.returncode}")

            img = np.ascontiguousarray(frames[frame_idx])  # BGR, decoded on demand

            if frame_idx in left_joints:
                if render_mode in ("mesh", "both"):
                    if renderer_left is not None:
                        img = renderer_left.render_hand(
                            img,
                            left_vertices[frame_idx],
                            left_faces,
                            color_left,
                            mesh_alpha,
                        )
                if render_mode in ("skeleton", "both"):
                    draw_hand_skeleton(img, left_joints[frame_idx], color_left, thickness=4)
            if frame_idx in right_joints:
                if render_mode in ("mesh", "both"):
                    if renderer_right is not None:
                        img = renderer_right.render_hand(
                            img,
                            right_vertices[frame_idx],
                            right_faces,
                            color_right,
                            mesh_alpha,
                        )
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
        # Release pyrender OpenGL contexts
        if renderer_left is not None:
            renderer_left.delete()
        if renderer_right is not None:
            renderer_right.delete()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize HaWoR hand annotations as overlay on video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video_path", required=True, help="Input video path")
    parser.add_argument("--cam_space_dir", required=True, help="Path to cam_space annotations (track_0/, track_1/)")
    parser.add_argument("--output", required=True, help="Output MP4 video path")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate for decoding and output (match annotation fps)")
    parser.add_argument("--focal", type=float, default=None, help="Camera focal length (auto-detected if omitted)")
    parser.add_argument("--device", default="cpu", help="Device for MANO rendering, e.g. cpu or cuda")
    parser.add_argument("--render_mode",
                        choices=["skeleton", "mesh", "both"],
                        default="both",
                        help="Hand rendering style")
    parser.add_argument("--mesh_alpha", type=float, default=0.5, help="Opacity of hand mesh (0.0 - 1.0)")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Zero-based index of the first frame to render")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to render")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    visualize_to_video(
        video_path=Path(args.video_path),
        cam_space_dir=Path(args.cam_space_dir),
        output_path=Path(args.output),
        fps=args.fps,
        focal=args.focal,
        device=args.device,
        render_mode=args.render_mode,
        mesh_alpha=args.mesh_alpha,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
    )
    print(f"Done: {args.output}", flush=True)


if __name__ == "__main__":
    main()
