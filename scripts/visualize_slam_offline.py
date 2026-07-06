"""Visualize offline HaWoR results: SLAM camera trajectory + 3D hand meshes in world space.

Runs purely from a processed seq folder (no detection/tracking/SLAM rerun):

    <seq_dir>/cam_space/<tid>/<start>_<end>.json   raw camera-space MANO chunks
    <seq_dir>/SLAM/hawor_slam_w_scale_<s>_<e>.npz  scaled camera trajectory
    <seq_dir>/world_space_res.pth                  infilled world-space result

Renders an aitviewer scene (y-up = first-frame camera up) with both hand
meshes, a wireframe camera frustum, and a world-axis triad. By default the
view is follow-cam: the SLAM camera is pinned at the scene center each frame
and the hands move relative to it (--no_follow for the absolute world frame).

- This script is read-only. If world_space_res.pth is missing, run
  scripts/build_world_space_res.py first.
- Merged segmented trajectories are stitched automatically when the merged
  SLAM file records interior `window_starts`. If the merge already aligned
  windows through overlap frames, or the result is single-window, the stitch
  step logs that branch and uses the original trajectory unchanged. Stitched
  trajectories are kept in memory and are not written to disk.

Examples:
    # Interactive world view
    python scripts/visualize_slam_offline.py --seq_dir example/video_0 --video_path example/video_0.mp4

    # Interactive world view with source-video point-cloud colors
    python scripts/visualize_slam_offline.py --seq_dir example/tmp_4/3 --video_path <video>
"""
import os
import sys
import math
import argparse
import faulthandler
from glob import glob

sys.path.insert(0, os.path.dirname(__file__) + '/..')

import cv2
import numpy as np
import torch

from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import quaternion_to_matrix
from lib.vis.run_vis2 import run_vis2_on_video
from scripts.slam_world_utils import (
    find_slam_npz,
    stitch_slam_data,
    load_or_build_world_res,
)


class _FrameDims:
    """Stand-in for a FrameSource when only frame dimensions are known.

    World-space viz never reads pixels; run_vis2 only queries `frame_shape`.
    """

    def __init__(self, height, width):
        self.frame_shape = (height, width)


def mano_faces():
    """Right/left face arrays incl. the wrist-closing faces, as in demo.py."""
    faces = get_mano_faces()
    faces_new = np.array([[92, 38, 234],
                          [234, 38, 239],
                          [38, 122, 239],
                          [239, 122, 279],
                          [122, 118, 279],
                          [279, 118, 215],
                          [118, 117, 215],
                          [215, 117, 214],
                          [117, 119, 214],
                          [214, 119, 121],
                          [119, 120, 121],
                          [121, 120, 78],
                          [120, 108, 78],
                          [78, 108, 79]])
    faces_right = np.concatenate([faces, faces_new], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]
    return faces_left, faces_right


def build_world_meshes(pred_trans, pred_rot, pred_hand_pose, pred_betas, vis_start, vis_end):
    """MANO forward for both hands over [vis_start, vis_end); mirrors demo.py."""
    faces_left, faces_right = mano_faces()
    hand2idx = {"left": 0, "right": 1}

    hand_idx = hand2idx['right']
    pred_glob_r = run_mano(pred_trans[hand_idx:hand_idx + 1, vis_start:vis_end],
                           pred_rot[hand_idx:hand_idx + 1, vis_start:vis_end],
                           pred_hand_pose[hand_idx:hand_idx + 1, vis_start:vis_end],
                           betas=pred_betas[hand_idx:hand_idx + 1, vis_start:vis_end])
    right_dict = {
        'vertices': pred_glob_r['vertices'][0].unsqueeze(0),
        'faces': faces_right,
    }

    hand_idx = hand2idx['left']
    pred_glob_l = run_mano_left(pred_trans[hand_idx:hand_idx + 1, vis_start:vis_end],
                                pred_rot[hand_idx:hand_idx + 1, vis_start:vis_end],
                                pred_hand_pose[hand_idx:hand_idx + 1, vis_start:vis_end],
                                betas=pred_betas[hand_idx:hand_idx + 1, vis_start:vis_end])
    left_dict = {
        'vertices': pred_glob_l['vertices'][0].unsqueeze(0),
        'faces': faces_left,
    }
    return left_dict, right_dict


def _depth_colormap(z, z_max):
    """RGB in [0,1] for metric depths z via 'turbo' (matplotlib) or a numpy ramp."""
    t = np.clip(np.asarray(z, dtype=np.float32) / max(z_max, 1e-6), 0.0, 1.0)
    try:
        import matplotlib.cm as _cm
        return _cm.get_cmap('turbo')(t)[:, :3].astype(np.float32)
    except Exception:
        # Simple blue->green->red ramp (near=blue, far=red).
        r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
        return np.stack([r, g, b], axis=-1).astype(np.float32)


def build_point_clouds(slam_path, R_c2w_all, t_c2w_all, R_x, vis_start, vis_end,
                       start_idx, max_points, z_max=10.0, video_path=None,
                       target_fps=30.0):
    """Per-frame world-space point clouds from the SLAM keyframe depth maps.

    `disps` exist per *keyframe*; each vis frame shows its nearest keyframe's
    cloud, unprojected with that keyframe's (possibly stitched) pose — points
    are static in the world while the camera moves. Metric depth = scale/disp.

    Points are colored with the real image RGB sampled from ``video_path`` when
    given (photometric), else by a depth colormap. Returns
    (points, colors) with shapes (F, max_points, 3) and (F, max_points, 4)
    float32, or (None, None) if no disps are available.
    """
    data = np.load(slam_path)
    if 'disps' not in data:
        print(f"No disps in {slam_path}; skipping point cloud")
        return None, None
    disps = data['disps']      # (K, h, w), SLAM (downscaled) res
    tstamp = np.asarray(data['tstamp']).astype(np.int64)
    scale = float(data['scale'])
    fx = float(data['img_focal'])
    cx, cy = np.asarray(data['img_center'], dtype=np.float64)
    _, h, w = disps.shape
    sx, sy = w / (2.0 * cx), h / (2.0 * cy)    # full-res -> disp-map intrinsics
    fxd, fyd = fx * sx, fx * sy
    cxd, cyd = cx * sx, cy * sy
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    uu, vv = uu.reshape(-1), vv.reshape(-1)

    R_all = np.asarray(R_c2w_all, dtype=np.float64)
    t_all = np.asarray(t_c2w_all, dtype=np.float64)
    Rx = np.asarray(R_x, dtype=np.float64)
    rng = np.random.default_rng(0)
    cache = {}

    # Photometric color: decode the source video frame per keyframe via cv2 (no
    # decord — a live decord reader in the viewer process segfaults). Map the
    # keyframe's processed index to a native frame exactly like FrameSource.
    cap = native_fps = n_native = None
    if video_path:
        cap = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or float(target_fps)
        n_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if not cap.isOpened() or n_native <= 0:
            print(f"Point cloud: could not open {video_path}; using depth colormap")
            cap.release()
            cap = None
    photometric = cap is not None
    ratio = (native_fps / float(target_fps)) if photometric else None

    def sample_rgb(k, sel):
        """Real image RGB in [0,1] for the selected disp pixels, or None."""
        native = int(np.clip(math.floor(tstamp[k] * ratio + 1e-6), 0, n_native - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, native)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        frame = cv2.resize(frame, (w, h))[:, :, ::-1]   # BGR->RGB, disp-map res
        return frame.reshape(-1, 3)[sel].astype(np.float32) / 255.0

    def keyframe_cloud(k):
        if k in cache:
            return cache[k]
        disp = disps[k].reshape(-1)
        valid = np.flatnonzero(disp > max(1e-4, scale / z_max))  # z = scale/disp <= z_max
        if valid.size == 0:
            cloud = np.zeros((max_points, 3), dtype=np.float32)
            color = np.zeros((max_points, 4), dtype=np.float32)
        else:
            sel = rng.choice(valid, size=min(max_points, valid.size), replace=False)
            if sel.size < max_points:
                sel = np.concatenate([sel, rng.choice(valid, size=max_points - sel.size)])
            z = scale / disp[sel]
            pc = np.stack([(uu[sel] - cxd) / fxd * z, (vv[sel] - cyd) / fyd * z, z], axis=-1)
            row = int(np.clip(tstamp[k] - start_idx, 0, len(R_all) - 1))
            pw = pc @ R_all[row].T + t_all[row]
            cloud = (pw @ Rx.T).astype(np.float32)
            rgb = sample_rgb(k, sel) if photometric else None
            if rgb is None:                       # colormap (fallback or no video)
                rgb = _depth_colormap(z, z_max)
            color = np.concatenate([rgb, np.ones((len(rgb), 1), np.float32)], axis=1)
        cache[k] = (cloud, color)
        return cache[k]

    F = vis_end - vis_start
    points = np.zeros((F, max_points, 3), dtype=np.float32)
    colors = np.zeros((F, max_points, 4), dtype=np.float32)
    for i, t in enumerate(range(vis_start, vis_end)):
        j = int(np.searchsorted(tstamp, t))
        if j >= len(tstamp) or (j > 0 and t - tstamp[j - 1] <= tstamp[j] - t):
            j -= 1
        points[i], colors[i] = keyframe_cloud(max(j, 0))
    if cap is not None:
        cap.release()
    print(f"Point cloud: {len(cache)} keyframes used, {max_points} pts/frame "
          f"({points.nbytes / 1e6:.0f} MB), color="
          f"{'photometric' if photometric else 'colormap'}")
    return points, colors


def load_slam_cam_data(pred_cam):
    """load_slam_cam equivalent for an in-memory SLAM npz dict."""
    pred_traj = pred_cam['traj']
    scale = float(np.asarray(pred_cam['scale']).reshape(-1)[0])
    t_c2w_sla = torch.tensor(pred_traj[:, :3]) * scale
    pred_camq = torch.tensor(pred_traj[:, 3:])
    R_c2w_sla = quaternion_to_matrix(pred_camq[:, [3, 0, 1, 2]])
    R_w2c_sla = R_c2w_sla.transpose(-1, -2)
    t_w2c_sla = -torch.einsum("bij,bj->bi", R_w2c_sla, t_c2w_sla)
    return R_w2c_sla, t_w2c_sla, R_c2w_sla, t_c2w_sla


def read_focal(seq_dir, slam_path):
    """Focal in pixels, for the camera frustum shape."""
    est_focal_path = os.path.join(seq_dir, 'est_focal.txt')
    if os.path.exists(est_focal_path):
        try:
            return float(open(est_focal_path).read().strip())
        except ValueError:
            pass
    data = np.load(slam_path)
    if 'img_focal' in data:
        return float(data['img_focal'])
    return None


def resolve_image_source(args, seq_dir, vis_start, vis_end):
    """Only frame dimensions are needed for world viz; use whatever is at hand."""
    if args.video_path:
        # Probe dims via cv2 metadata only — a live decord reader in the same
        # process as the aitviewer Qt/GL context segfaults (bundled-lib clash),
        # and the world view never reads pixels anyway.
        cap = cv2.VideoCapture(args.video_path)
        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        if h > 0 and w > 0:
            return _FrameDims(h, w)
        return _FrameDims(args.height, args.width)
    img_dir = os.path.join(seq_dir, 'extracted_images')
    imgfiles = sorted(glob(os.path.join(img_dir, '*.jpg'))) or sorted(glob(os.path.join(img_dir, '*.png')))
    if imgfiles:
        return imgfiles[vis_start:vis_end]
    return _FrameDims(args.height, args.width)


def main():
    faulthandler.enable()  # native crashes print a traceback, not a bare SIGSEGV
    parser = argparse.ArgumentParser(description="Visualize offline HaWoR results (SLAM camera + 3D hands, world space)")
    parser.add_argument('--seq_dir', required=True, help='processed seq folder with cam_space/ and SLAM/')
    parser.add_argument('--video_path', default=None,
                        help='source video; needed for the infiller fallback (frame count)')
    parser.add_argument('--slam_npz', default=None, help='explicit SLAM npz (needed if seq has multiple)')
    parser.add_argument('--infiller_weight', default='./weights/hawor/checkpoints/infiller.pt')
    parser.add_argument('--target_fps', type=float, default=30, help='fps the seq was processed at')
    parser.add_argument('--headless', action='store_true',
                        help='disabled: this read-only viewer does not render files')
    parser.add_argument('--show_traj', action='store_true', help='dotted hand trajectory trail')
    parser.add_argument('--show_ghost', action='store_true', help='fading ghost hands')
    parser.add_argument('--frustum_depth', type=float, default=0.2,
                        help='camera frustum size (meters from optical center to image plane)')
    parser.add_argument('--max_points', type=int, default=20000,
                        help='points per frame for the SLAM depth point cloud (0 = off)')
    parser.add_argument('--point_size', type=float, default=8.0,
                        help='SLAM depth point cloud dot size in pixels (larger = easier to see up close)')
    parser.add_argument('--axes_length', type=float, default=0.3, help='world-origin axis triad length (meters)')
    parser.add_argument('--vis_start', type=int, default=None, help='first frame to visualize (target_fps timeline)')
    parser.add_argument('--vis_end', type=int, default=None, help='end frame (exclusive) to visualize')
    parser.add_argument('--no_follow', action='store_true',
                        help='start from the free viewer camera instead of the chase camera that '
                             'follows the SLAM camera through the fixed world')
    parser.add_argument('--follow_offset', default='0,0.4,1.2',
                        help='chase-camera offset in the SLAM camera frame, "right,up,back" meters '
                             '(default: 0.4 m above, 1.2 m behind; rotation follows the camera). '
                             'Use "0,0,0" for the exact ego view.')
    parser.add_argument('--center', action='store_true',
                        help='translate the world so the first visualized camera pose sits at the '
                             'origin (useful when SLAM drift pushed the sequence far away)')
    parser.add_argument('--output_dir', default=None,
                        help='kept for CLI compatibility; ignored in read-only interactive mode')
    parser.add_argument('--width', type=int, default=1920, help='viewer size when no frames are available')
    parser.add_argument('--height', type=int, default=1080, help='viewer size when no frames are available')
    parser.add_argument('--window_type', default=None,
                        help='aitviewer window backend override, e.g. glfw or pyglet '
                             '(default: aitviewer config, usually Qt)')
    args = parser.parse_args()

    if args.window_type:
        from aitviewer.configuration import CONFIG as _C
        _C.update_conf({"window_type": args.window_type})
    if args.headless:
        raise SystemExit(
            "--headless would render a video file; visualize_slam_offline.py is read-only. "
            "Use lib/vis/run_vis2.py callers or a dedicated render script for exports."
        )

    seq_dir = args.seq_dir
    slam_path, start_idx, end_idx = find_slam_npz(seq_dir, args.slam_npz)
    orig_slam_path = slam_path  # keeps disps (the stitched copy drops them)
    # Always stitch in memory: no-op for single-window results; for multi-window
    # merges the visualized camera poses must be continuous, but this script must
    # not create the old vis_stitched/ cache.
    slam_data, _ = stitch_slam_data(slam_path)
    _, _, R_c2w_all, t_c2w_all = load_slam_cam_data(slam_data)

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = load_or_build_world_res(
        args, seq_dir, start_idx, end_idx, allow_build=False)

    # Hand tensors live on the full video timeline; the SLAM traj covers
    # [start_idx, start_idx + len(traj)). Visualize the overlap.
    T = min(pred_trans.shape[1] - start_idx, R_c2w_all.shape[0])
    if T <= 0:
        raise SystemExit(f"No overlap between hand timeline ({pred_trans.shape[1]} frames) "
                         f"and SLAM window starting at {start_idx}")
    vis_start, vis_end = start_idx, start_idx + T
    if args.vis_start is not None:
        vis_start = max(vis_start, args.vis_start)
    if args.vis_end is not None:
        vis_end = min(vis_end, args.vis_end)
    if vis_end <= vis_start:
        raise SystemExit(f"Empty vis range [{vis_start}, {vis_end})")
    print(f"vis {vis_start} to {vis_end}")

    left_dict, right_dict = build_world_meshes(pred_trans, pred_rot, pred_hand_pose, pred_betas,
                                               vis_start, vis_end)

    # Map the SLAM camera frame to the viewer frame (same R_x flip as demo.py).
    R_x = torch.tensor([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]]).float()
    cam_sl = slice(vis_start - start_idx, vis_end - start_idx)
    R_c2w = torch.einsum('ij,njk->nik', R_x, R_c2w_all[cam_sl])
    t_c2w = torch.einsum('ij,nj->ni', R_x, t_c2w_all[cam_sl])
    left_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, left_dict['vertices'].cpu())
    right_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, right_dict['vertices'].cpu())

    # No further rotation: the SLAM world frame is the first camera pose, and
    # after the R_x flip +Y is exactly the first frame's camera-up. The scene is
    # therefore y-up by construction (up to how level the camera was at t=0).

    points = point_colors = None
    if args.max_points > 0:
        points, point_colors = build_point_clouds(
            orig_slam_path, R_c2w_all, t_c2w_all, R_x, vis_start, vis_end,
            start_idx, args.max_points, video_path=args.video_path,
            target_fps=args.target_fps)

    if args.center:
        # Rigid translation of the whole world; relative (metric) motion unchanged.
        origin = t_c2w[0].clone()
        t_c2w = t_c2w - origin
        left_dict['vertices'] = left_dict['vertices'] - origin
        right_dict['vertices'] = right_dict['vertices'] - origin
        if points is not None:
            points = points - np.asarray(origin, dtype=np.float32)
        print(f"Centered world on first vis frame (offset {origin.tolist()})")

    output_pth = args.output_dir or seq_dir
    image_source = resolve_image_source(args, seq_dir, vis_start, vis_end)
    img_focal = read_focal(seq_dir, slam_path)
    run_vis2_on_video(left_dict, right_dict, output_pth, img_focal, image_source,
                      R_c2w=R_c2w, t_c2w=t_c2w,
                      interactive=not args.headless,
                      show_traj=args.show_traj, show_ghost=args.show_ghost,
                      show_frustum=True, frustum_depth=args.frustum_depth,
                      show_axes=True, axes_length=args.axes_length,
                      show_ground=False, ground_up='y',
                      follow_camera=not args.no_follow,
                      follow_offset=tuple(float(x) for x in args.follow_offset.split(',')),
                      points=points, point_colors=point_colors,
                      point_size=args.point_size)
    print("finish")


if __name__ == '__main__':
    main()
