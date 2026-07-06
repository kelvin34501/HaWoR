"""SLAM stitching + offline world-space reconstruction helpers.

These functions operate purely on a processed seq folder:

    <seq_dir>/cam_space/<tid>/<start>_<end>.json   raw camera-space MANO chunks
    <seq_dir>/SLAM/hawor_slam_w_scale_<s>_<e>.npz  scaled camera trajectory
    <seq_dir>/world_space_res.pth                  (optional) infilled world-space result

They are deliberately kept free of any aitviewer / GL imports so they can be
reused from headless contexts (the batch service, offline build scripts) as well
as from the interactive visualizer (`scripts/visualize_slam_offline.py`).
"""
import os
import re
import argparse
from collections import defaultdict
from glob import glob

import numpy as np
import torch

CHUNK_NAME_RE = re.compile(r'^(\d+)_(\d+)$')


SLAM_NAME_RE = re.compile(r'hawor_slam_w_scale_(\d+)_(\d+)(?:_50fps)?\.npz$')


def find_slam_npz(seq_dir, explicit=None):
    if explicit:
        path = explicit
    else:
        # Exact-name filter: SLAM/ may also hold `_disps_` and `_50fps` variants.
        candidates = sorted(
            p for p in glob(os.path.join(seq_dir, 'SLAM', 'hawor_slam_w_scale_*.npz'))
            if re.search(r'hawor_slam_w_scale_(\d+)_(\d+)\.npz$', os.path.basename(p))
        )
        if not candidates:
            raise SystemExit(f"No SLAM npz found under {os.path.join(seq_dir, 'SLAM')}")
        if len(candidates) > 1:
            listing = '\n  '.join(candidates)
            raise SystemExit(f"Multiple SLAM files found; pick one with --slam_npz:\n  {listing}")
        path = candidates[0]
    m = SLAM_NAME_RE.search(os.path.basename(path))
    if not m:
        raise SystemExit(f"Unexpected SLAM file name: {path}")
    return path, int(m.group(1)), int(m.group(2))


def list_cam_space_chunks(seq_dir):
    """{tid: [(start, end_inclusive, json_path), ...]} from cam_space filenames."""
    cam_space = os.path.join(seq_dir, 'cam_space')
    if not os.path.isdir(cam_space):
        raise SystemExit(f"cam_space folder not found: {cam_space}")
    chunks_all = defaultdict(list)
    for tid_name in sorted(os.listdir(cam_space)):
        if not tid_name.isdigit():
            continue
        for cf in glob(os.path.join(cam_space, tid_name, '*.json')):
            m = CHUNK_NAME_RE.match(os.path.splitext(os.path.basename(cf))[0])
            if not m:
                print(f"Skip chunk with unexpected name: {cf}")
                continue
            chunks_all[int(tid_name)].append((int(m.group(1)), int(m.group(2)), cf))
        chunks_all[int(tid_name)].sort()
    return chunks_all


def stitch_slam_data(slam_path):
    """Return SLAM npz data with metadata stitching applied in memory."""
    from hawor.utils.rotation import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion

    data = dict(np.load(slam_path, allow_pickle=True))
    if bool(np.asarray(data.get('overlap_aligned', False)).reshape(-1)[0]):
        print("Stitch: branch=overlap_aligned; using merged SLAM unchanged")
        return data, False

    traj = np.asarray(data['traj'], dtype=np.float32).copy()
    t = torch.from_numpy(traj[:, :3])
    q_wxyz = torch.from_numpy(traj[:, [6, 3, 4, 5]].copy())

    # Seams are the recorded per-window SLAM restart frames (the hard chunk
    # boundaries the merge wrote); they are never inferred from pose geometry.
    # Older single-window outputs do not carry this metadata, and need no stitch.
    if 'window_starts' not in data:
        print("Stitch: branch=legacy_single_window; no window_starts metadata, "
              "using SLAM unchanged")
        return data, False
    starts = sorted(int(x) for x in np.asarray(data['window_starts']).reshape(-1))
    boundaries = [b for b in starts if b > starts[0] and 0 < b < len(t)]
    if not boundaries:
        print(f"Stitch: branch=single_window; window_starts={starts}, using SLAM unchanged")
        return data, False
    print(f"Stitch: branch=metadata_stitch; chaining {len(boundaries)} window boundary(ies) from recorded "
          f"window_starts: {boundaries}")

    T4 = torch.eye(4).repeat(len(t), 1, 1)
    T4[:, :3, :3] = quaternion_to_rotation_matrix(q_wxyz)
    T4[:, :3, 3] = t
    ends = boundaries[1:] + [len(t)]
    for b, e in zip(boundaries, ends):
        A = T4[b - 1] @ torch.linalg.inv(T4[b])
        T4[b:e] = A @ T4[b:e]
    traj[:, :3] = T4[:, :3, 3].numpy()
    traj[:, 3:7] = rotation_matrix_to_quaternion(T4[:, :3, :3])[:, [1, 2, 3, 0]].numpy()

    out_data = {k: v for k, v in data.items() if k != 'disps'}
    out_data['traj'] = traj
    return out_data, True


def stitch_slam_npz(seq_dir, slam_path):
    """Make a merged SLAM trajectory continuous across window restarts.

    The segmented merge records each independent SLAM window start in
    `window_starts`. If the merge already aligned windows through overlap
    frames, this is a no-op. Otherwise, each recorded interior window start is
    treated as a hard seam, and that window is left-composed with the rigid
    correction that makes its first pose equal to the previous frame's pose.

    Returns (slam_path, seq_dir) to use downstream. Unchanged when the
    trajectory is already continuous; otherwise a stitched npz is written under
    <seq_dir>/vis_stitched/SLAM/ (same filename, `disps` dropped) with
    cam_space symlinked next to it so the infiller runs against stitched poses.
    """
    data, changed = stitch_slam_data(slam_path)
    if not changed:
        return slam_path, seq_dir

    shadow = os.path.join(seq_dir, 'vis_stitched')
    os.makedirs(os.path.join(shadow, 'SLAM'), exist_ok=True)
    out_npz = os.path.join(shadow, 'SLAM', os.path.basename(slam_path))
    np.savez(out_npz, **data)
    cam_link = os.path.join(shadow, 'cam_space')
    if not os.path.exists(cam_link):
        os.symlink(os.path.abspath(os.path.join(seq_dir, 'cam_space')), cam_link)
    print(f"Stitch: saved {out_npz}")
    return out_npz, shadow


def load_or_build_world_res(args, seq_dir, start_idx, end_idx, allow_build=True):
    res_path = os.path.join(seq_dir, 'world_space_res.pth')
    if os.path.exists(res_path):
        print(f"Loading {res_path}")
        import joblib
        return joblib.load(res_path)

    if not allow_build:
        raise SystemExit(
            f"{res_path} not found; offline SLAM visualization is read-only. "
            "Run scripts/build_world_space_res.py first."
        )
    if not args.video_path:
        raise SystemExit(
            f"{res_path} not found; rebuilding it runs the infiller, which needs "
            "--video_path (frame count) and --infiller_weight."
        )
    print(f"{res_path} not found; running infiller from cam_space chunks ...")
    # Heavy import (loads torch model code); keep it off the fast path.
    from scripts.scripts_test_video.hawor_video import hawor_infiller
    frame_chunks_all = defaultdict(list)
    for tid, chunks in list_cam_space_chunks(seq_dir).items():
        frame_chunks_all[tid] = [torch.arange(s, e + 1) for s, e, _ in chunks]
    infill_args = argparse.Namespace(
        video_path=args.video_path,
        seq_dir=seq_dir,
        infiller_weight=args.infiller_weight,
        target_fps=args.target_fps,
        frame_start=0,
        frame_end=None,
        input_type='file',
    )
    # Also saves world_space_res.pth into seq_dir as a side effect.
    return hawor_infiller(infill_args, start_idx, end_idx, frame_chunks_all)
