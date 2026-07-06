#!/usr/bin/env python3
"""Build a full-sequence world-space hand result from a merged seq folder.

Given a processed seq folder that already holds camera-space chunks and a
(merged, multi-window) SLAM trajectory:

    <seq_dir>/cam_space/<tid>/<start>_<end>.json
    <seq_dir>/SLAM/hawor_slam_w_scale_0_<last>.npz

this stitches the SLAM trajectory into one continuous world frame when the
merged SLAM file records unaligned window boundaries, runs the motion infiller
over the whole timeline, and writes:

    <seq_dir>/world_space_res.pth   joblib list [pred_trans, pred_rot,
                                    pred_hand_pose, pred_betas, pred_valid]

The stitch step logs whether it used an overlap-aligned, single-window, legacy
single-window, or metadata-stitch branch. It may create a
<seq_dir>/vis_stitched/ shadow (stitched SLAM + cam_space symlink) so the
infiller runs against continuous poses; that shadow is removed before exit so
it never leaks into the published result.

This is the headless counterpart of scripts/visualize_slam_offline.py's
load_or_build_world_res path, sharing scripts/slam_world_utils.py (no aitviewer
import, safe to run inside the batch service).
"""
import os
import sys
import shutil
import argparse

import joblib

sys.path.insert(0, os.path.dirname(__file__) + '/..')

from scripts.slam_world_utils import find_slam_npz, stitch_slam_npz, load_or_build_world_res


def build_world_space_res(seq_dir, video_path, infiller_weight, target_fps):
    """Stitch + infill and return the world-space result path."""
    slam_path, start_idx, end_idx = find_slam_npz(seq_dir)
    # Always stitch: a no-op (returns the original path/seq_dir) for single-window
    # results; for multi-window merges the world result must come from the
    # stitched poses. A stitched run writes a vis_stitched/ shadow next to seq_dir.
    stitched_path, world_seq_dir = stitch_slam_npz(seq_dir, slam_path)

    args = argparse.Namespace(
        video_path=video_path,
        infiller_weight=infiller_weight,
        target_fps=target_fps,
    )
    # Runs the infiller (and dumps world_space_res.pth into world_seq_dir).
    result = load_or_build_world_res(args, world_seq_dir, start_idx, end_idx)

    # Canonicalize: always land the result at <seq_dir>/world_space_res.pth,
    # regardless of whether the stitched shadow was used.
    out_path = os.path.join(seq_dir, 'world_space_res.pth')
    joblib.dump(result, out_path)
    print(f"Saved {out_path}")

    # Drop the transient stitch shadow so it is not copied into the output.
    if world_seq_dir != seq_dir:
        shutil.rmtree(os.path.join(seq_dir, 'vis_stitched'), ignore_errors=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Stitch merged SLAM + infill full sequence -> world_space_res.pth")
    parser.add_argument('--seq_dir', required=True,
                        help='processed seq folder with cam_space/ and SLAM/')
    parser.add_argument('--video_path', required=True,
                        help='source video (needed by the infiller for the frame count)')
    parser.add_argument('--infiller_weight', default='./weights/hawor/checkpoints/infiller.pt')
    parser.add_argument('--target_fps', type=float, default=30,
                        help='fps the seq was processed at')
    args = parser.parse_args()

    build_world_space_res(args.seq_dir, args.video_path, args.infiller_weight, args.target_fps)
    print("finish")


if __name__ == '__main__':
    main()
