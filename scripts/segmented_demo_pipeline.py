#!/usr/bin/env python3
"""Segment -> demo.py -> merge chunk results.

This script is designed for long videos that are expensive to process in one pass.
It runs demo.py over consecutive frame-index windows of the input video (decoded on
demand via FrameSource — no chunk files, no JPEG dump) and merges per-frame
camera-space parameters plus SLAM/disparity artifacts back into a single
timeline.

Notes
- It merges camera-space params (init_root_orient/init_hand_pose/init_trans/init_betas)
  and the per-window SLAM/disparity artifacts.
- No world-space hand result is built here.
- Each window is processed in its own seq dir (work_dir/chunk_NNNNNN); the global
  frame offset is the window start index.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.pipeline.frame_source import FrameSource
from lib.pipeline.window_planner import compute_windows, processing_windows


RANGE_RE = re.compile(r"(\d+)_(\d+)(?:_50fps)?\.json$")
RANGE_NAME_RE = re.compile(r"^(\d+)_(\d+)((?:_50fps)?\.json)$")
FIELDS = ["init_root_orient", "init_hand_pose", "init_trans", "init_betas"]
SLAM_MAIN_RE = re.compile(r"^hawor_slam_w_scale_(\d+)_(\d+)\.npz$")


@dataclass
class ChunkMeta:
    chunk_path: str
    seq_dir: str
    owned_start: int
    owned_end: int
    process_start: int
    process_end: int

    @property
    def frame_count(self) -> int:
        return self.owned_end - self.owned_start


def log(msg: str) -> None:
    print(msg, flush=True)


def run_cmd(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> None:
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def parse_range(path: str) -> Tuple[int, int]:
    m = RANGE_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Invalid chunk json name (expect start_end.json): {path}")
    return int(m.group(1)), int(m.group(2))


def seq_name_for_video(video_path: str) -> str:
    # Keep behavior consistent with detect_track_video.py
    return os.path.basename(video_path).split(".")[0]


def count_frames(video_path: str, target_fps: float) -> int:
    """Number of frames in the (resampled) timeline demo.py will see."""
    return len(FrameSource(video_path, target_fps=target_fps))


def run_demo_on_chunk(
    video_abs: str,
    seq_dir: str,
    frame_start: int,
    frame_end: int,
    target_fps: float,
    project_dir: str,
    python_bin: str,
    vis_mode: str,
    gpu_id: Optional[int],
) -> str:
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    os.makedirs(seq_dir, exist_ok=True)
    cmd = [
        python_bin, "demo.py",
        "--video_path", video_abs,
        "--vis_mode", vis_mode,
        "--target_fps", str(target_fps),
        "--frame_start", str(frame_start),
        "--frame_end", str(frame_end),
        "--seq_dir", seq_dir,
    ]
    log(f"[demo] {' '.join(cmd)}")
    run_cmd(cmd, cwd=project_dir, env=env)

    if not os.path.isdir(seq_dir):
        raise RuntimeError(f"Chunk output seq dir missing: {seq_dir}")
    return seq_dir


def chunk_local_span(cam_space_dir: str) -> int:
    max_end = -1
    for hand_dir in sorted(glob.glob(os.path.join(cam_space_dir, "*"))):
        if not os.path.isdir(hand_dir):
            continue
        for fp in glob.glob(os.path.join(hand_dir, "*.json")):
            _, end = parse_range(fp)
            max_end = max(max_end, end)
    return max_end + 1 if max_end >= 0 else 0


def to_np(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float32)


def ensure_capacity(arr: np.ndarray, new_t: int) -> np.ndarray:
    if arr.shape[1] >= new_t:
        return arr
    pad_shape = list(arr.shape)
    pad_shape[1] = new_t - arr.shape[1]
    pad = np.zeros(pad_shape, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)


def ensure_capacity_mask(mask: np.ndarray, new_t: int) -> np.ndarray:
    if mask.shape[0] >= new_t:
        return mask
    pad = np.zeros((new_t - mask.shape[0],), dtype=bool)
    return np.concatenate([mask, pad], axis=0)


def shifted_json_name(json_path: str, offset: int) -> str:
    name = os.path.basename(json_path)
    m = RANGE_NAME_RE.match(name)
    if not m:
        raise ValueError(f"Invalid chunk json name (expect start_end.json): {json_path}")
    start = int(m.group(1)) + offset
    end = int(m.group(2)) + offset
    suffix = m.group(3)
    return f"{start}_{end}{suffix}"


def clipped_cam_space_chunk(json_path: str, meta: ChunkMeta):
    name = os.path.basename(json_path)
    m = RANGE_NAME_RE.match(name)
    if not m:
        raise ValueError(f"Invalid chunk json name (expect start_end.json): {json_path}")

    local_start = int(m.group(1))
    local_end = int(m.group(2))
    suffix = m.group(3)
    global_start = meta.process_start + local_start
    global_end = meta.process_start + local_end
    keep_start = max(global_start, meta.owned_start)
    keep_end = min(global_end, meta.owned_end - 1)
    if keep_start > keep_end:
        return None

    local_keep_start = keep_start - global_start
    local_keep_end = keep_end - global_start + 1
    with open(json_path, "r") as f:
        data = json.load(f)

    out = {}
    expected_len = local_end - local_start + 1
    for key, value in data.items():
        if key in FIELDS:
            arr = np.asarray(value)
            if arr.ndim < 2 or arr.shape[1] < local_keep_end:
                raise ValueError(
                    f"Invalid {key} shape in {json_path}: {arr.shape}; "
                    f"expected at least {expected_len} frames on axis 1"
                )
            out[key] = arr[:, local_keep_start:local_keep_end].tolist()
        else:
            out[key] = value

    return f"{keep_start}_{keep_end}{suffix}", out


def merge_cam_space(
    chunk_metas: List[ChunkMeta],
    out_dir: str,
    overlap_policy: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    copied_count = 0

    for meta in chunk_metas:
        cam_space_dir = os.path.join(meta.seq_dir, "cam_space")
        if not os.path.isdir(cam_space_dir):
            log(f"[merge] Skip chunk with no cam_space: {meta.chunk_path}")
            continue

        hand_dirs = [d for d in sorted(glob.glob(os.path.join(cam_space_dir, "*"))) if os.path.isdir(d)]
        for hand_dir in hand_dirs:
            hand_id = os.path.basename(hand_dir)
            json_files = sorted(glob.glob(os.path.join(hand_dir, "*.json")))
            if not json_files:
                continue
            hand_out_dir = os.path.join(out_dir, hand_id)
            os.makedirs(hand_out_dir, exist_ok=True)

            for fp in json_files:
                clipped = clipped_cam_space_chunk(fp, meta)
                if clipped is None:
                    continue
                dst_name, data = clipped
                dst_path = os.path.join(hand_out_dir, dst_name)

                if os.path.exists(dst_path):
                    if overlap_policy == "keep_first":
                        continue
                    os.remove(dst_path)

                with open(dst_path, "w") as f:
                    json.dump(data, f, indent=1)
                copied_count += 1

    if copied_count == 0:
        raise RuntimeError("No cam_space chunks found to merge")
    log(f"[merge] Copied {copied_count} cam_space json files into {out_dir}")


def merge_slam(chunk_metas: List[ChunkMeta], out_dir: str, overlap_policy: str) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    from hawor.utils.rotation import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion
    import torch

    traj_buf: Optional[np.ndarray] = None
    traj_valid = np.zeros((0,), dtype=bool)
    # Sparse keyframe storage for disps: global_frame_idx -> disps_frame (H, W[, C...]).
    disps_map: Dict[int, np.ndarray] = {}

    img_focal = None
    img_center = None
    # Each window is an independent SLAM run. Normalize every window to metric,
    # align overlapping boundary frames when present, and write only the owned
    # non-overlapping range to the merged trajectory.
    window_starts: List[int] = []
    slam_run_ranges: List[Tuple[int, int]] = []
    overlap_frames: List[int] = []
    anchors: Dict[int, torch.Tensor] = {}
    expected_end = max((m.owned_end for m in chunk_metas), default=0)
    saw_alignment_gap = False

    def _traj_to_t4(traj_arr: np.ndarray) -> torch.Tensor:
        t4 = torch.eye(4, dtype=torch.float32).repeat(traj_arr.shape[0], 1, 1)
        q_wxyz = torch.from_numpy(traj_arr[:, [6, 3, 4, 5]].copy()).float()
        t4[:, :3, :3] = quaternion_to_rotation_matrix(q_wxyz)
        t4[:, :3, 3] = torch.from_numpy(traj_arr[:, :3].copy()).float()
        return t4

    def _t4_to_traj(t4: torch.Tensor, template: np.ndarray) -> np.ndarray:
        out = np.asarray(template, dtype=np.float32).copy()
        out[:, :3] = t4[:, :3, 3].cpu().numpy()
        q_wxyz = rotation_matrix_to_quaternion(t4[:, :3, :3])
        out[:, 3:7] = q_wxyz[:, [1, 2, 3, 0]].cpu().numpy()
        return out

    def _ensure_traj_capacity(new_t: int) -> None:
        nonlocal traj_buf, traj_valid
        if traj_valid.shape[0] >= new_t:
            return
        old_t = traj_valid.shape[0]
        traj_valid = np.concatenate([traj_valid, np.zeros((new_t - old_t,), dtype=bool)], axis=0)
        if traj_buf is not None:
            pad = np.zeros((new_t - old_t, traj_buf.shape[1]), dtype=traj_buf.dtype)
            traj_buf = np.concatenate([traj_buf, pad], axis=0)

    def _normalize_local_tstamp(local_tstamp: np.ndarray, local_start: int, local_end: int) -> np.ndarray:
        if local_tstamp.size == 0:
            return local_tstamp
        min_ts = int(local_tstamp.min())
        max_ts = int(local_tstamp.max())
        # Two common conventions are supported:
        # 1) local absolute frame index in [local_start, local_end)
        # 2) local relative index in [0, local_end-local_start), shift by local_start
        if min_ts >= local_start and max_ts < local_end:
            return local_tstamp
        local_span = local_end - local_start
        if min_ts >= 0 and max_ts < local_span:
            return local_tstamp + local_start
        return local_tstamp

    for meta in sorted(chunk_metas, key=lambda m: m.owned_start):
        slam_dir = os.path.join(meta.seq_dir, "SLAM")
        if not os.path.isdir(slam_dir):
            log(f"[merge] Skip chunk with no SLAM: {meta.chunk_path}")
            continue

        slam_files = []
        for fp in sorted(glob.glob(os.path.join(slam_dir, "hawor_slam_w_scale_*.npz"))):
            name = os.path.basename(fp)
            if "_disps_" in name or name.endswith("_50fps.npz"):
                continue
            m = SLAM_MAIN_RE.match(name)
            if m:
                slam_files.append((int(m.group(1)), int(m.group(2)), fp))

        for local_start, local_end, fp in slam_files:
            data = dict(np.load(fp, allow_pickle=True))
            traj = np.asarray(data.get("traj"), dtype=np.float32)
            disps = np.asarray(data.get("disps"), dtype=np.float32)
            if traj.ndim != 2 or traj.shape[1] < 7:
                raise ValueError(f"Invalid traj shape in {fp}: {traj.shape}")
            if disps.ndim < 1:
                raise ValueError(f"Invalid disps shape in {fp}: {disps.shape}")

            local_span = local_end - local_start
            traj_len = min(traj.shape[0], local_span)
            if traj_len <= 0:
                continue

            # Normalize this window to metric before placement: translations to
            # meters, disps to inverse-metric-depth. Rotations (traj[:, 3:7]) are
            # scale-invariant. After this every window shares one metric frame.
            win_scale = float(np.asarray(data["scale"]).reshape(-1)[0]) if "scale" in data else 1.0
            traj = traj.copy()
            traj[:, :3] *= win_scale
            disps = disps / win_scale
            traj = traj[:traj_len]
            t4 = _traj_to_t4(traj)
            file_global_start = meta.process_start + local_start
            file_global_end = file_global_start + traj_len
            slam_run_ranges.append((file_global_start, file_global_end))

            boundary = meta.owned_start
            boundary_row = boundary - file_global_start
            if boundary in anchors and 0 <= boundary_row < traj_len:
                A = anchors[boundary] @ torch.linalg.inv(t4[boundary_row])
                t4 = A.unsqueeze(0) @ t4
                overlap_frames.append(int(boundary))
            elif meta.owned_start > 0:
                saw_alignment_gap = True

            anchor_boundary = meta.owned_end
            anchor_row = anchor_boundary - file_global_start
            if meta.process_end > meta.owned_end and 0 <= anchor_row < traj_len:
                anchors[int(anchor_boundary)] = t4[anchor_row].clone()

            traj = _t4_to_traj(t4, traj)

            tstamp = np.asarray(data.get("tstamp", []), dtype=np.int64).reshape(-1)
            use_tstamp = tstamp.shape[0] == disps.shape[0] and tstamp.shape[0] > 0

            if traj_buf is None:
                traj_buf = np.zeros((0, traj.shape[1]), dtype=np.float32)

            row_globals = np.arange(file_global_start, file_global_end, dtype=np.int64)
            owned_rows = np.where((row_globals >= meta.owned_start) & (row_globals < meta.owned_end))[0]
            if owned_rows.size == 0:
                continue

            global_start = int(row_globals[owned_rows[0]])
            global_end = int(row_globals[owned_rows[-1]]) + 1
            _ensure_traj_capacity(global_end)
            # Record each owned window start so downstream code can either verify
            # a single-window/no-op path or stitch unaligned multi-window merges.
            window_starts.append(int(meta.owned_start))

            if overlap_policy == "keep_first":
                write_globals = row_globals[owned_rows]
                write_mask = ~traj_valid[write_globals]
                write_idx = owned_rows[write_mask]
                if write_idx.size > 0:
                    for j in write_idx:
                        g = int(row_globals[j])
                        traj_buf[g] = traj[j]
                        traj_valid[g] = True
            else:
                for j in owned_rows:
                    g = int(row_globals[j])
                    traj_buf[g] = traj[j]
                    traj_valid[g] = True

            if use_tstamp:
                local_tstamp = _normalize_local_tstamp(tstamp, local_start, local_end)
            else:
                disp_len = min(disps.shape[0], local_span)
                local_tstamp = np.arange(disp_len, dtype=np.int64) + local_start
                disps = disps[:disp_len]

            if local_tstamp.shape[0] != disps.shape[0]:
                keep = min(local_tstamp.shape[0], disps.shape[0])
                local_tstamp = local_tstamp[:keep]
                disps = disps[:keep]

            for i, ts_local in enumerate(local_tstamp):
                ts_global = int(meta.process_start + int(ts_local))
                if ts_global < meta.owned_start or ts_global >= meta.owned_end:
                    continue
                if overlap_policy == "keep_first" and ts_global in disps_map:
                    continue
                disps_map[ts_global] = np.asarray(disps[i], dtype=np.float32)

            if img_focal is None and "img_focal" in data:
                img_focal = data["img_focal"]
            if img_center is None and "img_center" in data:
                img_center = data["img_center"]

    if traj_buf is None or not traj_valid.any() or not disps_map:
        log("[merge] No SLAM chunks found to merge")
        return None

    starts = sorted(set(window_starts))
    if not starts:
        # Should be unreachable given the disps_map/traj_valid guard above, but a
        # merged SLAM without recorded boundaries cannot be stitched -> hard fail.
        raise RuntimeError("[merge] No window_starts recorded; aborting SLAM merge")

    if expected_end <= 0:
        raise RuntimeError("[merge] Empty owned frame range; aborting SLAM merge")
    if traj_valid.shape[0] < expected_end or not traj_valid[:expected_end].all():
        missing = np.where(~traj_valid[:expected_end])[0]
        preview = ", ".join(str(int(x)) for x in missing[:10])
        raise RuntimeError(f"[merge] Merged SLAM is missing owned frame(s): {preview}")

    traj_out = traj_buf[:expected_end]
    tstamp = np.asarray(sorted(disps_map.keys()), dtype=np.int32)
    disps_out = np.stack([disps_map[int(ts)] for ts in tstamp], axis=0).astype(np.float32)
    # Every window was normalized to metres in-loop, so the merged frame is metric
    # and the single stored scale is exactly 1.0.
    scale_out = np.float32(1.0)

    overlap_aligned = bool(overlap_frames and not saw_alignment_gap)
    out_path = os.path.join(out_dir, f"hawor_slam_w_scale_0_{expected_end}.npz")
    np.savez(
        out_path,
        tstamp=tstamp,
        traj=traj_out,
        disps=disps_out,
        img_focal=np.asarray(0.0 if img_focal is None else img_focal),
        img_center=np.asarray([0.0, 0.0] if img_center is None else img_center),
        scale=np.asarray(scale_out),
        window_starts=np.asarray(starts, dtype=np.int64),
        slam_run_ranges=np.asarray(slam_run_ranges, dtype=np.int64),
        overlap_frames=np.asarray(sorted(set(overlap_frames)), dtype=np.int64),
        overlap_aligned=np.asarray(overlap_aligned),
    )
    log(f"[merge] Saved {out_path} (frames=[0, {expected_end}))")
    return out_path


def build_chunk_meta(windows: List[Tuple[int, int]], proc_windows: List[Tuple[int, int]], seq_dirs: List[str]) -> List[ChunkMeta]:
    metas: List[ChunkMeta] = []
    for (owned_start, owned_end), (process_start, process_end), seq_dir in zip(windows, proc_windows, seq_dirs):
        metas.append(
            ChunkMeta(
                chunk_path=seq_dir,
                seq_dir=seq_dir,
                owned_start=owned_start,
                owned_end=owned_end,
                process_start=process_start,
                process_end=process_end,
            )
        )
    return metas


def write_manifest(manifest_path: str, video_path: str, metas: List[ChunkMeta]) -> None:
    data = {
        "video": video_path,
        "chunks": [
            {
                "chunk_path": m.chunk_path,
                "seq_dir": m.seq_dir,
                "owned_start": m.owned_start,
                "owned_end": m.owned_end,
                "process_start": m.process_start,
                "process_end": m.process_end,
                "frame_count": m.frame_count,
            }
            for m in metas
        ],
    }
    with open(manifest_path, "w") as f:
        json.dump(data, f, indent=2)
    log(f"[manifest] {manifest_path}")


def process_video(args: argparse.Namespace, video_path: str) -> None:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    video_abs = os.path.abspath(video_path)
    video_stem = os.path.splitext(os.path.basename(video_abs))[0]

    if args.work_dir:
        work_root = os.path.abspath(args.work_dir)
    else:
        work_root = os.path.join(os.path.dirname(video_abs), f"{video_stem}_segmented")

    merged_root = os.path.join(work_root, "merged")
    os.makedirs(work_root, exist_ok=True)

    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    log(f"[video] {video_abs}")
    log(f"[work ] {work_root}")

    # Plan consecutive frame-index windows over the (resampled) video timeline.
    chunk_frames = max(1, int(round(args.segment_seconds * args.target_fps)))
    min_last_frames = max(0, int(round(args.min_last_segment_seconds * args.target_fps)))
    n_frames = count_frames(video_abs, args.target_fps)
    windows = compute_windows(
        n_frames,
        chunk_frames,
        min_last_frames,
        max_chunk_frames=args.max_chunk_frames,
    )
    if not windows:
        raise RuntimeError(f"No frames decoded from {video_abs}")
    proc_windows = processing_windows(windows, n_frames)
    largest_process_window = max(end - start for start, end in proc_windows)
    log(
        f"[plan ] {n_frames} frames @ {args.target_fps}fps -> {len(windows)} windows "
        f"(+1 SLAM overlap, largest={largest_process_window}, cap={args.max_chunk_frames})"
    )

    seq_dirs: List[str] = [os.path.join(work_root, f"chunk_{i:06d}") for i in range(len(windows))]

    if args.skip_demo:
        for seq_dir in seq_dirs:
            if not os.path.isdir(seq_dir):
                raise RuntimeError(f"Missing seq_dir (cannot --skip_demo): {seq_dir}")
        log("[demo ] skip")
    else:
        for i, ((owned_start, owned_end), (process_start, process_end), seq_dir) in enumerate(zip(windows, proc_windows, seq_dirs)):
            log(f"[demo ] window {i + 1}/{len(windows)} owned [{owned_start}, {owned_end}) process [{process_start}, {process_end})")
            run_demo_on_chunk(
                video_abs=video_abs,
                seq_dir=seq_dir,
                frame_start=process_start,
                frame_end=process_end,
                target_fps=args.target_fps,
                project_dir=project_dir,
                python_bin=args.python_bin,
                vis_mode=args.vis_mode,
                gpu_id=args.gpu_id,
            )

    if args.skip_merge:
        log("[merge] skip")
        return

    metas = build_chunk_meta(windows, proc_windows, seq_dirs)
    manifest_path = os.path.join(work_root, "chunk_manifest.json")
    write_manifest(manifest_path, video_abs, metas)

    merged_cam_dir = os.path.join(merged_root, "cam_space")
    merge_cam_space(
        chunk_metas=metas,
        out_dir=merged_cam_dir,
        overlap_policy=args.overlap_policy,
    )

    merged_slam_dir = os.path.join(merged_root, "SLAM")
    merged_slam_file = merge_slam(
        chunk_metas=metas,
        out_dir=merged_slam_dir,
        overlap_policy=args.overlap_policy,
    )

    # Also place merged result under the original video seq folder for downstream tools.
    # Skip when --skip_copy_back is set (e.g. service mode where the video lives on
    # s3mount and shutil.copy2 / writes next to the video are both forbidden).
    if not args.skip_copy_back:
        orig_seq_dir = os.path.join(os.path.dirname(video_abs), seq_name_for_video(video_abs))
        os.makedirs(orig_seq_dir, exist_ok=True)
        target_dir = os.path.join(orig_seq_dir, args.output_subdir)
        os.makedirs(target_dir, exist_ok=True)

        for root, _, files in os.walk(merged_cam_dir):
            for fn in files:
                if not fn.endswith(".json"):
                    continue
                fp = os.path.join(root, fn)
                rel = os.path.relpath(fp, merged_cam_dir)
                dst = os.path.join(target_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(fp, dst)
                log(f"[copy ] {dst}")

        if merged_slam_file is not None:
            orig_slam_dir = os.path.join(orig_seq_dir, "SLAM")
            os.makedirs(orig_slam_dir, exist_ok=True)
            slam_dst = os.path.join(orig_slam_dir, os.path.basename(merged_slam_file))
            shutil.copy2(merged_slam_file, slam_dst)
            log(f"[copy ] {slam_dst}")

        log(f"[done ] merged cam_space available at: {target_dir}")
    else:
        log(f"[done ] merged results at: {merged_root} (copy-back skipped)")


def collect_videos(args: argparse.Namespace) -> List[str]:
    if args.video_path:
        return [args.video_path]

    if not args.video_dir:
        raise ValueError("Either --video_path or --video_dir is required")

    found: List[str] = []
    for root, _, files in os.walk(args.video_dir):
        for fn in files:
            lower = fn.lower()
            if lower.endswith(".mp4"):
                found.append(os.path.join(root, fn))
    found.sort()
    return found


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Segment video(s), run demo.py per chunk, merge chunk outputs")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video_path", type=str, help="Single input video path")
    src.add_argument("--video_dir", type=str, help="Recursively process all .mp4 in directory")

    p.add_argument("--segment_seconds", type=int, default=120, help="Window duration in seconds (converted to frames via --target_fps)")
    p.add_argument(
        "--min_last_segment_seconds",
        type=int,
        default=50,
        help="Preferred minimum duration for the final window; merge or rebalance it without exceeding --max_chunk_frames",
    )
    p.add_argument(
        "--max_chunk_frames",
        type=int,
        default=3001,
        help="Hard cap on frames processed by one window, including the one-frame SLAM overlap (default: 3001)",
    )
    p.add_argument("--target_fps", type=float, default=30, help="Decode/resample fps (matches old ffmpeg fps=N); passed to demo.py")

    p.add_argument("--python_bin", type=str, default=sys.executable, help="Python executable for demo.py")
    p.add_argument("--vis_mode", type=str, default="off", help="demo.py --vis_mode value")
    p.add_argument("--gpu_id", type=int, default=None, help="Set CUDA_VISIBLE_DEVICES for demo.py")

    p.add_argument("--work_dir", type=str, default=None, help="Working directory (single-video mode recommended)")
    p.add_argument("--output_subdir", type=str, default="cam_space", help="Output subdir under original seq folder")
    p.add_argument("--skip_copy_back", action="store_true",
                   help="Skip copying merged results back to the original video sequence directory "
                        "(required when the video lives on an s3mount target)")
    p.add_argument("--overlap_policy", choices=["keep_last", "keep_first"], default="keep_last")

    # Deprecated no-ops kept for backward compatibility with existing launchers.
    # The pipeline no longer splits the video into chunk files; it decodes
    # frame-index windows on demand instead.
    p.add_argument("--reencode", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--overwrite_chunks", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--skip_split", action="store_true", help=argparse.SUPPRESS)

    p.add_argument("--skip_demo", action="store_true")
    p.add_argument("--skip_merge", action="store_true")

    return p


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.min_last_segment_seconds < 0:
        raise ValueError("--min_last_segment_seconds must be >= 0")
    if args.max_chunk_frames < 2:
        raise ValueError("--max_chunk_frames must be >= 2")

    videos = collect_videos(args)
    if not videos:
        raise RuntimeError("No videos found")

    if args.video_dir and args.work_dir:
        raise ValueError("--work_dir is only supported in single video mode (--video_path)")

    log(f"[main ] videos={len(videos)}")
    for i, video_path in enumerate(videos, start=1):
        log("=" * 80)
        log(f"[main ] ({i}/{len(videos)}) {video_path}")
        process_video(args, video_path)


if __name__ == "__main__":
    main()
