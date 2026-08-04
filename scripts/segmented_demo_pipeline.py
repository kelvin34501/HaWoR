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


def merge_cam_space(
    chunk_metas: List[ChunkMeta],
    out_dir: str,
    overlap_policy: str,
) -> None:
    """Blend camera-space predictions from shared context into one timeline.

    Independent outer windows restart tracking, and a hand track's first frame
    need not share the global 16-frame model phase.  Both adjacent runs predict
    the shared context, so taper those predictions across the boundary instead
    of hard-cutting from one temporal phase to another.
    """
    os.makedirs(out_dir, exist_ok=True)
    # hand -> global frame -> [(window weight, per-field frame values)]
    samples: Dict[str, Dict[int, List[Tuple[float, Dict[str, np.ndarray]]]]] = {}
    timeline_end = max((meta.owned_end for meta in chunk_metas), default=0)
    output_boundaries = {
        meta.owned_end
        for meta in chunk_metas
        if meta.owned_end < timeline_end
    }

    def _window_weight(meta: ChunkMeta, global_frame: int) -> float:
        if global_frame < meta.owned_start:
            span = max(1, meta.owned_start - meta.process_start)
            return (global_frame - meta.process_start + 1) / (span + 1)
        if global_frame >= meta.owned_end:
            span = max(1, meta.process_end - meta.owned_end)
            return (meta.process_end - global_frame) / (span + 1)
        return 1.0

    def _weighted_rotation(values: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
        matrices = np.stack(values, axis=0).astype(np.float64)
        weight_shape = (len(weights),) + (1,) * (matrices.ndim - 1)
        mixed = np.sum(matrices * weights.reshape(weight_shape), axis=0)
        u, _, vh = np.linalg.svd(mixed)
        rotation = u @ vh
        negative = np.linalg.det(rotation) < 0
        if np.any(negative):
            u = u.copy()
            u[..., :, -1] *= np.where(negative, -1.0, 1.0)[..., None]
            rotation = u @ vh
        return rotation.astype(np.float32)

    def _blend_frame(
        frame_samples: List[Tuple[float, Dict[str, np.ndarray]]]
    ) -> Dict[str, np.ndarray]:
        if len(frame_samples) == 1:
            return {
                field: np.asarray(frame_samples[0][1][field], dtype=np.float32).copy()
                for field in FIELDS
            }
        weights = np.asarray([sample[0] for sample in frame_samples], dtype=np.float64)
        weights /= weights.sum()
        blended: Dict[str, np.ndarray] = {}
        for field in FIELDS:
            values = [sample[1][field] for sample in frame_samples]
            if field in {"init_root_orient", "init_hand_pose"}:
                blended[field] = _weighted_rotation(values, weights)
            else:
                stacked = np.stack(values, axis=0).astype(np.float64)
                weight_shape = (len(weights),) + (1,) * (stacked.ndim - 1)
                blended[field] = np.sum(
                    stacked * weights.reshape(weight_shape), axis=0
                ).astype(np.float32)
        return blended

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
            hand_samples = samples.setdefault(hand_id, {})

            for fp in json_files:
                local_start, local_end = parse_range(fp)
                with open(fp, "r") as f:
                    data = json.load(f)
                arrays = {field: np.asarray(data[field]) for field in FIELDS}
                if any(array.ndim < 2 or array.shape[0] < 1 for array in arrays.values()):
                    shapes = {field: array.shape for field, array in arrays.items()}
                    raise ValueError(f"Invalid camera-space field shape in {fp}: {shapes}")
                lengths = {field: array.shape[1] for field, array in arrays.items()}
                expected_length = local_end - local_start + 1
                if any(length < expected_length for length in lengths.values()):
                    raise ValueError(
                        f"Invalid camera-space field length in {fp}: {lengths}; "
                        f"expected at least {expected_length}"
                    )

                for local_offset in range(expected_length):
                    global_frame = meta.process_start + local_start + local_offset
                    if global_frame < 0:
                        continue
                    # Never publish right-context samples beyond the owned video
                    # timeline. They remain available only as contributions to an
                    # adjacent window's owned frames.
                    if global_frame >= timeline_end:
                        continue
                    frame_values = {
                        field: np.asarray(
                            array[0, local_offset],
                            dtype=np.float32,
                        ).copy()
                        for field, array in arrays.items()
                    }
                    hand_samples.setdefault(global_frame, []).append(
                        (_window_weight(meta, global_frame), frame_values)
                    )

    written_chunks = 0
    for hand_id, hand_samples in sorted(samples.items()):
        if not hand_samples:
            continue
        hand_out_dir = os.path.join(out_dir, hand_id)
        os.makedirs(hand_out_dir, exist_ok=True)
        frames = sorted(hand_samples)
        breaks = [0]
        breaks.extend(
            index
            for index in range(1, len(frames))
            if (
                frames[index] != frames[index - 1] + 1
                or frames[index] in output_boundaries
            )
        )
        breaks.append(len(frames))
        for begin, finish in zip(breaks, breaks[1:]):
            segment_frames = frames[begin:finish]
            blended_frames = [
                _blend_frame(hand_samples[global_frame])
                for global_frame in segment_frames
            ]
            output = {
                field: np.stack(
                    [frame[field] for frame in blended_frames],
                    axis=0,
                )[None].tolist()
                for field in FIELDS
            }
            path = os.path.join(
                hand_out_dir,
                f"{segment_frames[0]}_{segment_frames[-1]}.json",
            )
            with open(path, "w") as f:
                json.dump(output, f, indent=1)
            written_chunks += 1

    if written_chunks == 0:
        raise RuntimeError("No cam_space chunks found to merge")
    log(
        f"[merge] Blended camera-space context into {written_chunks} contiguous "
        f"track chunk(s) under {out_dir} (policy={overlap_policy})"
    )


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
    # robustly align it from all shared context poses, and write only the owned
    # non-overlapping range to the merged trajectory.
    window_starts: List[int] = []
    slam_run_ranges: List[Tuple[int, int]] = []
    overlap_frames: List[int] = []
    slam_fallback_ranges: List[Tuple[int, int]] = []
    slam_fallback_modes: List[str] = []
    alignment_refs: Dict[int, torch.Tensor] = {}
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

    def _robust_rigid_alignment(corrections: torch.Tensor) -> torch.Tensor:
        """Average per-overlap rigid corrections without averaging matrices."""
        rotation_sum = corrections[:, :3, :3].sum(dim=0)
        u, _, vh = torch.linalg.svd(rotation_sum)
        rotation = u @ vh
        if torch.linalg.det(rotation) < 0:
            u = u.clone()
            u[:, -1] *= -1
            rotation = u @ vh
        translation = corrections[:, :3, 3].median(dim=0).values
        aligned = torch.eye(4, dtype=corrections.dtype)
        aligned[:3, :3] = rotation
        aligned[:3, 3] = translation
        return aligned

    def _blend_rigid(reference: torch.Tensor, current: torch.Tensor, alpha: float) -> torch.Tensor:
        """Blend a close pair of rigid poses and project rotation back to SO(3)."""
        mixed_rotation = (
            (1.0 - alpha) * reference[:3, :3]
            + alpha * current[:3, :3]
        )
        u, _, vh = torch.linalg.svd(mixed_rotation)
        rotation = u @ vh
        if torch.linalg.det(rotation) < 0:
            u = u.clone()
            u[:, -1] *= -1
            rotation = u @ vh
        blended = torch.eye(4, dtype=current.dtype)
        blended[:3, :3] = rotation
        blended[:3, 3] = (
            (1.0 - alpha) * reference[:3, 3]
            + alpha * current[:3, 3]
        )
        return blended

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
            slam_valid = bool(
                np.asarray(data.get("slam_valid", True)).reshape(-1)[0]
            )
            if not slam_valid:
                fallback_mode = str(
                    np.asarray(
                        data.get("slam_fallback", "unspecified")
                    ).reshape(-1)[0]
                )
                fallback_start = max(file_global_start, meta.owned_start)
                fallback_end = min(file_global_end, meta.owned_end)
                if fallback_start < fallback_end:
                    slam_fallback_ranges.append(
                        (fallback_start, fallback_end)
                    )
                    slam_fallback_modes.append(fallback_mode)

            all_row_globals = np.arange(
                file_global_start,
                file_global_end,
                dtype=np.int64,
            )
            if meta.owned_start > 0:
                matched = [
                    (row, int(global_frame))
                    for row, global_frame in enumerate(all_row_globals)
                    if int(global_frame) in alignment_refs
                ]
                if matched:
                    corrections = torch.stack(
                        [
                            alignment_refs[global_frame]
                            @ torch.linalg.inv(t4[row])
                            for row, global_frame in matched
                        ],
                        dim=0,
                    )
                    alignment = _robust_rigid_alignment(corrections)
                    t4 = alignment.unsqueeze(0) @ t4
                    overlap_frames.extend(
                        global_frame for _, global_frame in matched
                    )
                    owned_overlap = [
                        (row, global_frame)
                        for row, global_frame in matched
                        if global_frame >= meta.owned_start
                    ]
                    for index, (row, global_frame) in enumerate(owned_overlap):
                        alpha = (index + 1) / len(owned_overlap)
                        t4[row] = _blend_rigid(
                            alignment_refs[global_frame],
                            t4[row],
                            alpha,
                        )
                else:
                    saw_alignment_gap = True

            # Keep aligned context poses for the next independent SLAM window.
            # Later windows replace shared entries after they themselves have
            # been aligned, forming a stable chain across long videos.
            for row, global_frame in enumerate(all_row_globals):
                alignment_refs[int(global_frame)] = t4[row].clone()

            traj = _t4_to_traj(t4, traj)

            tstamp = np.asarray(data.get("tstamp", []), dtype=np.int64).reshape(-1)
            use_tstamp = tstamp.shape[0] == disps.shape[0] and tstamp.shape[0] > 0

            if traj_buf is None:
                traj_buf = np.zeros((0, traj.shape[1]), dtype=np.float32)

            row_globals = all_row_globals
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
        slam_valid=np.asarray(not slam_fallback_ranges),
        slam_fallback=np.asarray(
            "none" if not slam_fallback_ranges else "partial"
        ),
        slam_fallback_ranges=np.asarray(
            slam_fallback_ranges,
            dtype=np.int64,
        ).reshape(-1, 2),
        slam_fallback_modes=np.asarray(slam_fallback_modes),
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


def write_manifest(
    manifest_path: str,
    video_path: str,
    metas: List[ChunkMeta],
    *,
    window_context_frames: int = 0,
    temporal_block_frames: int = 1,
) -> None:
    data = {
        "video": video_path,
        "window_context_frames": window_context_frames,
        "temporal_block_frames": temporal_block_frames,
        "camera_space_merge": "rotation_aware_context_crossfade",
        "slam_merge": "multi_pose_rigid_alignment_and_overlap_blend",
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
        left_context_frames=args.window_context_frames,
        right_context_frames=args.window_context_frames,
        alignment_frames=args.temporal_block_frames,
    )
    proc_windows = processing_windows(
        windows,
        n_frames,
        left_context_frames=args.window_context_frames,
        right_context_frames=args.window_context_frames,
    )
    if not windows:
        raise RuntimeError(f"No frames decoded from {video_abs}")
    largest_process_window = max(end - start for start, end in proc_windows)
    log(
        f"[plan ] {n_frames} frames @ {args.target_fps}fps -> {len(windows)} windows "
        f"(context=+/-{args.window_context_frames}, "
        f"alignment={args.temporal_block_frames}, "
        f"largest={largest_process_window}, cap={args.max_chunk_frames})"
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
    write_manifest(
        manifest_path,
        video_abs,
        metas,
        window_context_frames=args.window_context_frames,
        temporal_block_frames=args.temporal_block_frames,
    )

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
        help="Hard cap on frames processed by one window, including temporal context (default: 3001)",
    )
    p.add_argument(
        "--window_context_frames",
        type=int,
        default=16,
        help=(
            "Real-frame context added on each side of an owned window. Context "
            "predictions are crossfaded at seams, while only the owned global "
            "timeline is published (default: 16)."
        ),
    )
    p.add_argument(
        "--temporal_block_frames",
        type=int,
        default=16,
        help=(
            "Align every interior owned boundary to this global temporal-model "
            "block size (default: 16)."
        ),
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
    if args.window_context_frames < 0:
        raise ValueError("--window_context_frames must be >= 0")
    if args.temporal_block_frames < 1:
        raise ValueError("--temporal_block_frames must be >= 1")

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
