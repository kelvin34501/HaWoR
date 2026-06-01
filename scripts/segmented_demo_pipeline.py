#!/usr/bin/env python3
"""Segment -> demo.py -> merge cam_space chunk results.

This script is designed for long videos that are expensive to process in one pass.
It splits input video(s) into chunks, runs demo.py on each chunk, and merges per-frame
camera-space parameters back into a single timeline.

Notes
- It merges camera-space params only (init_root_orient/init_hand_pose/init_trans/init_betas).
- No world-space alignment is performed.
- Frame timeline is recovered by cumulative extracted frame count per chunk.
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


RANGE_RE = re.compile(r"(\d+)_(\d+)(?:_50fps)?\.json$")
RANGE_NAME_RE = re.compile(r"^(\d+)_(\d+)((?:_50fps)?\.json)$")
IMG_NAME_RE = re.compile(r"^(\d+)(\.[^.]+)$")
FIELDS = ["init_root_orient", "init_hand_pose", "init_trans", "init_betas"]
SLAM_MAIN_RE = re.compile(r"^hawor_slam_w_scale_(\d+)_(\d+)\.npz$")


@dataclass
class ChunkMeta:
    chunk_path: str
    seq_dir: str
    offset: int
    frame_count: int


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


def ensure_ffmpeg_exists() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found in PATH")


def probe_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {proc.stderr.strip()}")
    try:
        return max(0.0, float(proc.stdout.strip()))
    except ValueError as e:
        raise RuntimeError(f"Invalid ffprobe duration output for {video_path}: {proc.stdout!r}") from e


def merge_last_two_chunks(chunks_dir: str, prev_chunk: str, last_chunk: str) -> None:
    concat_list = os.path.join(chunks_dir, "_concat_last_two.txt")
    tmp_merged = prev_chunk + ".tmpmerge.mp4"

    try:
        with open(concat_list, "w") as f:
            f.write(f"file '{os.path.abspath(prev_chunk)}'\n")
            f.write(f"file '{os.path.abspath(last_chunk)}'\n")

        cmd_copy = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list,
            "-c",
            "copy",
            tmp_merged,
        ]

        try:
            run_cmd(cmd_copy)
        except RuntimeError:
            cmd_reencode = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_list,
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "fast",
                "-c:a",
                "aac",
                tmp_merged,
            ]
            run_cmd(cmd_reencode)

        os.replace(tmp_merged, prev_chunk)
        os.remove(last_chunk)
    finally:
        if os.path.isfile(concat_list):
            os.remove(concat_list)
        if os.path.isfile(tmp_merged):
            os.remove(tmp_merged)


def enforce_min_last_chunk_duration(
    chunks_dir: str,
    chunks: List[str],
    min_last_segment_seconds: int,
) -> List[str]:
    if min_last_segment_seconds <= 0 or len(chunks) < 2:
        return chunks

    last_chunk = chunks[-1]
    last_duration = probe_video_duration(last_chunk)

    if last_duration >= float(min_last_segment_seconds):
        return chunks

    prev_chunk = chunks[-2]
    log(
        "[split] last chunk too short "
        f"({last_duration:.3f}s < {min_last_segment_seconds}s), merge into previous chunk"
    )
    merge_last_two_chunks(chunks_dir, prev_chunk, last_chunk)
    normalized = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.mp4")))
    if not normalized:
        raise RuntimeError(f"No chunks left after last-chunk normalization in {chunks_dir}")
    return normalized


def split_video(
    video_path: str,
    chunks_dir: str,
    segment_seconds: int,
    min_last_segment_seconds: int,
    reencode: bool,
    overwrite: bool,
) -> List[str]:
    os.makedirs(chunks_dir, exist_ok=True)

    if overwrite:
        for p in glob.glob(os.path.join(chunks_dir, "chunk_*.mp4")):
            os.remove(p)

    existing = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.mp4")))
    if existing:
        log(f"[split] Reuse existing chunks: {len(existing)}")
        return enforce_min_last_chunk_duration(
            chunks_dir=chunks_dir,
            chunks=existing,
            min_last_segment_seconds=min_last_segment_seconds,
        )

    out_pattern = os.path.join(chunks_dir, "chunk_%06d.mp4")
    cmd = ["ffmpeg", "-y", "-i", video_path]

    if reencode:
        cmd += [
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-c:a",
            "aac",
        ]
    else:
        cmd += ["-c", "copy"]

    cmd += [
        "-f",
        "segment",
        "-segment_time",
        str(segment_seconds),
        "-reset_timestamps",
        "1",
        out_pattern,
    ]

    log(f"[split] {' '.join(cmd)}")
    run_cmd(cmd)

    chunks = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.mp4")))
    if not chunks:
        raise RuntimeError(f"No chunks generated for {video_path}")
    return enforce_min_last_chunk_duration(
        chunks_dir=chunks_dir,
        chunks=chunks,
        min_last_segment_seconds=min_last_segment_seconds,
    )


def run_demo_on_chunk(
    chunk_path: str,
    project_dir: str,
    python_bin: str,
    vis_mode: str,
    gpu_id: Optional[int],
) -> str:
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [python_bin, "demo.py", "--video_path", chunk_path, "--vis_mode", vis_mode]
    log(f"[demo] {' '.join(cmd)}")
    run_cmd(cmd, cwd=project_dir, env=env)

    seq_dir = os.path.join(os.path.dirname(chunk_path), seq_name_for_video(chunk_path))
    if not os.path.isdir(seq_dir):
        raise RuntimeError(f"Chunk output seq dir missing: {seq_dir}")
    return seq_dir


def count_extracted_frames(seq_dir: str) -> int:
    img_dir = os.path.join(seq_dir, "extracted_images")
    if not os.path.isdir(img_dir):
        return 0
    patterns = ["*.jpg", "*.png", "*.jpeg"]
    count = 0
    for pat in patterns:
        count += len(glob.glob(os.path.join(img_dir, pat)))
    return count


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
                dst_name = shifted_json_name(fp, meta.offset)
                dst_path = os.path.join(hand_out_dir, dst_name)

                if os.path.exists(dst_path):
                    if overlap_policy == "keep_first":
                        continue
                    os.remove(dst_path)

                shutil.copy2(fp, dst_path)
                copied_count += 1

    if copied_count == 0:
        raise RuntimeError("No cam_space chunks found to merge")
    log(f"[merge] Copied {copied_count} cam_space json files into {out_dir}")


def _renumber_image_name(image_path: str, global_idx: int, local_idx: int) -> str:
    name = os.path.basename(image_path)
    m = IMG_NAME_RE.match(name)
    if m:
        width = max(6, len(m.group(1)))
        return f"{global_idx:0{width}d}{m.group(2)}"
    _, ext = os.path.splitext(name)
    return f"{global_idx:06d}_{local_idx:06d}{ext}"


def merge_extracted_images(
    chunk_metas: List[ChunkMeta],
    out_dir: str,
    overlap_policy: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    copied_count = 0
    patterns = ["*.jpg", "*.png", "*.jpeg"]

    for meta in chunk_metas:
        img_dir = os.path.join(meta.seq_dir, "extracted_images")
        if not os.path.isdir(img_dir):
            log(f"[merge] Skip chunk with no extracted_images: {meta.chunk_path}")
            continue

        image_files: List[str] = []
        for pat in patterns:
            image_files.extend(glob.glob(os.path.join(img_dir, pat)))
        image_files = sorted(image_files)
        if not image_files:
            continue

        for i, fp in enumerate(image_files):
            global_idx = meta.offset + i
            dst_name = _renumber_image_name(fp, global_idx, i)
            dst_path = os.path.join(out_dir, dst_name)

            if os.path.exists(dst_path):
                if overlap_policy == "keep_first":
                    continue
                os.remove(dst_path)

            shutil.copy2(fp, dst_path)
            copied_count += 1

    if copied_count == 0:
        log("[merge] No extracted_images chunks found to copy")
        return
    log(f"[merge] Copied {copied_count} extracted image files into {out_dir}")


def merge_slam(chunk_metas: List[ChunkMeta], out_dir: str, overlap_policy: str) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)

    traj_buf: Optional[np.ndarray] = None
    traj_valid = np.zeros((0,), dtype=bool)
    # Sparse keyframe storage for disps: global_frame_idx -> disps_frame (H, W[, C...]).
    disps_map: Dict[int, np.ndarray] = {}

    img_focal = None
    img_center = None
    # Weighted average by valid segment length when scale differs across chunks.
    scale_weighted_sum = 0.0
    scale_weight_sum = 0.0

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
        # 1) local absolute frame index in [local_start, local_end]
        # 2) local relative index in [0, local_end-local_start], shift by local_start
        if min_ts >= local_start and max_ts <= local_end:
            return local_tstamp
        local_span = local_end - local_start
        if min_ts >= 0 and max_ts <= local_span:
            return local_tstamp + local_start
        return local_tstamp

    for meta in chunk_metas:
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

            local_span = local_end - local_start + 1
            traj_len = min(traj.shape[0], local_span)
            if traj_len <= 0:
                continue

            tstamp = np.asarray(data.get("tstamp", []), dtype=np.int64).reshape(-1)
            use_tstamp = tstamp.shape[0] == disps.shape[0] and tstamp.shape[0] > 0

            if traj_buf is None:
                traj_buf = np.zeros((0, traj.shape[1]), dtype=np.float32)

            global_start = meta.offset + local_start
            global_end = global_start + traj_len
            _ensure_traj_capacity(global_end)

            if overlap_policy == "keep_first":
                dst_mask = traj_valid[global_start:global_end]
                write_idx = np.where(~dst_mask)[0]
                if write_idx.size > 0:
                    for j in write_idx:
                        traj_buf[global_start + j] = traj[j]
                    traj_valid[global_start:global_end][write_idx] = True
            else:
                traj_buf[global_start:global_end] = traj[:traj_len]
                traj_valid[global_start:global_end] = True

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
                ts_global = int(meta.offset + int(ts_local))
                if ts_global < 0:
                    continue
                if overlap_policy == "keep_first" and ts_global in disps_map:
                    continue
                disps_map[ts_global] = np.asarray(disps[i], dtype=np.float32)

            if img_focal is None and "img_focal" in data:
                img_focal = data["img_focal"]
            if img_center is None and "img_center" in data:
                img_center = data["img_center"]
            if "scale" in data:
                scale_weighted_sum += float(np.asarray(data["scale"]).reshape(-1)[0]) * traj_len
                scale_weight_sum += float(traj_len)

    if traj_buf is None or not traj_valid.any() or not disps_map:
        log("[merge] No SLAM chunks found to merge")
        return None

    last = int(np.where(traj_valid)[0][-1])
    traj_out = traj_buf[: last + 1]
    tstamp = np.asarray(sorted(disps_map.keys()), dtype=np.int32)
    disps_out = np.stack([disps_map[int(ts)] for ts in tstamp], axis=0).astype(np.float32)
    scale_out = np.float32(scale_weighted_sum / scale_weight_sum) if scale_weight_sum > 0 else np.float32(1.0)

    out_path = os.path.join(out_dir, f"hawor_slam_w_scale_0_{last}.npz")
    np.savez(
        out_path,
        tstamp=tstamp,
        traj=traj_out,
        disps=disps_out,
        img_focal=np.asarray(0.0 if img_focal is None else img_focal),
        img_center=np.asarray([0.0, 0.0] if img_center is None else img_center),
        scale=np.asarray(scale_out),
    )
    log(f"[merge] Saved {out_path} (frames=0..{last})")
    return out_path


def build_chunk_meta(chunks: List[str], seq_dirs: List[str]) -> List[ChunkMeta]:
    metas: List[ChunkMeta] = []
    offset = 0

    for chunk_path, seq_dir in zip(chunks, seq_dirs):
        frame_count = count_extracted_frames(seq_dir)
        if frame_count <= 0:
            cam_space_dir = os.path.join(seq_dir, "cam_space")
            frame_count = chunk_local_span(cam_space_dir)

        metas.append(
            ChunkMeta(
                chunk_path=chunk_path,
                seq_dir=seq_dir,
                offset=offset,
                frame_count=frame_count,
            )
        )
        offset += frame_count

    return metas


def write_manifest(manifest_path: str, video_path: str, metas: List[ChunkMeta]) -> None:
    data = {
        "video": video_path,
        "chunks": [
            {
                "chunk_path": m.chunk_path,
                "seq_dir": m.seq_dir,
                "offset": m.offset,
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

    chunks_dir = os.path.join(work_root, "chunks")
    merged_root = os.path.join(work_root, "merged")
    os.makedirs(work_root, exist_ok=True)

    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    log(f"[video] {video_abs}")
    log(f"[work ] {work_root}")

    if args.skip_split:
        chunks = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.mp4")))
        if not chunks:
            raise RuntimeError(f"No chunks found in {chunks_dir}; cannot --skip_split")
        log(f"[split] skip, found {len(chunks)} chunks")
    else:
        ensure_ffmpeg_exists()
        chunks = split_video(
            video_path=video_abs,
            chunks_dir=chunks_dir,
            segment_seconds=args.segment_seconds,
            min_last_segment_seconds=args.min_last_segment_seconds,
            reencode=args.reencode,
            overwrite=args.overwrite_chunks,
        )
        log(f"[split] generated {len(chunks)} chunks")

    seq_dirs: List[str] = []
    if args.skip_demo:
        for chunk in chunks:
            seq_dir = os.path.join(os.path.dirname(chunk), seq_name_for_video(chunk))
            if not os.path.isdir(seq_dir):
                raise RuntimeError(f"Missing seq_dir for chunk (cannot --skip_demo): {seq_dir}")
            seq_dirs.append(seq_dir)
        log("[demo ] skip")
    else:
        for i, chunk in enumerate(chunks):
            log(f"[demo ] chunk {i + 1}/{len(chunks)}")
            seq_dir = run_demo_on_chunk(
                chunk_path=chunk,
                project_dir=project_dir,
                python_bin=args.python_bin,
                vis_mode=args.vis_mode,
                gpu_id=args.gpu_id,
            )
            seq_dirs.append(seq_dir)

    if args.skip_merge:
        log("[merge] skip")
        return

    metas = build_chunk_meta(chunks, seq_dirs)
    manifest_path = os.path.join(work_root, "chunk_manifest.json")
    write_manifest(manifest_path, video_abs, metas)

    merged_cam_dir = os.path.join(merged_root, "cam_space")
    merge_cam_space(
        chunk_metas=metas,
        out_dir=merged_cam_dir,
        overlap_policy=args.overlap_policy,
    )

    merged_img_dir = os.path.join(merged_root, "extracted_images")
    merge_extracted_images(
        chunk_metas=metas,
        out_dir=merged_img_dir,
        overlap_policy=args.overlap_policy,
    )

    merged_slam_dir = os.path.join(merged_root, "SLAM")
    merged_slam_file = merge_slam(
        chunk_metas=metas,
        out_dir=merged_slam_dir,
        overlap_policy=args.overlap_policy,
    )

    # Also place merged result under the original video seq folder for downstream tools.
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

    orig_img_dir = os.path.join(orig_seq_dir, "extracted_images")
    os.makedirs(orig_img_dir, exist_ok=True)
    for fn in sorted(os.listdir(merged_img_dir)) if os.path.isdir(merged_img_dir) else []:
        src = os.path.join(merged_img_dir, fn)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(orig_img_dir, fn)
        if os.path.exists(dst):
            if args.overlap_policy == "keep_first":
                continue
            os.remove(dst)
        shutil.copy2(src, dst)
        log(f"[copy ] {dst}")

    log(f"[done ] merged cam_space available at: {target_dir}")


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
    p = argparse.ArgumentParser(description="Segment video(s), run demo.py per chunk, merge cam_space params")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video_path", type=str, help="Single input video path")
    src.add_argument("--video_dir", type=str, help="Recursively process all .mp4 in directory")

    p.add_argument("--segment_seconds", type=int, default=120, help="Chunk duration in seconds")
    p.add_argument(
        "--min_last_segment_seconds",
        type=int,
        default=10,
        help="Minimum allowed duration (seconds) for the final chunk; if shorter, merge into previous chunk",
    )
    p.add_argument("--reencode", action="store_true", help="Use re-encoding split for exact cut boundaries")
    p.add_argument("--overwrite_chunks", action="store_true", help="Delete existing chunk_*.mp4 before splitting")

    p.add_argument("--python_bin", type=str, default=sys.executable, help="Python executable for demo.py")
    p.add_argument("--vis_mode", type=str, default="off", help="demo.py --vis_mode value")
    p.add_argument("--gpu_id", type=int, default=None, help="Set CUDA_VISIBLE_DEVICES for demo.py")

    p.add_argument("--work_dir", type=str, default=None, help="Working directory (single-video mode recommended)")
    p.add_argument("--output_subdir", type=str, default="cam_space", help="Output subdir under original seq folder")
    p.add_argument("--overlap_policy", choices=["keep_last", "keep_first"], default="keep_last")

    p.add_argument("--skip_split", action="store_true")
    p.add_argument("--skip_demo", action="store_true")
    p.add_argument("--skip_merge", action="store_true")

    return p


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.min_last_segment_seconds < 0:
        raise ValueError("--min_last_segment_seconds must be >= 0")

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
