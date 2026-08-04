"""Interpolate HaWoR world-space predictions to a target FPS.

Example:
    /bin/python 'scripts/interpolation.py' --folder_path example/video_0
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from glob import glob

_IMPORT_ERROR = None
try:
    import joblib
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation, Slerp
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc


DISPS_U16_SCALE = 10000.0


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _interp_linear_with_valid(values, valid, t_old, t_new):
    """Linear interpolation for values shaped (T, C)."""
    valid_idx = np.where(valid)[0]
    out = np.zeros((len(t_new), values.shape[1]), dtype=np.float32)

    if len(valid_idx) == 0:
        return out
    if len(valid_idx) == 1:
        out[:] = values[valid_idx[0]][None, :]
        return out

    valid_times = t_old[valid_idx]
    for c in range(values.shape[1]):
        out[:, c] = np.interp(t_new, valid_times, values[valid_idx, c]).astype(np.float32)
    return out


def _interp_linear(values, t_old, t_new):
    """Linear interpolation for values shaped (T, ...)."""
    t_old = np.asarray(t_old, dtype=np.float64)
    t_new = np.asarray(t_new, dtype=np.float64)
    values = np.asarray(values)

    # Ensure interpolation axis is sorted and strictly increasing.
    sort_idx = np.argsort(t_old)
    t_old = t_old[sort_idx]
    values = values[sort_idx]

    if len(t_old) > 1:
        keep = np.concatenate(([True], np.diff(t_old) > 1e-9))
        t_old = t_old[keep]
        values = values[keep]

    if values.shape[0] == 1:
        out_shape = (len(t_new),) + values.shape[1:]
        return np.repeat(values, len(t_new), axis=0).reshape(out_shape)

    flat = values.reshape(values.shape[0], -1)
    out = np.zeros((len(t_new), flat.shape[1]), dtype=np.float32)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(t_new, t_old, flat[:, c]).astype(np.float32)
    return out.reshape((len(t_new),) + values.shape[1:])


def _slerp_rotvec_with_valid(rotvec, valid, t_old, t_new):
    """SLERP for rotation vectors shaped (T, 3)."""
    valid_idx = np.where(valid)[0]
    out = np.zeros((len(t_new), 3), dtype=np.float32)

    if len(valid_idx) == 0:
        return out
    if len(valid_idx) == 1:
        out[:] = rotvec[valid_idx[0]][None, :]
        return out

    valid_times = t_old[valid_idx].astype(np.float64)
    t_min = float(valid_times[0])
    t_max = float(valid_times[-1])

    out[t_new <= t_min] = rotvec[valid_idx[0]][None, :]
    out[t_new >= t_max] = rotvec[valid_idx[-1]][None, :]

    inside = (t_new > t_min) & (t_new < t_max)
    if np.any(inside):
        rots = Rotation.from_rotvec(rotvec[valid_idx])
        slerp = Slerp(valid_times, rots)
        out[inside] = slerp(t_new[inside].astype(np.float64)).as_rotvec().astype(np.float32)

    return out


def _slerp_matrices(mats, t_old, t_new):
    """SLERP for rotation matrices shaped (T, 3, 3)."""
    mats = np.asarray(mats)
    t_old = np.asarray(t_old, dtype=np.float64)
    t_new = np.asarray(t_new, dtype=np.float64)

    if mats.shape[0] == 1:
        return np.repeat(mats, len(t_new), axis=0).astype(np.float32)

    t_min = float(t_old[0])
    t_max = float(t_old[-1])
    out = np.zeros((len(t_new), 3, 3), dtype=np.float32)

    rots = Rotation.from_matrix(mats)
    slerp = Slerp(t_old, rots)

    low = t_new <= t_min
    high = t_new >= t_max
    mid = ~(low | high)

    if np.any(low):
        out[low] = mats[0]
    if np.any(high):
        out[high] = mats[-1]
    if np.any(mid):
        out[mid] = slerp(t_new[mid]).as_matrix().astype(np.float32)

    return out


def _parse_chunk_range(chunk_name):
    base = os.path.splitext(os.path.basename(chunk_name))[0]
    start_str, end_str = base.split("_")
    return int(start_str), int(end_str)


def interpolate_cam_space(folder_path, src_frame_count, dst_frame_count, output_dir_name="cam_space_50fps"):
    src_cam_space_dir = os.path.join(folder_path, "cam_space")
    if not os.path.isdir(src_cam_space_dir):
        print("Skip cam_space interpolation: source cam_space folder not found")
        return

    dst_cam_space_dir = os.path.join(folder_path, output_dir_name)
    os.makedirs(dst_cam_space_dir, exist_ok=True)

    t_new_global = np.linspace(0, src_frame_count - 1, dst_frame_count, dtype=np.float64)
    total_out_chunks = 0

    for hand_dir in sorted(glob(os.path.join(src_cam_space_dir, "*"))):
        if not os.path.isdir(hand_dir):
            continue

        hand_name = os.path.basename(hand_dir)
        chunk_files = sorted(glob(os.path.join(hand_dir, "*.json")))
        if not chunk_files:
            continue

        hand_out_dir = os.path.join(dst_cam_space_dir, hand_name)
        os.makedirs(hand_out_dir, exist_ok=True)

        for chunk_file in chunk_files:
            start, end = _parse_chunk_range(chunk_file)
            new_idx = np.where((t_new_global >= start) & (t_new_global <= end))[0]
            if len(new_idx) == 0:
                continue

            with open(chunk_file, "r") as f:
                src = json.load(f)

            root = np.asarray(src["init_root_orient"], dtype=np.float32)      # (B, T, 3, 3)
            pose = np.asarray(src["init_hand_pose"], dtype=np.float32)        # (B, T, 15, 3, 3)
            trans = np.asarray(src["init_trans"], dtype=np.float32)           # (B, T, 3)
            betas = np.asarray(src["init_betas"], dtype=np.float32)           # (B, T, 10)

            bsz, t_chunk = trans.shape[:2]
            t_old_local = np.linspace(start, end, t_chunk, dtype=np.float64)
            t_new_local = t_new_global[new_idx]

            out_trans = np.zeros((bsz, len(t_new_local), 3), dtype=np.float32)
            out_betas = np.zeros((bsz, len(t_new_local), betas.shape[-1]), dtype=np.float32)
            out_root = np.zeros((bsz, len(t_new_local), 3, 3), dtype=np.float32)
            out_pose = np.zeros((bsz, len(t_new_local), pose.shape[2], 3, 3), dtype=np.float32)

            for b in range(bsz):
                out_trans[b] = _interp_linear(trans[b], t_old_local, t_new_local)
                out_betas[b] = _interp_linear(betas[b], t_old_local, t_new_local)
                out_root[b] = _slerp_matrices(root[b], t_old_local, t_new_local)
                for j in range(pose.shape[2]):
                    out_pose[b, :, j] = _slerp_matrices(pose[b, :, j], t_old_local, t_new_local)

            out_data = {
                "init_root_orient": out_root.tolist(),
                "init_hand_pose": out_pose.tolist(),
                "init_trans": out_trans.tolist(),
                "init_betas": out_betas.tolist(),
            }

            out_chunk_name = f"{int(new_idx[0])}_{int(new_idx[-1])}_50fps.json"
            out_chunk_path = os.path.join(hand_out_dir, out_chunk_name)
            with open(out_chunk_path, "w") as f:
                json.dump(out_data, f, indent=1)
            total_out_chunks += 1

    print(f"Saved interpolated cam_space chunks to: {dst_cam_space_dir} (files: {total_out_chunks})")


def _normalize_quat_xyzw(quat):
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return quat / norm


def _interpolated_slam_metadata(data, t_new):
    """Carry raw-SLAM merge metadata onto the interpolated timeline."""
    meta = {}
    for key in ("slam_valid", "slam_fallback", "slam_fallback_modes"):
        if key in data:
            meta[key] = data[key]
    if "slam_fallback_ranges" in data:
        mapped_ranges = []
        ranges = np.asarray(data["slam_fallback_ranges"], dtype=np.int64).reshape(-1, 2)
        for start, end in ranges:
            mapped_start = int(np.searchsorted(t_new, float(start), side="left"))
            mapped_end = int(np.searchsorted(t_new, float(end), side="left"))
            mapped_start = max(0, min(mapped_start, len(t_new) - 1))
            mapped_end = max(mapped_start + 1, min(mapped_end, len(t_new)))
            mapped_ranges.append((mapped_start, mapped_end))
        meta["slam_fallback_ranges"] = np.asarray(
            mapped_ranges,
            dtype=np.int64,
        ).reshape(-1, 2)
    if "overlap_aligned" in data:
        meta["overlap_aligned"] = data["overlap_aligned"]
    if "window_starts" in data:
        starts = sorted(int(x) for x in np.asarray(data["window_starts"]).reshape(-1))
        mapped = []
        for start in starts:
            idx = int(np.searchsorted(t_new, float(start), side="left"))
            idx = max(0, min(idx, len(t_new) - 1))
            mapped.append(idx)
        meta["window_starts"] = np.asarray(sorted(set(mapped)), dtype=np.int64)
        meta["source_window_starts"] = np.asarray(starts, dtype=np.int64)
    return meta


def interpolate_slam_artifacts(folder_path, src_frame_count, dst_frame_count):
    slam_dir = os.path.join(folder_path, "SLAM")
    if not os.path.isdir(slam_dir):
        print("Skip SLAM interpolation: SLAM folder not found")
        return []

    slam_candidates = sorted(glob(os.path.join(slam_dir, "hawor_slam_w_scale_*.npz")))
    slam_files = []
    for path in slam_candidates:
        name = os.path.basename(path)
        if "_disps_" in name or name.endswith("_50fps.npz"):
            continue
        if re.match(r"^hawor_slam_w_scale_\d+_\d+\.npz$", name):
            slam_files.append(path)

    if not slam_files:
        print("Skip SLAM interpolation: no matching hawor_slam_w_scale_*.npz found")
        return []

    t_new = np.linspace(0, src_frame_count - 1, dst_frame_count, dtype=np.float64)
    out_count = 0
    out_disps_files = []

    for slam_file in slam_files:
        data = dict(np.load(slam_file, allow_pickle=True))
        traj = np.asarray(data["traj"], dtype=np.float32)
        tstamp = np.asarray(data["tstamp"], dtype=np.int32)
        disps = np.asarray(data["disps"], dtype=np.float32)

        if traj.shape[0] == len(tstamp) and len(tstamp) >= 2:
            t_old = tstamp.astype(np.float64)
        else:
            t_old = np.linspace(0, src_frame_count - 1, traj.shape[0], dtype=np.float64)

        if disps.shape[0] == len(tstamp) and len(tstamp) >= 2:
            t_disp_old = tstamp.astype(np.float64)
        else:
            t_disp_old = np.linspace(0, src_frame_count - 1, disps.shape[0], dtype=np.float64)

        trans_old = traj[:, :3]
        quat_old = _normalize_quat_xyzw(traj[:, 3:7])  # qx,qy,qz,qw

        trans_new = _interp_linear(trans_old, t_old, t_new)
        disps_new = _interp_linear(disps, t_disp_old, t_new).astype(np.float32)

        if len(t_old) == 1:
            quat_new = np.repeat(quat_old, len(t_new), axis=0)
        else:
            t_min = float(t_old[0])
            t_max = float(t_old[-1])
            quat_new = np.zeros((len(t_new), 4), dtype=np.float32)

            low = t_new <= t_min
            high = t_new >= t_max
            mid = ~(low | high)
            quat_new[low] = quat_old[0]
            quat_new[high] = quat_old[-1]
            if np.any(mid):
                slerp = Slerp(t_old, Rotation.from_quat(quat_old))
                quat_new[mid] = slerp(t_new[mid]).as_quat().astype(np.float32)

        traj_new = np.concatenate([trans_new.astype(np.float32), quat_new.astype(np.float32)], axis=1)
        tstamp_new = np.arange(dst_frame_count, dtype=np.int32)

        base_name = os.path.basename(slam_file)
        base_name = os.path.splitext(base_name)[0]
        base_name = re.sub(r'_\d+_\d+$', '', base_name)
        out_file = os.path.join(os.path.dirname(slam_file), f"{base_name}_0_{dst_frame_count}_50fps.npz")
        out_disps_file = os.path.join(
            os.path.dirname(slam_file),
            f"{base_name}_disps_0_{dst_frame_count}_50fps.npz",
        )
        np.savez(
            out_file,
            tstamp=tstamp_new,
            traj=traj_new,
            img_focal=data["img_focal"],
            img_center=data["img_center"],
            scale=data["scale"],
            **_interpolated_slam_metadata(data, t_new),
        )
        np.savez(out_disps_file, disps=disps_new)
        out_disps_files.append(out_disps_file)
        out_count += 1

    print(f"Saved interpolated SLAM artifacts: {out_count} file(s) in {slam_dir}")
    print(f"Saved disps-only artifacts: {len(out_disps_files)} file(s) in {slam_dir}")
    return out_disps_files


def _collect_disps_npz_files(folder_path):
    slam_dir = os.path.join(folder_path, "SLAM")
    if not os.path.isdir(slam_dir):
        return []
    return sorted(glob(os.path.join(slam_dir, "*_disps_*_50fps.npz")))


def _disps_video_path(disps_npz_path):
    return os.path.splitext(disps_npz_path)[0] + "_uint16.mkv"


def _quantize_disps_frame(frame):
    """Apply the legacy disparity quantization to one frame."""
    disp = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(
        np.clip(
            np.round(disp * DISPS_U16_SCALE),
            0.0,
            65535.0,
        ).astype(np.uint16)
    )


def _write_all(stream, data):
    """Write a contiguous array completely, including across partial writes."""
    remaining = memoryview(data).cast("B")
    while remaining:
        written = stream.write(remaining)
        if not written:
            raise BrokenPipeError("ffmpeg stdin closed before the frame was written")
        remaining = remaining[written:]


def disps_npz_to_uint16_video(disps_npz_path, fps=50, overwrite=False):
    out_video = _disps_video_path(disps_npz_path)
    if (not overwrite) and os.path.exists(out_video):
        print(f"Skip disps video: already exists -> {out_video}")
        return out_video

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        print("Skip disps video: ffmpeg not found in PATH")
        return None

    with np.load(disps_npz_path, allow_pickle=True) as data:
        disps = np.asarray(data["disps"], dtype=np.float32)
    if disps.ndim != 3 or disps.shape[0] < 1:
        print(f"Skip disps video: invalid disps shape in {disps_npz_path} -> {disps.shape}")
        return None

    num_frames, height, width = disps.shape
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray16le",
        "-s",
        f"{width}x{height}",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "ffv1",
        out_video,
    ]

    # Keep the exact legacy quantization and frame order, but avoid retaining
    # full-timeline float, uint16, and bytes copies at the same time.
    with (
        tempfile.TemporaryFile() as stdout_file,
        tempfile.TemporaryFile() as stderr_file,
    ):
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=stdout_file,
            stderr=stderr_file,
        )
        try:
            for frame in disps:
                _write_all(proc.stdin, _quantize_disps_frame(frame))
            proc.stdin.close()
            proc.wait()
        except Exception:
            proc.kill()
            proc.wait()
            raise

        if proc.returncode != 0:
            stderr_file.seek(0)
            err_msg = stderr_file.read().decode("utf-8", errors="ignore")
            print(f"Skip disps video: ffmpeg failed for {disps_npz_path}")
            print(err_msg)
            return None

    print(f"Saved uint16 disps video ({num_frames} frames, S={DISPS_U16_SCALE:g}): {out_video}")
    return out_video


def _count_images(folder):
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    files = []
    for pattern in patterns:
        files.extend(glob(os.path.join(folder, pattern)))
    return len(files)
def _is_world_space_done(out_path, dst_frame_count):
    if not os.path.exists(out_path):
        return False
    try:
        data = joblib.load(out_path)
        if not isinstance(data, (list, tuple)) or len(data) < 1:
            return False
        return _to_numpy(data[0]).shape[1] == dst_frame_count
    except Exception:
        return False


def _is_cam_space_done(folder_path, output_dir_name, dst_frame_count):
    dst_cam_space_dir = os.path.join(folder_path, output_dir_name)
    if not os.path.isdir(dst_cam_space_dir):
        return False

    hand_dirs = [d for d in glob(os.path.join(dst_cam_space_dir, "*")) if os.path.isdir(d)]
    if not hand_dirs:
        return False

    found_valid_file = False
    for hand_dir in hand_dirs:
        chunk_files = sorted(glob(os.path.join(hand_dir, "*.json")))
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, "r") as f:
                    data = json.load(f)
                trans = np.asarray(data["init_trans"], dtype=np.float32)  # (B, T, 3)
                if trans.ndim == 3 and trans.shape[1] >= 1:
                    found_valid_file = True
                    # 这里只检查文件是有效插值产物，不强制每个 chunk 长度等于 dst_frame_count
                    # 因为 cam_space 是分 chunk 保存的
                    if not chunk_file.endswith("_50fps.json"):
                        return False
            except Exception:
                return False

    return found_valid_file


def _is_slam_done(folder_path, dst_frame_count):
    slam_dir = os.path.join(folder_path, "SLAM")
    if not os.path.isdir(slam_dir):
        return True  # 没有 SLAM 目录时，本来就会 skip

    slam_files = []
    for path in sorted(glob(os.path.join(slam_dir, "*_50fps.npz"))):
        name = os.path.basename(path)
        if "_disps_" in name:
            continue
        if re.match(r"^hawor_slam_w_scale_\d+_\d+_50fps\.npz$", name):
            slam_files.append(path)

    if not slam_files:
        return False

    found_valid_file = False
    for slam_file in slam_files:
        try:
            data = np.load(slam_file, allow_pickle=True)
            tstamp = np.asarray(data["tstamp"])
            traj = np.asarray(data["traj"])
            if len(tstamp) != dst_frame_count:
                return False
            if traj.shape[0] != dst_frame_count:
                return False
            found_valid_file = True
        except Exception:
            return False

    return found_valid_file


def _is_disps_done(folder_path, dst_frame_count):
    slam_dir = os.path.join(folder_path, "SLAM")
    if not os.path.isdir(slam_dir):
        return True

    disps_files = _collect_disps_npz_files(folder_path)
    if not disps_files:
        return False

    found_valid_file = False
    for disps_file in disps_files:
        try:
            data = np.load(disps_file, allow_pickle=True)
            disps = np.asarray(data["disps"])
            if disps.ndim != 3:
                return False
            if disps.shape[0] != dst_frame_count:
                return False
            found_valid_file = True
        except Exception:
            return False
    return found_valid_file


def _is_disps_video_done(folder_path):
    slam_dir = os.path.join(folder_path, "SLAM")
    if not os.path.isdir(slam_dir):
        return True

    disps_files = _collect_disps_npz_files(folder_path)
    if not disps_files:
        return False
    for disps_file in disps_files:
        if not os.path.exists(_disps_video_path(disps_file)):
            return False
    return True


def interpolate_world_space(world_data, src_frame_count, dst_frame_count):
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = world_data

    pred_trans = _to_numpy(pred_trans).astype(np.float32)
    pred_rot = _to_numpy(pred_rot).astype(np.float32)
    pred_hand_pose = _to_numpy(pred_hand_pose).astype(np.float32)
    pred_betas = _to_numpy(pred_betas).astype(np.float32)
    pred_valid = _to_numpy(pred_valid)

    pred_valid_bool = pred_valid > 0.5

    if pred_trans.ndim != 3 or pred_rot.ndim != 3 or pred_hand_pose.ndim != 3:
        raise ValueError("Unexpected tensor shape in world_space_res.pth")

    n_hands, t_data, _ = pred_trans.shape
    if t_data < 2:
        raise ValueError("At least 2 frames are required for FPS interpolation")
    if src_frame_count < 2 or dst_frame_count < 2:
        raise ValueError("Both source and target image folders must contain at least 2 frames")

    # Use extraction-time frame counts as the timeline definition.
    t_old = np.linspace(0, src_frame_count - 1, t_data, dtype=np.float64)
    t_new = np.linspace(0, src_frame_count - 1, dst_frame_count, dtype=np.float64)

    trans_out = np.zeros((n_hands, dst_frame_count, 3), dtype=np.float32)
    rot_out = np.zeros((n_hands, dst_frame_count, 3), dtype=np.float32)
    betas_out = np.zeros((n_hands, dst_frame_count, pred_betas.shape[-1]), dtype=np.float32)

    num_joints = pred_hand_pose.shape[-1] // 3
    hand_pose_in = pred_hand_pose.reshape(n_hands, t_data, num_joints, 3)
    hand_pose_slerp = np.zeros((n_hands, dst_frame_count, num_joints, 3), dtype=np.float32)

    for h in range(n_hands):
        valid_h = pred_valid_bool[h]
        trans_out[h] = _interp_linear_with_valid(pred_trans[h], valid_h, t_old, t_new)
        betas_out[h] = _interp_linear_with_valid(pred_betas[h], valid_h, t_old, t_new)
        rot_out[h] = _slerp_rotvec_with_valid(pred_rot[h], valid_h, t_old, t_new)

        for j in range(num_joints):
            hand_pose_slerp[h, :, j] = _slerp_rotvec_with_valid(hand_pose_in[h, :, j], valid_h, t_old, t_new)

    hand_pose_out = hand_pose_slerp.reshape(n_hands, dst_frame_count, num_joints * 3)

    nearest_old_idx = np.argmin(np.abs(t_old[None, :] - t_new[:, None]), axis=1)
    valid_out = pred_valid_bool[:, nearest_old_idx].astype(np.float32)

    return [
        torch.from_numpy(trans_out),
        torch.from_numpy(rot_out),
        torch.from_numpy(hand_pose_out),
        torch.from_numpy(betas_out),
        torch.from_numpy(valid_out),
    ]


def main():
    parser = argparse.ArgumentParser(description="Interpolation")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the input reconstruction folder")
    parser.add_argument(
        "--source_images_dir",
        type=str,
        default="extracted_images",
        help="Source frame folder name under folder_path",
    )
    parser.add_argument(
        "--target_images_dir",
        type=str,
        default="extracted_images_50fps",
        help="Target frame folder name under folder_path (legacy; ignored when --video_path is given)",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Source video. When provided, source/target frame counts are derived by "
             "decoding the video at --source_fps/--target_fps instead of counting JPEGs on disk.",
    )
    parser.add_argument("--source_fps", type=float, default=30, help="Source timeline fps (matches reconstruction fps)")
    parser.add_argument("--target_fps", type=float, default=50, help="Target (interpolated) timeline fps")
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output file name under folder_path. Defaults to world_space_res_interp_by_image_count.pth",
    )
    args = parser.parse_args()

    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "Missing runtime dependency for interpolation. "
            "Install project requirements first (numpy, scipy, torch, joblib)."
        ) from _IMPORT_ERROR

    in_path = os.path.join(args.folder_path, "world_space_res.pth")
    has_world = os.path.exists(in_path)

    if args.video_path:
        # Derive frame counts by decoding the video on demand (no JPEGs on disk).
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from lib.pipeline.frame_source import FrameSource
        src_frame_count = len(FrameSource(args.video_path, target_fps=args.source_fps))
        dst_frame_count = len(FrameSource(args.video_path, target_fps=args.target_fps))
    else:
        # Legacy path: count extracted JPEGs.
        source_dir = os.path.join(args.folder_path, args.source_images_dir)
        target_dir = os.path.join(args.folder_path, args.target_images_dir)
        if not os.path.isdir(source_dir):
            raise FileNotFoundError(f"Source image folder not found: {source_dir}")
        if not os.path.isdir(target_dir):
            raise FileNotFoundError(f"Target image folder not found: {target_dir}")
        src_frame_count = _count_images(source_dir)
        dst_frame_count = _count_images(target_dir)

    out_name = args.output_name or "world_space_res_50fps.pth"
    out_path = os.path.join(args.folder_path, out_name)

    # world_space_res.pth is optional: only interpolate it when the infiller
    # produced one. When absent, treat world-space as "done" so it never blocks
    # the cam_space/SLAM/disps interpolation that always runs.
    world_done = (not has_world) or _is_world_space_done(out_path, dst_frame_count)
    cam_done = _is_cam_space_done(args.folder_path, "cam_space_50fps", dst_frame_count)
    slam_done = _is_slam_done(args.folder_path, dst_frame_count)
    disps_done = _is_disps_done(args.folder_path, dst_frame_count)
    disps_video_done = _is_disps_video_done(args.folder_path)

    print(f"Image count (timeline): {src_frame_count} -> {dst_frame_count}")
    print(f"Check existing outputs:")
    print(f"  world_space: {'done' if world_done else 'missing/incomplete'}"
          f"{'' if has_world else ' (no world_space_res.pth; skipped)'}")
    print(f"  cam_space:   {'done' if cam_done else 'missing/incomplete'}")
    print(f"  SLAM:        {'done' if slam_done else 'missing/incomplete'}")
    print(f"  disps_npz:   {'done' if disps_done else 'missing/incomplete'}")
    print(f"  disps_video: {'done' if disps_video_done else 'missing/incomplete'}")

    if world_done and cam_done and slam_done and disps_done and disps_video_done:
        print("All interpolation outputs already exist and match target frame count. Skip.")
        return

    if has_world and not world_done:
        world_data = joblib.load(in_path)
        old_len = _to_numpy(world_data[0]).shape[1]
        interp_data = interpolate_world_space(world_data, src_frame_count, dst_frame_count)
        joblib.dump(interp_data, out_path)
        new_len = _to_numpy(interp_data[0]).shape[1]
        print(f"Saved interpolated world-space result to: {out_path}")
        print(f"World data frame count: {old_len} -> {new_len}")
    elif has_world:
        print(f"Skip world-space interpolation: already done -> {out_path}")

    if not cam_done:
        interpolate_cam_space(args.folder_path, src_frame_count, dst_frame_count)
    else:
        print("Skip cam_space interpolation: already done")

    if (not slam_done) or (not disps_done):
        disps_files = interpolate_slam_artifacts(args.folder_path, src_frame_count, dst_frame_count)
    else:
        print("Skip SLAM interpolation: already done")
        disps_files = _collect_disps_npz_files(args.folder_path)

    if not disps_done and not disps_files:
        disps_files = _collect_disps_npz_files(args.folder_path)

    if not disps_files:
        print("Skip disps video: no disps npz found")
        return

    for disps_file in disps_files:
        disps_npz_to_uint16_video(disps_file, fps=args.target_fps, overwrite=False)


if __name__ == "__main__":
    main()
