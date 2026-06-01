import os
import json
import argparse
import sys
import re
from glob import glob

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from pathlib import Path

import numpy as np
import cv2
import torch

from natsort import natsorted
from collections import defaultdict

from lib.models.mano_wrapper import MANO
from lib.pipeline.masked_droid_slam import est_calib




# python scripts/project_hands_to_images.py \
#   --seq example/24fps \
#   --tracks 0,1 \
#   --left 0 \
#   --combine \
#   --draw mesh \
#   --out example/24fps/overlays/cam_proj

def np_squeeze_arr(arr, target_shape=None):
    x = np.array(arr)
    x = np.squeeze(x)
    if target_shape is not None and x.shape != target_shape:
        raise ValueError(f"Unexpected shape {x.shape}, expected {target_shape}")
    return x


def load_cam_space_chunk(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    R_root = np_squeeze_arr(data["init_root_orient"])  # expect (T, 3, 3)
    R_hand = np_squeeze_arr(data["init_hand_pose"])  # expect (T, 15, 3, 3)
    t_root = np_squeeze_arr(data["init_trans"])  # expect (T, 3)
    betas = np_squeeze_arr(data["init_betas"])  # expect (10,) or (T,10)

    # Normalize shapes
    if R_root.ndim == 2 and R_root.shape == (3, 3):
        R_root = R_root[None, ...]
    if R_hand.ndim == 3 and R_hand.shape == (15, 3, 3):
        R_hand = R_hand[None, ...]
    if t_root.ndim == 1 and t_root.shape == (3,):
        t_root = t_root[None, ...]
    if betas.ndim == 1:
        betas = betas[None, ...]  # (1, 10)

    T = R_root.shape[0]
    # Tile betas over time if needed
    if betas.shape[0] == 1:
        betas = np.repeat(betas, T, axis=0)

    return R_root, R_hand, t_root, betas


def mano_forward_rotmat(R_root, R_hand, t_root, betas, is_left=False, device="cpu", fix_shapedirs=True):
    # Create MANO layer
    if is_left:
        mano_cfg = {
            'data_dir': '_DATA/data_left/',
            'model_path': '_DATA/data_left/mano_left',
            'gender': 'neutral',
            'num_hand_joints': 15,
            'create_body_pose': False,
            'is_rhand': False,
        }
    else:
        mano_cfg = {
            'data_dir': '_DATA/data/',
            'model_path': '_DATA/data/mano',
            'gender': 'neutral',
            'num_hand_joints': 15,
            'create_body_pose': False,
        }

    mano = MANO(**mano_cfg)
    mano = mano.to(device)

    # Align with project utils: fix MANO left shapedirs bug
    if is_left and fix_shapedirs and hasattr(mano, 'shapedirs'):
        # https://github.com/vchoutas/smplx/issues/48
        with torch.no_grad():
            mano.shapedirs[:, 0, :] *= -1

    T = R_root.shape[0]
    NUM = 15

    # Convert numpy to torch tensors with expected shapes for pose2rot=False
    global_orient = torch.from_numpy(R_root).float().to(device).unsqueeze(1)  # (T,1,3,3)
    hand_pose = torch.from_numpy(R_hand).float().to(device)  # (T,15,3,3)
    transl = torch.from_numpy(t_root).float().to(device)  # (T,3)
    betas_t = torch.from_numpy(betas).float().to(device)  # (T,10)

    with torch.no_grad():
        out = mano(global_orient=global_orient, hand_pose=hand_pose, betas=betas_t, transl=transl, pose2rot=False)

    # joints: (T, J, 3), vertices: (T, V, 3)
    joints = out.joints.detach().cpu().numpy()
    verts = out.vertices.detach().cpu().numpy()

    # Augmented faces consistent with pipeline utils
    faces_right = mano.faces.copy()
    faces_new = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279], [122, 118, 279],
                          [279, 118, 215], [118, 117, 215], [215, 117, 214], [117, 119, 214], [214, 119, 121],
                          [119, 120, 121], [121, 120, 78], [120, 108, 78], [78, 108, 79]],
                         dtype=np.int32)
    faces_right = np.concatenate([faces_right, faces_new], axis=0)
    faces = faces_right[:, [0, 2, 1]] if is_left else faces_right

    # Unique edges for wireframe
    edges_set = set()
    for tri in faces:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        edges_set.add((min(i, j), max(i, j)))
        edges_set.add((min(j, k), max(j, k)))
        edges_set.add((min(k, i), max(k, i)))
    edges = np.array(sorted(list(edges_set)), dtype=np.int32)

    return joints, verts, faces, edges


def project_points_cam(points, fx, fy, cx, cy):
    # points: (N,3) in camera coordinates
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    eps = 1e-6
    Zc = np.clip(Z, eps, None)
    u = fx * (X / Zc) + cx
    v = fy * (Y / Zc) + cy
    return np.stack([u, v], axis=-1), Z


def draw_openpose_skeleton(img, kpts2d, color=(0, 255, 0)):
    # OpenPose hand 21 keypoints connections
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),  # thumb
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),  # index
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),  # middle
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),  # ring
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20)  # pinky
    ]
    for i, (x, y) in enumerate(kpts2d):
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 255), -1)
    for a, b in edges:
        xa, ya = kpts2d[a]
        xb, yb = kpts2d[b]
        if np.isfinite(xa) and np.isfinite(xb) and np.isfinite(ya) and np.isfinite(yb):
            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), color, 1)
    return img


def draw_mesh_wireframe(img, verts2d, edges, color=(0, 200, 255)):
    for (a, b) in edges:
        xa, ya = verts2d[a]
        xb, yb = verts2d[b]
        if (np.isfinite(xa) and np.isfinite(ya) and np.isfinite(xb) and np.isfinite(yb)):
            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), color, 1)
    return img


def parse_chunk_range(chunk_path):
    stem = Path(chunk_path).stem
    match = re.match(r'^(\d+)_(\d+)(?:_.*)?$', stem)
    if not match:
        raise ValueError(f"Unexpected chunk name: {stem}")
    return int(match.group(1)), int(match.group(2))


def default_output_root(seq, images_subdir, cam_space_subdir):
    suffix = ''
    if cam_space_subdir != 'cam_space':
        if cam_space_subdir.startswith('cam_space'):
            suffix = cam_space_subdir[len('cam_space'):]
        else:
            suffix = f'_{cam_space_subdir}'
    elif images_subdir != 'extracted_images':
        if images_subdir.startswith('extracted_images'):
            suffix = images_subdir[len('extracted_images'):]
        else:
            suffix = f'_{images_subdir}'
    return seq / 'overlays' / f'cam_proj{suffix}'


def main():
    parser = argparse.ArgumentParser(description="Project cam_space MANO poses back to images")
    parser.add_argument('--seq', required=True, help='Sequence folder, e.g., example/24fps')
    parser.add_argument('--out', default=None, help='Output overlay folder; default: <seq>/overlays/cam_proj')
    parser.add_argument('--images_subdir',
                        default='extracted_images',
                        help='Image subdir under seq, e.g., extracted_images or extracted_images_50fps')
    parser.add_argument('--cam_space_subdir',
                        default='cam_space',
                        help='cam_space subdir under seq, e.g., cam_space or cam_space_50fps')
    parser.add_argument('--tracks',
                        default=None,
                        help='Comma-separated track ids to process, e.g., "0,1"; default: all')
    parser.add_argument('--left', default=None, help='Comma-separated track ids to treat as left hand (MANO left)')
    parser.add_argument('--draw',
                        default='joints',
                        choices=['joints', 'mesh', 'both'],
                        help='What to draw: joints, mesh, or both')
    parser.add_argument('--combine', action='store_true', help='Draw all selected tracks on the same image')
    args = parser.parse_args()

    seq = Path(args.seq)
    img_dir = seq / args.images_subdir
    if not img_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {img_dir}")

    imgfiles = natsorted(glob(str(img_dir / '*.jpg'))) or natsorted(glob(str(img_dir / '*.png')))
    if len(imgfiles) == 0:
        raise FileNotFoundError(f"No images found in {img_dir}")

    # Intrinsics: focal from est_focal.txt if present; principal point from image center
    focal = None
    est_focal_path = seq / 'est_focal.txt'
    if est_focal_path.exists():
        try:
            focal = float(est_focal_path.read_text().strip())
        except Exception:
            focal = None
    if focal is None:
        focal = 600.0

    calib = np.array(est_calib(imgfiles))  # [f, f, cx, cy] with cx,cy from first image center
    calib[:2] = focal
    fx, fy, cx, cy = calib[:4]

    out_root = Path(args.out) if args.out else default_output_root(seq, args.images_subdir, args.cam_space_subdir)
    out_root.mkdir(parents=True, exist_ok=True)

    cam_space = seq / args.cam_space_subdir
    if not cam_space.exists():
        raise FileNotFoundError(f"cam_space folder not found: {cam_space}")

    track_ids = []
    if args.tracks is None:
        for d in sorted(os.listdir(cam_space)):
            if d.isdigit():
                track_ids.append(int(d))
    else:
        track_ids = [int(x) for x in args.tracks.split(',') if x.strip()]

    left_set = set()
    if args.left is not None:
        left_set = set(int(x) for x in args.left.split(',') if x.strip())
    if left_set:
        print(f"Treating tracks as LEFT hand: {sorted(left_set)}")

    # If combine mode, accumulate projected joints per frame across all tracks
    combined = defaultdict(list) if args.combine else None

    for tid in track_ids:
        tid_dir = cam_space / str(tid)
        if not tid_dir.exists():
            continue
        out_dir = out_root / str(tid)
        if not args.combine:
            out_dir.mkdir(parents=True, exist_ok=True)
        chunk_files = natsorted(glob(str(tid_dir / '*_*.json')))
        is_left = tid in left_set
        color = (255, 0, 0) if is_left else (0, 255, 0)  # BGR: left=blue, right=green

        for cf in chunk_files:
            cf_path = Path(cf)
            try:
                s_idx, e_idx = parse_chunk_range(cf_path)
            except Exception:
                print(f"Skip chunk with unexpected name: {cf_path}")
                continue

            R_root, R_hand, t_root, betas = load_cam_space_chunk(cf_path)
            # Sanity check
            T = R_root.shape[0]
            if not (R_hand.shape[0] == t_root.shape[0] == betas.shape[0] == T):
                print(f"Shape mismatch in {cf_path}")
                continue

            try:
                joints, verts, faces, edges = mano_forward_rotmat(R_root,
                                                                  R_hand,
                                                                  t_root,
                                                                  betas,
                                                                  is_left=is_left,
                                                                  device='cpu')
            except Exception as e:
                print(f"MANO forward failed for {cf_path}: {e}")
                continue

            # For each frame in chunk
            for i in range(T):
                frame_idx = s_idx + i
                if frame_idx < 0 or frame_idx >= len(imgfiles):
                    continue

                # joints
                k3d = joints[i]
                k2d, Z = project_points_cam(k3d, fx, fy, cx, cy)
                k2d[Z <= 1e-6] = np.nan
                # mesh verts
                v3d = verts[i]
                v2d, ZV = project_points_cam(v3d, fx, fy, cx, cy)
                v2d[ZV <= 1e-6] = np.nan

                if args.combine:
                    combined[frame_idx].append((k2d, v2d, edges, color))
                else:
                    img = cv2.imread(imgfiles[frame_idx])
                    if img is None:
                        continue
                    canvas = img.copy()
                    if args.draw in ('mesh', 'both'):
                        canvas = draw_mesh_wireframe(canvas, v2d, edges, color=(0, 200, 255))
                    if args.draw in ('joints', 'both'):
                        canvas = draw_openpose_skeleton(canvas, k2d, color=color)
                    out_path = out_dir / (Path(imgfiles[frame_idx]).stem + '.jpg')
                    cv2.imwrite(str(out_path), canvas)

            if not args.combine:
                print(f"Projected track {tid} chunk {cf_path.stem} to {out_dir}")

    # Write combined overlays if requested
    if args.combine and combined:
        comb_dir = out_root / 'combined'
        comb_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx in sorted(combined.keys()):
            img = cv2.imread(imgfiles[frame_idx])
            if img is None:
                continue
            canvas = img.copy()
            for item in combined[frame_idx]:
                if len(item) == 2:
                    k2d, color = item
                    canvas = draw_openpose_skeleton(canvas, k2d, color=color)
                else:
                    k2d, v2d, edges, color = item
                    if args.draw in ('mesh', 'both'):
                        canvas = draw_mesh_wireframe(canvas, v2d, edges, color=(0, 200, 255))
                    if args.draw in ('joints', 'both'):
                        canvas = draw_openpose_skeleton(canvas, k2d, color=color)
            out_path = comb_dir / (Path(imgfiles[frame_idx]).stem + '.jpg')
            cv2.imwrite(str(out_path), canvas)
        print(f"Projected combined overlays to {comb_dir}")

    print(f"All done. Results in {out_root}")


if __name__ == '__main__':
    main()
