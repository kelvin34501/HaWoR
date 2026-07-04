import os
import sys

# glibc reads MALLOC_* tunables once, at process startup (before any Python code
# runs), so setting them via os.environ here would be too late. Re-exec the
# interpreter once with them applied instead: MALLOC_ARENA_MAX=2 collapses the
# many per-thread malloc arenas -- whose stranded free lists ballooned host RSS
# to ~70 GiB on a 3000-frame 4K window -- down to 2, and MALLOC_TRIM_THRESHOLD_=0
# makes free() hand pages back to the OS. This alone dropped the measured peak to
# ~31 GiB. Any value the caller already exported is respected; the sentinel env
# var prevents an exec loop.
_mem_defaults = {"MALLOC_ARENA_MAX": "2", "MALLOC_TRIM_THRESHOLD_": "0"}
_mem_missing = {k: v for k, v in _mem_defaults.items() if k not in os.environ}
if _mem_missing and not os.environ.get("_HAWOR_MALLOC_TUNED"):
    os.environ.update(_mem_missing)
    os.environ["_HAWOR_MALLOC_TUNED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import ctypes
import ctypes.util
import gc
import time

# Cap intra-op CPU thread pools BEFORE torch/numpy import their BLAS backends
# (OMP/MKL/OpenBLAS read these env vars once, at first use). Oversubscribed pools
# on a many-core node both waste CPU and multiply glibc malloc arenas, whose
# stranded per-arena free lists are what balloon RSS (see MALLOC_ARENA_MAX in the
# launch scripts). Respect any value the caller already exported.
_thread_cap = os.environ.get("HAWOR_NUM_THREADS")
if not _thread_cap:
    _thread_cap = str(min(8, os.cpu_count() or 8))
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, _thread_cap)

import torch
torch.set_num_threads(int(_thread_cap))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import joblib
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from lib.pipeline.frame_source import frame_source_from_args
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam
from lib.vis.run_vis2 import run_vis2_on_video, run_vis2_on_video_cam


_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)


def reclaim_memory():
    """Drop dead buffers between pipeline stages and return pages to the OS.

    Each stage churns through large transient host buffers (decoded 4K frames,
    crops, render targets). Once a stage returns they are unreferenced, but
    glibc keeps them on per-arena free lists so RSS never falls -- the next
    stage then stacks its peak on top. gc.collect() breaks any ref cycles,
    empty_cache() frees the CUDA caching allocator, and malloc_trim(0) hands
    the freed top-of-heap pages back to the OS so the peak doesn't accumulate.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        _libc.malloc_trim(0)
    except AttributeError:
        pass  # non-glibc allocator: no malloc_trim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default="example/clip_3_handonly/00.00.04.840-00.00.26.540--seg02.MP4")
    parser.add_argument("--input_type", type=str, default='file')
    parser.add_argument("--checkpoint",  type=str, default='./weights/hawor/checkpoints/hawor.ckpt')
    parser.add_argument("--infiller_weight",  type=str, default='./weights/hawor/checkpoints/infiller.pt')
    parser.add_argument("--vis_mode",  type=str, default='world', help='cam | world')
    parser.add_argument("--target_fps", type=float, default=30, help='decode/resample fps (matches old ffmpeg fps=N)')
    parser.add_argument("--frame_start", type=int, default=0, help='first frame (in target_fps timeline) of the window to process')
    parser.add_argument("--frame_end", type=int, default=None, help='end frame (exclusive) of the window to process; None = to end')
    parser.add_argument("--seq_dir", type=str, default=None, help='output seq folder override (segmented pipeline uses a distinct dir per window)')
    args = parser.parse_args()

    start = time.perf_counter()
    start_idx, end_idx, seq_folder, frame_source = detect_track_video(args)
    elapsed = time.perf_counter() - start
    print(f"Detection and tracking time: {elapsed:.4f} seconds, num frames: {end_idx - start_idx}")
    reclaim_memory()  # release detect/track host buffers before motion estimation stacks on top

    start = time.perf_counter()
    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)
    elapsed = time.perf_counter() - start
    print(f"Motion estimation time: {elapsed:.4f} seconds, num frames: {end_idx - start_idx}")
    reclaim_memory()  # release motion-estimation host buffers before SLAM

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        start = time.perf_counter()
        hawor_slam(args, start_idx, end_idx)
        elapsed = time.perf_counter() - start
        print(f"SLAM time: {elapsed:.4f} seconds, num frames: {end_idx - start_idx}")
        reclaim_memory()  # release SLAM/Metric3D host buffers before infilling
    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

    start = time.perf_counter()
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(args, start_idx, end_idx, frame_chunks_all)
    elapsed = time.perf_counter() - start
    print(f"Infilling time: {elapsed:.4f} seconds, num frames: {end_idx - start_idx}")
    reclaim_memory()  # release infiller host buffers before (optional) visualization
    # vis sequence for this video
    hand2idx = {
        "right": 1,
        "left": 0
    }
    vis_start = 0
    vis_end = pred_trans.shape[1] - 1
            
    # get faces
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

    # get right hand vertices
    hand = 'right'
    hand_idx = hand2idx[hand]
    pred_glob_r = run_mano(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
    right_verts = pred_glob_r['vertices'][0]
    right_dict = {
            'vertices': right_verts.unsqueeze(0),
            'faces': faces_right,
        }

    # get left hand vertices
    faces_left = faces_right[:,[0,2,1]]
    hand = 'left'
    hand_idx = hand2idx[hand]
    pred_glob_l = run_mano_left(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
    left_verts = pred_glob_l['vertices'][0]
    left_dict = {
            'vertices': left_verts.unsqueeze(0),
            'faces': faces_left,
        }

    R_x = torch.tensor([[1,  0,  0],
                        [0, -1,  0],
                        [0,  0, -1]]).float()
    R_c2w_sla_all = torch.einsum('ij,njk->nik', R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum('ij,nj->ni', R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)
    left_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, left_dict['vertices'].cpu())
    right_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, right_dict['vertices'].cpu())
    
    # Here we use aitviewer(https://github.com/eth-ait/aitviewer) for simple visualization.
    if args.vis_mode == 'off':
        print("Skipping visualization (vis_mode=off)")
    elif args.vis_mode == 'world':
        output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
        if not os.path.exists(output_pth):
            os.makedirs(output_pth)
        # Decode background frames on demand (RGB) for the vis range.
        image_source = frame_source_from_args(args, color='rgb')[vis_start:vis_end]
        print(f"vis {vis_start} to {vis_end}")
        run_vis2_on_video(left_dict, right_dict, output_pth, img_focal, image_source, R_c2w=R_c2w_sla_all[vis_start:vis_end], t_c2w=t_c2w_sla_all[vis_start:vis_end])
    elif args.vis_mode == 'cam':
        output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
        if not os.path.exists(output_pth):
            os.makedirs(output_pth)
        image_source = frame_source_from_args(args, color='rgb')[vis_start:vis_end]
        print(f"vis {vis_start} to {vis_end}")
        run_vis2_on_video_cam(left_dict, right_dict, output_pth, img_focal, image_source, R_w2c=R_w2c_sla_all[vis_start:vis_end], t_w2c=t_w2c_sla_all[vis_start:vis_end])

    print("finish")



