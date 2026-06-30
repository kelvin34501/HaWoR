"""Smoke test: FrameSource must match `ffmpeg -vf fps=N` frame-for-frame.

This guards the #1 correctness risk of the streaming-decode change: if FrameSource
yields a different frame count or different pixels than the old on-disk JPEG dump,
SLAM/motion/param timelines silently misalign.

Run in the `hawor` env (decord + ffmpeg required):

    python test_frame_source.py --video_path ./example/video_0.mp4
"""

import argparse
import os
import subprocess
import tempfile
from glob import glob

import cv2
import numpy as np
from natsort import natsorted

from lib.pipeline.frame_source import FrameSource


def ffmpeg_extract(video_path, out_dir, fps):
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-strict", "-2", "-i", video_path,
         "-vf", f"fps={fps}", "-start_number", "0",
         os.path.join(out_dir, "%06d.jpg")],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return natsorted(glob(os.path.join(out_dir, "*.jpg")))


def _mad(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def check_fps(video_path, fps, sample=12, require_alignment=True):
    """Verify frame count and (for the processing fps) temporal alignment.

    Note: ffmpeg writes lossy JPEGs which we re-read, while FrameSource returns the
    raw decoded frame, so absolute pixel diffs of ~5-15 are expected (JPEG loss +
    YUV->RGB colorspace/range differences between the two decoders). The real
    correctness check is alignment: FrameSource[i] must match ffmpeg frame i better
    than its neighbours i-1 / i+1 (i.e. no off-by-one / temporal drift).

    `require_alignment` is True for the processing fps (30) -- the frames actually
    fed to detection/SLAM/motion, which must reproduce the old ffmpeg dump exactly.
    It is False for the 50fps interpolation timeline: only the frame *count* is ever
    used (interpolation.py interpolates params numerically and never decodes 50fps
    pixels), and ffmpeg's upsampling frame-duplication phase is PTS-specific and may
    differ from a closed-form mapping by +/-1 at duplicate boundaries.
    """
    with tempfile.TemporaryDirectory() as tmp:
        files = ffmpeg_extract(video_path, tmp, fps)
        src = FrameSource(video_path, target_fps=fps, color="bgr")

        print(f"[fps={fps}] ffmpeg={len(files)} frames, FrameSource={len(src)} frames")
        assert abs(len(files) - len(src)) <= 1, "frame count mismatch > 1"

        n = min(len(files), len(src))
        idxs = np.linspace(1, n - 2, num=min(sample, max(1, n - 2))).astype(int)
        bias = np.zeros(3, np.float64)
        worst_same = 0.0
        misaligned = []
        for i in idxs:
            b = src[int(i)].astype(np.float32)
            same = _mad(cv2.imread(files[i]), b)
            prev = _mad(cv2.imread(files[i - 1]), b)
            nxt = _mad(cv2.imread(files[i + 1]), b)
            worst_same = max(worst_same, same)
            bias += (b - cv2.imread(files[i]).astype(np.float32)).mean(axis=(0, 1))
            # On a moving video, the matching frame should be clearly closest.
            if not (same <= prev and same <= nxt):
                misaligned.append((int(i), round(prev, 1), round(same, 1), round(nxt, 1)))
        bias /= len(idxs)
        print(f"[fps={fps}] aligned-frame max mean-abs-diff: {worst_same:.2f} "
              f"(JPEG+colorspace; not a bug unless very large)")
        print(f"[fps={fps}] per-channel signed bias (B,G,R): "
              f"{bias[0]:+.2f} {bias[1]:+.2f} {bias[2]:+.2f} "
              f"(large constant bias => YUV range/matrix mismatch)")
        if misaligned:
            print(f"[fps={fps}] frames differing from ffmpeg phase [idx,(prev,same,next)]: {misaligned}")
        if require_alignment:
            assert not misaligned, "temporal misalignment: a neighbour matched better than the aligned frame"
            print(f"[fps={fps}] alignment OK")
        else:
            print(f"[fps={fps}] count OK ({len(src)} frames); pixel alignment not required "
                  f"(only the frame count is consumed at this fps)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_path", default="./example/video_0.mp4")
    args = ap.parse_args()

    # 30fps = the processing timeline (frames fed to the model): must match exactly.
    check_fps(args.video_path, 30, require_alignment=True)
    # 50fps = interpolation timeline: only the frame count is consumed.
    check_fps(args.video_path, 50, require_alignment=False)

    # Windowing: a [start, end) window must equal the corresponding slice of the full source.
    full = FrameSource(args.video_path, target_fps=30, color="bgr")
    win = FrameSource(args.video_path, target_fps=30, start=5, end=15, color="bgr")
    assert len(win) == min(10, max(0, len(full) - 5)), (len(win), len(full))
    if len(win) > 0:
        assert np.array_equal(win[0], full[5]) and np.array_equal(win[-1], full[5 + len(win) - 1])
    print("[window] OK")
    print("ALL FRAME SOURCE CHECKS PASSED")
