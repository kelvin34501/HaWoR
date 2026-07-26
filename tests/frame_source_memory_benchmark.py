#!/usr/bin/env python3
"""Decode a real video sequentially and enforce the loader RSS acceptance gate."""

from __future__ import annotations

import argparse
import hashlib
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.pipeline.frame_source import FrameSource


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--min-frames", type=int, default=600)
    parser.add_argument("--max-peak-gib", type=float, default=20.0)
    parser.add_argument("--color", choices=("rgb", "bgr"), default="rgb")
    parser.add_argument(
        "--expected-sha256",
        help="Optionally require the concatenated decoded frame bytes to match",
    )
    args = parser.parse_args()

    checksum = 0
    decoded_sha256 = hashlib.sha256()
    started = time.perf_counter()
    source = FrameSource(args.video_path, target_fps=30, color=args.color)
    try:
        if len(source) < args.min_frames:
            raise RuntimeError(
                f"benchmark input has {len(source)} frames; need at least {args.min_frames}"
            )
        for index in range(args.min_frames):
            frame = source[index]
            decoded_sha256.update(memoryview(frame))
            checksum = (checksum + int(frame[0, 0, 0])) % 1_000_000_007
    finally:
        source.close()
    elapsed_seconds = time.perf_counter() - started

    # Linux reports ru_maxrss in KiB.
    peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    limit_bytes = int(args.max_peak_gib * 1024 ** 3)
    decoded_digest = decoded_sha256.hexdigest()
    print(
        f"frames={args.min_frames} seconds={elapsed_seconds:.3f} "
        f"fps={args.min_frames / elapsed_seconds:.3f} peak_rss_bytes={peak_rss_bytes} "
        f"peak_rss_gib={peak_rss_bytes / 1024 ** 3:.3f} checksum={checksum} "
        f"decoded_sha256={decoded_digest}"
    )
    if args.expected_sha256 and decoded_digest != args.expected_sha256.lower():
        raise SystemExit(
            f"FAIL: decoded SHA-256 {decoded_digest} does not match "
            f"{args.expected_sha256.lower()}"
        )
    if peak_rss_bytes >= limit_bytes:
        raise SystemExit(
            f"FAIL: loader peak RSS {peak_rss_bytes / 1024 ** 3:.3f} GiB "
            f"is not below {args.max_peak_gib:.3f} GiB"
        )
    print(f"PASS: loader peak RSS is below {args.max_peak_gib:.3f} GiB")


if __name__ == "__main__":
    main()
