"""Quick NPZ inspector.

Examples:
  python scripts/read_npz.py --npz_path example/video_0/SLAM/hawor_slam_w_scale_0_121.npz
  python scripts/read_npz.py --npz_path example/video_0/SLAM/hawor_slam_w_scale_0_121.npz --keys traj tstamp
  python scripts/read_npz.py --npz_path example/video_0/SLAM/hawor_slam_w_scale_0_121.npz --max_entries 8 --show_values
"""

import argparse
import os


def _format_scalar(x):
    try:
        return f"{float(x):.6g}"
    except Exception:
        return str(x)


def _preview_values(arr, max_entries):
    flat = arr.reshape(-1)
    n = min(len(flat), max_entries)
    preview = ", ".join(_format_scalar(v) for v in flat[:n])
    suffix = " ..." if len(flat) > n else ""
    return f"[{preview}{suffix}]"


def _describe_array(name, arr, max_entries, show_values):
    print(f"- {name}")
    print(f"  type: {type(arr).__name__}")

    if not hasattr(arr, "shape"):
        print(f"  value: {arr}")
        return

    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")

    if arr.ndim == 0:
        print(f"  scalar: {_format_scalar(arr.item())}")
        return

    if arr.size == 0:
        print("  empty array")
        return

    if arr.dtype.kind in "iufb":
        print(f"  min/max: {_format_scalar(arr.min())} / {_format_scalar(arr.max())}")
        print(f"  mean: {_format_scalar(arr.mean())}")

    if show_values:
        print(f"  preview: {_preview_values(arr, max_entries)}")


def main():
    parser = argparse.ArgumentParser(description="Inspect NPZ file content")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to .npz file")
    parser.add_argument("--keys", nargs="*", default=None, help="Only inspect selected keys")
    parser.add_argument("--max_entries", type=int, default=6, help="Max preview values")
    parser.add_argument("--show_values", action="store_true", help="Show flattened preview values")
    args = parser.parse_args()

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("numpy is required to inspect npz files") from exc

    if not os.path.exists(args.npz_path):
        raise FileNotFoundError(f"NPZ not found: {args.npz_path}")

    with np.load(args.npz_path, allow_pickle=True) as data:
        all_keys = list(data.keys())
        print(f"NPZ: {args.npz_path}")
        print(f"keys ({len(all_keys)}): {all_keys}")

        inspect_keys = args.keys if args.keys else all_keys
        missing = [k for k in inspect_keys if k not in data]
        if missing:
            print(f"warning: missing keys: {missing}")

        for k in inspect_keys:
            if k in data:
                _describe_array(k, data[k], args.max_entries, args.show_values)


if __name__ == "__main__":
    main()
