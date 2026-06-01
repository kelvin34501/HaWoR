#!/usr/bin/env python3
import os
import re
import json
import glob
import argparse
import numpy as np

RANGE_RE = re.compile(r"(\d+)_(\d+)(?:_50fps)?\.json$")

FIELDS = ["init_root_orient", "init_hand_pose", "init_trans", "init_betas"]

def parse_range(path):
    m = RANGE_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Bad chunk name: {path}")
    return int(m.group(1)), int(m.group(2))

def to_np(v):
    return np.asarray(v, dtype=np.float32)

def merge_hand_dir(hand_dir, out_path, overlap_policy="keep_last"):
    chunks = sorted(glob.glob(os.path.join(hand_dir, "*.json")))
    if not chunks:
        raise FileNotFoundError(f"No json chunks in {hand_dir}")

    parsed = []
    max_end = -1
    for p in chunks:
        s, e = parse_range(p)
        parsed.append((s, e, p))
        max_end = max(max_end, e)
    parsed.sort(key=lambda x: x[0])

    # read first chunk for shape template
    with open(parsed[0][2], "r") as f:
        first = json.load(f)

    buffers = {}
    valid = np.zeros(max_end + 1, dtype=bool)

    for k in FIELDS:
        a = to_np(first[k])  # (B,T,...) expected
        if a.ndim < 2:
            raise ValueError(f"{k} has invalid shape: {a.shape}")
        shape = (a.shape[0], max_end + 1) + a.shape[2:]
        buffers[k] = np.zeros(shape, dtype=np.float32)

    for s, e, p in parsed:
        with open(p, "r") as f:
            d = json.load(f)

        # chunk length from data, not only filename
        t_len = min(to_np(d["init_trans"]).shape[1], e - s + 1)
        if t_len <= 0:
            continue
        dst = slice(s, s + t_len)

        if overlap_policy == "keep_first":
            mask = ~valid[dst]
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            for k in FIELDS:
                src = to_np(d[k])[:, :t_len]
                for j in idx:
                    buffers[k][:, s + j] = src[:, j]
            valid[s:s+t_len][idx] = True
        else:  # keep_last
            for k in FIELDS:
                src = to_np(d[k])[:, :t_len]
                buffers[k][:, s:s+t_len] = src
            valid[s:s+t_len] = True

    # optionally trim to covered range
    if not valid.any():
        raise RuntimeError("No valid frames merged")
    last = np.where(valid)[0][-1]
    out = {k: buffers[k][:, :last+1].tolist() for k in FIELDS}

    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved: {out_path}, frames=0..{last}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_space_dir", required=True, help=".../cam_space")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--overlap_policy", choices=["keep_last", "keep_first"], default="keep_last")
    args = ap.parse_args()

    hand_dirs = sorted([p for p in glob.glob(os.path.join(args.cam_space_dir, "*")) if os.path.isdir(p)])
    if not hand_dirs:
        raise FileNotFoundError(f"No hand dirs under {args.cam_space_dir}")

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.cam_space_dir), "cam_space_merged")
    os.makedirs(out_dir, exist_ok=True)

    for hd in hand_dirs:
        hand = os.path.basename(hd)
        out_path = os.path.join(out_dir, f"{hand}.json")
        merge_hand_dir(hd, out_path, args.overlap_policy)

if __name__ == "__main__":
    main()