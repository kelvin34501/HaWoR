"""On-demand video frame source.

`FrameSource` decodes frames lazily from a video file instead of relying on a
pre-extracted directory of JPEGs. It is a drop-in replacement for the ``imgfiles``
path list that the pipeline used to glob from ``extracted_images/``: it supports
``len()``, integer indexing (returns an HWC uint8 ndarray), fancy/slice indexing
(returns a lightweight view), sequential iteration, and exposes ``frame_shape``.

Backend is ``decord`` today. It is wrapped behind this thin interface so a
``torchcodec`` backend can be swapped in later (torchcodec needs torch>=2.4; this
repo is pinned to torch 1.13 for the vendored CUDA extensions).

fps resampling
--------------
The pipeline historically extracted frames with ``ffmpeg -vf fps=30`` (and
``fps=50`` for the interpolation post-step). To keep frame counts and per-frame
indices identical to that behaviour, :class:`FrameSource` resamples the native
video to ``target_fps`` by mapping each output time ``n / target_fps`` to the most
recent native frame at that time (the same nearest-preceding selection ffmpeg's
``fps`` filter performs, for both constant- and variable-frame-rate inputs).

Color order
-----------
decord decodes to RGB. ``color`` selects what callers receive: detection/track and
SLAM expect BGR (they used ``cv2.imread`` directly); motion estimation and Metric3D
expect RGB.

fork safety
-----------
decord ``VideoReader`` objects are not fork-safe and must not be shared across
processes. The reader is opened lazily and cached per ``os.getpid()`` so DataLoader
workers and subprocesses each get their own.
"""

import os
from glob import glob

import cv2
import numpy as np
from natsort import natsorted


_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png")


def _is_video_path(source):
    if isinstance(source, (list, tuple)):
        return False
    if os.path.isdir(source):
        return False
    return True


class _FrameSubset:
    """A view over a subset of a :class:`FrameSource` (result of fancy indexing).

    Quacks like the old ``imgfiles[frame_ck]`` numpy slice: supports ``len()`` and
    integer indexing returning a decoded frame, plus further sub-indexing.

    ``indices`` are *virtual* full-timeline indices (the same space callers index
    ``FrameSource`` with), NOT native frame indices. Decoding routes back through
    ``parent[...]`` so the parent's ``index_map`` is applied exactly once; storing
    pre-mapped native indices here would double-apply it.
    """

    def __init__(self, parent, indices):
        self.parent = parent
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        if isinstance(key, slice) or isinstance(key, (list, np.ndarray)):
            return _FrameSubset(self.parent, self.indices[key])
        return self.parent[int(self.indices[key])]

    def __iter__(self):
        for i in self.indices:
            yield self.parent[int(i)]

    @property
    def frame_shape(self):
        return self.parent.frame_shape


class FrameSource:
    """Lazily decode frames from a video (or fall back to an image directory).

    Parameters
    ----------
    source : str | list
        Path to a video file, OR a directory of extracted frames, OR a list of
        image paths. The latter two are a backward-compatible fallback so eval
        scripts that still produce image directories keep working; ``target_fps``
        is ignored in that case (the frames are taken as-is).
    target_fps : float
        Resample rate for video inputs, matching the old ``ffmpeg -vf fps=N``.
    start, end : int
        Half-open ``[start, end)`` window into the (resampled) frame timeline. Used
        to process a chunk of a long video without splitting the file.
    color : {'bgr', 'rgb'}
        Color order of returned frames.
    """

    def __init__(self, source, target_fps=30, start=0, end=None, color="bgr"):
        assert color in ("bgr", "rgb"), color
        self.color = color
        self.target_fps = float(target_fps)
        self._readers = {}  # pid -> decord.VideoReader
        self._frame_shape = None

        if _is_video_path(source):
            self._mode = "video"
            self.video_path = source
            self._paths = None
            index_map = self._build_video_index_map()
        else:
            self._mode = "images"
            self.video_path = None
            if isinstance(source, (list, tuple)):
                self._paths = list(source)
            else:
                files = []
                for ext in _IMAGE_EXTS:
                    files.extend(glob(os.path.join(source, ext)))
                self._paths = natsorted(files)
            index_map = np.arange(len(self._paths), dtype=np.int64)

        full = len(index_map)
        end = full if end is None else min(int(end), full)
        start = max(0, int(start))
        # `index_map` maps a virtual full-timeline index -> underlying native frame
        # index (video) or path index (images); window it to [start, end).
        self._index_map = index_map[start:end]

    # ------------------------------------------------------------------ readers
    def _reader(self):
        pid = os.getpid()
        reader = self._readers.get(pid)
        if reader is None:
            import decord

            reader = decord.VideoReader(self.video_path, ctx=decord.cpu(0))
            self._readers[pid] = reader
        return reader

    def _build_video_index_map(self):
        reader = self._reader()
        n_native = len(reader)
        if n_native == 0:
            return np.zeros(0, dtype=np.int64)

        try:
            native_fps = float(reader.get_avg_fps())
        except Exception:
            native_fps = self.target_fps
        if not native_fps or native_fps <= 0:
            native_fps = self.target_fps

        # Reproduce `ffmpeg -vf fps=N` for constant-frame-rate input: output slot i
        # (presentation time i/target_fps) maps to native frame
        # floor(i * native_fps / target_fps). This matches ffmpeg exactly for CFR
        # video -- the repo's inputs are CFR (ffprobe r_frame_rate == avg_frame_rate)
        # and ffmpeg's own fps filter emits CFR output -- for both down- and
        # up-sampling. Crucially it uses the nominal frame rate rather than decord's
        # per-frame timestamps, whose ~1e-8 jitter (e.g. frame 1 reported at
        # 0.03333334 vs 1/30 = 0.03333333) otherwise causes an off-by-one.
        # (A variable-frame-rate source would need timestamp-based resampling here.)
        n_out = int(round(n_native * self.target_fps / native_fps))
        if n_out <= 0:
            return np.zeros(0, dtype=np.int64)

        ratio = native_fps / self.target_fps
        # +1e-6 guards floor() against tiny float undershoot at exact boundaries.
        native_idx = np.floor(np.arange(n_out, dtype=np.float64) * ratio + 1e-6)
        native_idx = np.clip(native_idx, 0, n_native - 1)
        return native_idx.astype(np.int64)

    # ------------------------------------------------------------------ reading
    def _to_color(self, frame_rgb):
        frame = frame_rgb[:, :, ::-1] if self.color == "bgr" else frame_rgb
        return np.ascontiguousarray(frame)

    def _read_native(self, native_idx):
        if self._mode == "images":
            img = cv2.imread(self._paths[native_idx])  # BGR
            if self.color == "rgb":
                img = img[:, :, ::-1]
            return np.ascontiguousarray(img)
        frame = self._reader()[native_idx].asnumpy()  # RGB HWC uint8
        return self._to_color(frame)

    def __len__(self):
        return len(self._index_map)

    def __getitem__(self, key):
        # `_FrameSubset` stores *virtual* (full-timeline) indices, not native ones:
        # decoding a subset element routes back through this method, which applies
        # `self._index_map` exactly once. Pre-mapping here would map twice (the
        # subset index then this index), reading the wrong native frame whenever
        # the map is non-identity (target_fps != native_fps).
        if isinstance(key, slice):
            idx = np.arange(len(self))[key]
            return _FrameSubset(self, idx)
        if isinstance(key, (list, np.ndarray)):
            idx = np.asarray(key, dtype=np.int64)
            return _FrameSubset(self, idx)
        if key < 0:
            key += len(self)
        return self._read_native(int(self._index_map[key]))

    def get_batch(self, indices):
        """Decode several frames at once (efficient for video; loops for images)."""
        indices = np.asarray(indices, dtype=np.int64)
        native = self._index_map[indices]
        if self._mode == "images":
            return np.stack([self._read_native(int(n)) for n in native], axis=0)
        frames = self._reader().get_batch(native.tolist()).asnumpy()  # N,H,W,3 RGB
        if self.color == "bgr":
            frames = frames[:, :, :, ::-1]
        return np.ascontiguousarray(frames)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def frame_shape(self):
        """(H, W) of decoded frames."""
        if self._frame_shape is None:
            if len(self) == 0:
                raise ValueError("empty FrameSource has no frame_shape")
            h, w = self[0].shape[:2]
            self._frame_shape = (int(h), int(w))
        return self._frame_shape


def make_frame_source(source, target_fps=30, start=0, end=None, color="bgr"):
    """Convenience factory mirroring the :class:`FrameSource` constructor."""
    return FrameSource(source, target_fps=target_fps, start=start, end=end, color=color)


def frame_source_from_args(args, color="bgr"):
    """Build a :class:`FrameSource` from a pipeline ``args`` namespace.

    Centralizes the CLI arg names (``video_path``, ``target_fps``, ``frame_start``,
    ``frame_end``) so every stage windows the source identically. ``frame_start``/
    ``frame_end`` select the chunk processed by this run (the segmented pipeline sets
    them per chunk); defaults cover the whole video.
    """
    return FrameSource(
        args.video_path,
        target_fps=getattr(args, "target_fps", None) or 30,
        start=getattr(args, "frame_start", None) or 0,
        end=getattr(args, "frame_end", None),
        color=color,
    )
