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
processes. The reader is opened lazily, discarded immediately in a forked child,
and never serialized into a spawned worker.
"""

import gc
import os
import threading
import weakref
import ctypes
import ctypes.util
from glob import glob

import cv2
import numpy as np
from natsort import natsorted


_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png")
_DEFAULT_DECORD_NUM_THREADS = 1
# Service windows contain at most 3,001 frames. Keeping one reader for the whole
# window avoids repeatedly rebuilding Decord's HEVC state; the real-video memory
# benchmark remains flat because Decord's internal queues are bounded.
_DEFAULT_DECORD_RECYCLE_AFTER = 4096

# Decord readers own native decoder threads and frame buffers. Keep exactly one
# active reader in a process even when callers construct several FrameSource
# instances (for example, the interpolation stage probes both 30 fps and 50 fps
# timelines). Opening a reader through one source releases the previous source's
# reader; it will reopen transparently if used again.
_ACTIVE_READER_LOCK = threading.RLock()
_ACTIVE_READER_PID = None
_ACTIVE_READER_OWNER = None
_LIBC = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)


def _discard_active_reader_after_fork():
    """Forget native state inherited into a forked child.

    Reinitialize the lock without acquiring the inherited instance: another
    parent thread may have held it at fork time.
    """
    global _ACTIVE_READER_LOCK, _ACTIVE_READER_PID, _ACTIVE_READER_OWNER
    owner_ref = _ACTIVE_READER_OWNER
    _ACTIVE_READER_LOCK = threading.RLock()
    _ACTIVE_READER_PID = None
    _ACTIVE_READER_OWNER = None
    owner = owner_ref() if owner_ref is not None else None
    if owner is not None:
        owner._drop_reader(unregister=False)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_discard_active_reader_after_fork)


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

    def __init__(self, source, target_fps=30, start=0, end=None, color="bgr",
                 num_threads=None, recycle_after=None):
        assert color in ("bgr", "rgb"), color
        self.color = color
        self.target_fps = float(target_fps)
        # decord defaults num_threads=0 -> one decode thread per CPU core, each
        # with its own frame buffers; on a many-core node that alone balloons RAM.
        # One thread was the best bounded-memory setting in the 4K loader probe.
        if num_threads is None:
            num_threads = os.environ.get(
                "HAWOR_DECORD_NUM_THREADS", str(_DEFAULT_DECORD_NUM_THREADS)
            )
        self.num_threads = _positive_int(num_threads, "num_threads")
        # Flush interval: drop & reopen the decord reader after this many decoded
        # frames so its internal decode/seek buffers can't grow without bound.
        if recycle_after is None:
            recycle_after = os.environ.get(
                "HAWOR_DECORD_RECYCLE_AFTER", str(_DEFAULT_DECORD_RECYCLE_AFTER)
            )
        self.recycle_after = _positive_int(recycle_after, "recycle_after")
        self._video_reader = None
        self._reader_pid = None
        # Native index returned by the next reader.next() call. ``None`` means
        # Decord's position is unknown and the next access must seek accurately.
        self._reader_next_index = None
        self._reads = 0
        self._frame_shape = None

        if _is_video_path(source):
            self._mode = "video"
            self.video_path = source
            self._paths = None
            try:
                index_map = self._build_video_index_map()
            except BaseException:
                self.close()
                raise
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
        global _ACTIVE_READER_PID, _ACTIVE_READER_OWNER
        pid = os.getpid()
        if self._reader_pid is not None and self._reader_pid != pid:
            self._drop_reader()
        if self._video_reader is not None:
            return self._video_reader

        with _ACTIVE_READER_LOCK:
            owner = None
            if _ACTIVE_READER_PID == pid and _ACTIVE_READER_OWNER is not None:
                owner = _ACTIVE_READER_OWNER()
            if owner is not None and owner is not self:
                owner._drop_reader(unregister=False)

            import decord

            try:
                reader = decord.VideoReader(
                    self.video_path, ctx=decord.cpu(0), num_threads=self.num_threads
                )
            except BaseException:
                self._drop_reader()
                raise
            self._video_reader = reader
            self._reader_pid = pid
            self._reader_next_index = None
            self._reads = 0
            _ACTIVE_READER_PID = pid
            _ACTIVE_READER_OWNER = weakref.ref(self)
            return reader

    def _count_reads(self, n):
        """Track decoded frames and recycle the reader once the cap is hit.

        decord's ``VideoReader`` holds decode/seek buffers for the life of the
        object; over a long chunk this grows steadily. Dropping the reader
        releases those buffers -- the next frame access reopens it transparently.
        """
        self._reads += int(n)
        if self._reads >= self.recycle_after:
            self._drop_reader()

    def _drop_reader(self, *, unregister=True):
        """Release this source's reader and its native buffers immediately."""
        global _ACTIVE_READER_PID, _ACTIVE_READER_OWNER
        reader = self._video_reader
        self._video_reader = None
        self._reader_pid = None
        self._reader_next_index = None
        self._reads = 0

        if unregister:
            with _ACTIVE_READER_LOCK:
                owner = _ACTIVE_READER_OWNER() if _ACTIVE_READER_OWNER is not None else None
                if owner is self:
                    _ACTIVE_READER_PID = None
                    _ACTIVE_READER_OWNER = None

        # CPython destroys the native VideoReader when its final reference is
        # deleted. Clear our state first so exceptions from native finalization
        # cannot leave a stale reader registered.
        had_reader = reader is not None
        del reader
        if had_reader:
            # Decord's Python/native wrapper can participate in reference cycles
            # through callbacks. Collect at the lifecycle boundary so recycle and
            # close have deterministic native-memory semantics.
            gc.collect()
            try:
                _LIBC.malloc_trim(0)
            except AttributeError:
                pass

    def close(self):
        """Release all cached decord readers (safe to call repeatedly).

        Non-destructive: a subsequent frame access reopens the reader lazily, so
        callers use this to free decoder memory between pipeline stages.
        """
        self._drop_reader()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        """Never serialize a native reader into a spawned worker."""
        state = self.__dict__.copy()
        state["_video_reader"] = None
        state["_reader_pid"] = None
        state["_reader_next_index"] = None
        state["_reads"] = 0
        return state

    def _build_video_index_map(self):
        with _ACTIVE_READER_LOCK:
            reader = self._reader()
            try:
                n_native = len(reader)
                if n_native == 0:
                    return np.zeros(0, dtype=np.int64)
                try:
                    native_fps = float(reader.get_avg_fps())
                except Exception:
                    native_fps = self.target_fps
            except BaseException:
                self._drop_reader()
                raise
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
        with _ACTIVE_READER_LOCK:
            try:
                reader = self._reader()
                # VideoReader.__getitem__ always calls seek_accurate(), even when
                # indices increase monotonically. The pipeline reads that way in
                # detection, SLAM, and visualization. Preserve the decoder cursor
                # and advance it directly; for 50 -> 30 fps, skip_frames decodes
                # dependency frames without needlessly converting/copying them.
                next_index = self._reader_next_index
                if next_index is None:
                    if native_idx:
                        reader.seek_accurate(native_idx)
                elif native_idx < next_index:
                    reader.seek_accurate(native_idx)
                elif native_idx > next_index:
                    reader.skip_frames(native_idx - next_index)
                frame = reader.next().asnumpy()  # RGB HWC uint8
                self._reader_next_index = native_idx + 1
                out = self._to_color(frame)
                self._count_reads(1)
                return out
            except BaseException:
                self._drop_reader()
                raise

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

        # A caller can request more than one recycle interval in a single batch.
        # Split it so no native reader decodes more than recycle_after frames.
        chunks = []
        offset = 0
        while offset < len(native):
            with _ACTIVE_READER_LOCK:
                remaining = self.recycle_after - self._reads
                chunk_native = native[offset:offset + remaining]
                try:
                    frames = self._reader().get_batch(chunk_native.tolist()).asnumpy()
                    if self.color == "bgr":
                        frames = frames[:, :, :, ::-1]
                    chunks.append(np.ascontiguousarray(frames))
                    # Decord does not expose the cursor left by get_batch().
                    # Force the next scalar access to establish it accurately.
                    self._reader_next_index = None
                    self._count_reads(len(chunk_native))
                    offset += len(chunk_native)
                except BaseException:
                    self._drop_reader()
                    raise

        if not chunks:
            return np.empty((0, *self.frame_shape, 3), dtype=np.uint8)
        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks, axis=0)

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


def make_frame_source(source, target_fps=30, start=0, end=None, color="bgr",
                      num_threads=None, recycle_after=None):
    """Convenience factory mirroring the :class:`FrameSource` constructor."""
    return FrameSource(
        source,
        target_fps=target_fps,
        start=start,
        end=end,
        color=color,
        num_threads=num_threads,
        recycle_after=recycle_after,
    )


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
        num_threads=getattr(args, "decord_num_threads", None),
        recycle_after=getattr(args, "decord_recycle_after", None),
    )


def _positive_int(value, name):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed
