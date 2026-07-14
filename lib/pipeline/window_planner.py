"""Pure frame-window planning for the segmented video pipeline."""

from typing import List, Tuple


FrameWindow = Tuple[int, int]


def compute_windows(
    n_frames: int,
    chunk_frames: int,
    min_last_frames: int,
    max_chunk_frames: int = 3001,
) -> List[FrameWindow]:
    """Split ``[0, n_frames)`` into capped consecutive owned windows.

    ``max_chunk_frames`` is a hard cap on frames passed to one demo run, including
    the one-frame overlap added to non-final windows. A trailing window shorter
    than ``min_last_frames`` is merged only when the merged final window fits the
    cap; otherwise the boundary between the final two windows is moved so the
    tail is long enough without creating an oversized run.
    """
    if n_frames <= 0:
        return []
    if max_chunk_frames < 2:
        raise ValueError("max_chunk_frames must be >= 2")

    # Every non-final owned window gets one extra SLAM overlap frame later.
    max_owned_frames = max_chunk_frames - 1
    owned_chunk_frames = min(
        chunk_frames if chunk_frames > 0 else max_owned_frames,
        max_owned_frames,
    )

    windows = [
        (start, min(start + owned_chunk_frames, n_frames))
        for start in range(0, n_frames, owned_chunk_frames)
    ]
    if len(windows) >= 2 and min_last_frames > 0:
        last_start, last_end = windows[-1]
        if (last_end - last_start) < min_last_frames:
            previous_start, _ = windows[-2]
            combined_frames = last_end - previous_start
            if combined_frames <= max_chunk_frames:
                windows[-2] = (previous_start, last_end)
                windows.pop()
            else:
                # Keep the preceding non-final window small enough for its
                # additional overlap frame while preferring the requested tail.
                min_safe_last = max(1, combined_frames - max_owned_frames)
                max_safe_last = min(max_chunk_frames, combined_frames - 1)
                target_last_frames = max(
                    min_safe_last,
                    min(min_last_frames, max_safe_last),
                )
                new_boundary = last_end - target_last_frames
                windows[-2] = (previous_start, new_boundary)
                windows[-1] = (new_boundary, last_end)
    return windows


def processing_windows(windows: List[FrameWindow], n_frames: int) -> List[FrameWindow]:
    """Add the one-frame SLAM overlap to every non-terminal owned window."""
    return [(start, min(end + 1, n_frames)) for start, end in windows]
