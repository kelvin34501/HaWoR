"""Pure frame-window planning for the segmented video pipeline.

The HaWoR pose network consumes fixed, non-overlapping temporal blocks.  A
segmented run therefore cannot safely start each outer window at an arbitrary
frame: doing so can change temporal grouping, and padding at an outer edge
changes the predictions on otherwise valid frames.  The helpers below keep
outer boundaries aligned and reserve real-frame context on both sides while
still respecting the hard per-process frame cap. Camera-space merging then
crossfades any remaining track-phase difference through that shared context.
"""

from typing import List, Tuple


FrameWindow = Tuple[int, int]


def compute_windows(
    n_frames: int,
    chunk_frames: int,
    min_last_frames: int,
    max_chunk_frames: int = 3001,
    left_context_frames: int = 0,
    right_context_frames: int = 1,
    alignment_frames: int = 1,
) -> List[FrameWindow]:
    """Split ``[0, n_frames)`` into capped consecutive owned windows.

    ``max_chunk_frames`` is a hard cap on frames passed to one demo run,
    including the requested left/right context.  Every interior boundary is a
    multiple of ``alignment_frames`` so outer processes begin on a stable global
    phase.  A short trailing window is merged or rebalanced only when all
    resulting processing windows still fit the cap.

    The defaults preserve the historical planner: no left context, one right
    overlap frame, and arbitrary frame boundaries.
    """
    if n_frames <= 0:
        return []
    if max_chunk_frames < 2:
        raise ValueError("max_chunk_frames must be >= 2")
    if left_context_frames < 0 or right_context_frames < 0:
        raise ValueError("context frame counts must be non-negative")
    if alignment_frames < 1:
        raise ValueError("alignment_frames must be positive")
    if n_frames <= max_chunk_frames:
        return [(0, n_frames)]

    max_owned_frames = (
        max_chunk_frames - left_context_frames - right_context_frames
    )
    if max_owned_frames < alignment_frames:
        raise ValueError(
            "max_chunk_frames is too small for the requested context and "
            "alignment"
        )

    requested_owned_frames = min(
        chunk_frames if chunk_frames > 0 else max_owned_frames,
        max_owned_frames,
    )
    owned_chunk_frames = (
        requested_owned_frames // alignment_frames * alignment_frames
    )
    if owned_chunk_frames <= 0:
        raise ValueError("no aligned owned frames fit in one processing window")

    windows = [
        (start, min(start + owned_chunk_frames, n_frames))
        for start in range(0, n_frames, owned_chunk_frames)
    ]
    if len(windows) >= 2 and min_last_frames > 0:
        last_start, last_end = windows[-1]
        if (last_end - last_start) < min_last_frames:
            previous_start, _ = windows[-2]
            merged_span = _processing_span(
                previous_start,
                last_end,
                n_frames,
                left_context_frames,
                right_context_frames,
            )
            if merged_span <= max_chunk_frames:
                windows[-2] = (previous_start, last_end)
                windows.pop()
            else:
                candidates = []
                first_boundary = (
                    (previous_start + alignment_frames - 1)
                    // alignment_frames
                    * alignment_frames
                )
                for boundary in range(
                    first_boundary,
                    last_end,
                    alignment_frames,
                ):
                    if boundary <= previous_start:
                        continue
                    previous_span = _processing_span(
                        previous_start,
                        boundary,
                        n_frames,
                        left_context_frames,
                        right_context_frames,
                    )
                    last_span = _processing_span(
                        boundary,
                        last_end,
                        n_frames,
                        left_context_frames,
                        right_context_frames,
                    )
                    if (
                        previous_span <= max_chunk_frames
                        and last_span <= max_chunk_frames
                    ):
                        candidates.append(boundary)

                if not candidates:
                    raise ValueError(
                        "cannot rebalance the final windows within the frame cap"
                    )

                long_enough = [
                    boundary
                    for boundary in candidates
                    if last_end - boundary >= min_last_frames
                ]
                if long_enough:
                    # Prefer the smallest tail satisfying the requested minimum;
                    # this leaves the preceding window as large as possible.
                    new_boundary = max(long_enough)
                else:
                    # The requested tail is impossible under the cap.  Avoid a
                    # pathological tiny edge by balancing the two final windows.
                    new_boundary = min(
                        candidates,
                        key=lambda boundary: abs(
                            (boundary - previous_start)
                            - (last_end - boundary)
                        ),
                    )
                windows[-2] = (previous_start, new_boundary)
                windows[-1] = (new_boundary, last_end)
    return windows


def _processing_span(
    start: int,
    end: int,
    n_frames: int,
    left_context_frames: int,
    right_context_frames: int,
) -> int:
    process_start = max(0, start - left_context_frames)
    process_end = min(n_frames, end + right_context_frames)
    return process_end - process_start


def processing_windows(
    windows: List[FrameWindow],
    n_frames: int,
    left_context_frames: int = 0,
    right_context_frames: int = 1,
) -> List[FrameWindow]:
    """Expand owned windows with real-frame context, clipped to the video."""
    if left_context_frames < 0 or right_context_frames < 0:
        raise ValueError("context frame counts must be non-negative")
    return [
        (
            max(0, start - left_context_frames),
            min(end + right_context_frames, n_frames),
        )
        for start, end in windows
    ]
