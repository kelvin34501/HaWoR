_EMPTY_PROXIMITY_GRAPH_MESSAGE = (
    "not enough values to unpack (expected 2, got 0)"
)


def _is_empty_proximity_graph_error(exc):
    if not isinstance(exc, ValueError):
        return False
    if str(exc) != _EMPTY_PROXIMITY_GRAPH_MESSAGE:
        return False

    frames = []
    traceback = exc.__traceback__
    while traceback is not None:
        code = traceback.tb_frame.f_code
        frames.append(
            (code.co_filename.replace("\\", "/"), code.co_name)
        )
        traceback = traceback.tb_next

    return (
        len(frames) >= 2
        and frames[-2][0].endswith("droid_slam/droid_backend.py")
        and frames[-2][1] == "__call__"
        and frames[-1][0].endswith("droid_slam/factor_graph.py")
        and frames[-1][1] == "add_proximity_factors"
    )


def run_with_unmasked_fallback(
    primary,
    fallback,
    cleanup,
    terminal_fallback=None,
):
    """Retry an empty masked graph unmasked, then optionally use a terminal fallback."""
    try:
        return primary()
    except ValueError as exc:
        if not _is_empty_proximity_graph_error(exc):
            raise

    # Run after leaving the except block so its traceback no longer retains the
    # failed DROID instance and its GPU tensors.
    cleanup()
    print(
        "[slam:fallback] Masked DROID produced an empty proximity graph; "
        "retrying the complete chunk without masks"
    )
    try:
        return fallback()
    except ValueError as exc:
        if terminal_fallback is None or not _is_empty_proximity_graph_error(exc):
            raise

    # As above, clean up only after leaving the except block so the traceback no
    # longer retains the failed DROID instance and its GPU tensors.
    cleanup()
    print(
        "[slam:fallback] Unmasked DROID also produced an empty proximity graph; "
        "using an identical camera pose throughout the chunk"
    )
    return terminal_fallback()
