import numpy as np


def constant_camera_trajectory(frame_count):
    """Return ``frame_count`` identical identity poses in DROID's xyzw format."""
    frame_count = int(frame_count)
    if frame_count < 1:
        raise ValueError("constant camera trajectory requires at least one frame")

    identity_xyzw = np.asarray(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    return np.repeat(identity_xyzw[None], frame_count, axis=0)


def metric_depth_to_disparity(metric_depth):
    """Convert metric depth to finite inverse depth for a degraded SLAM artifact."""
    metric_depth = np.asarray(metric_depth, dtype=np.float32)
    disparity = np.zeros_like(metric_depth, dtype=np.float32)
    valid = np.isfinite(metric_depth) & (metric_depth > 0)
    np.divide(1.0, metric_depth, out=disparity, where=valid)

    # Metric3D should normally produce positive finite depth. Keep the terminal
    # fallback terminal even for a fully invalid prediction by emitting a finite
    # unit-disparity plane; the artifact is explicitly marked degraded.
    used_unit_plane = not np.any(valid)
    if used_unit_plane:
        disparity.fill(1.0)

    return disparity, used_unit_plane
