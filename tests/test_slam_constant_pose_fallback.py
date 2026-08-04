import unittest

import numpy as np

from lib.pipeline.slam_artifact_fallback import (
    constant_camera_trajectory,
    metric_depth_to_disparity,
)


class ConstantPoseFallbackTests(unittest.TestCase):
    def test_constant_trajectory_has_full_length_identity_poses(self):
        trajectory = constant_camera_trajectory(4)

        self.assertEqual(trajectory.shape, (4, 7))
        self.assertEqual(trajectory.dtype, np.float32)
        np.testing.assert_array_equal(trajectory[:, :6], 0.0)
        np.testing.assert_array_equal(trajectory[:, 6], 1.0)

    def test_constant_trajectory_rejects_empty_stream(self):
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            constant_camera_trajectory(0)

    def test_metric_depth_is_converted_to_finite_disparity(self):
        depth = np.asarray([[2.0, 4.0], [0.0, np.nan]], dtype=np.float32)

        disparity, used_unit_plane = metric_depth_to_disparity(depth)

        np.testing.assert_array_equal(
            disparity,
            np.asarray([[0.5, 0.25], [0.0, 0.0]], dtype=np.float32),
        )
        self.assertFalse(used_unit_plane)
        self.assertTrue(np.isfinite(disparity).all())

    def test_fully_invalid_metric_depth_uses_unit_plane(self):
        depth = np.asarray([[0.0, -1.0], [np.nan, np.inf]], dtype=np.float32)

        disparity, used_unit_plane = metric_depth_to_disparity(depth)

        np.testing.assert_array_equal(disparity, np.ones_like(depth))
        self.assertTrue(used_unit_plane)


if __name__ == "__main__":
    unittest.main()
