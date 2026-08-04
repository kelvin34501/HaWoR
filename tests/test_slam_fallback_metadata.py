import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.interpolation import (
    _interpolated_slam_metadata as interpolated_metadata_with_disps,
)
from scripts.interpolation_no_disps import (
    _interpolated_slam_metadata as interpolated_metadata_without_disps,
)
from scripts.segmented_demo_pipeline import ChunkMeta, merge_slam


class SlamFallbackMetadataTests(unittest.TestCase):
    def test_segmented_merge_records_degraded_global_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = root / "chunk"
            slam_dir = chunk / "SLAM"
            slam_dir.mkdir(parents=True)

            trajectory = np.zeros((4, 7), dtype=np.float32)
            trajectory[:, 6] = 1.0
            np.savez(
                slam_dir / "hawor_slam_w_scale_0_4.npz",
                traj=trajectory,
                disps=np.ones((1, 2, 2), dtype=np.float32),
                tstamp=np.asarray([0], dtype=np.int32),
                scale=np.asarray(1.0, dtype=np.float32),
                img_focal=np.asarray(600.0, dtype=np.float32),
                img_center=np.asarray([320.0, 240.0], dtype=np.float32),
                slam_valid=np.asarray(False),
                slam_fallback=np.asarray("constant_pose"),
            )

            output = root / "merged" / "SLAM"
            merged_path = merge_slam(
                [ChunkMeta(str(chunk), str(chunk), 0, 4, 0, 4)],
                str(output),
                "keep_last",
            )
            data = dict(np.load(merged_path))

        self.assertFalse(bool(data["slam_valid"]))
        self.assertEqual(str(data["slam_fallback"]), "partial")
        np.testing.assert_array_equal(
            data["slam_fallback_ranges"],
            np.asarray([[0, 4]], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            data["slam_fallback_modes"],
            np.asarray(["constant_pose"]),
        )

    def test_interpolation_maps_degraded_ranges_to_new_timeline(self):
        data = {
            "slam_valid": np.asarray(False),
            "slam_fallback": np.asarray("partial"),
            "slam_fallback_modes": np.asarray(["constant_pose"]),
            "slam_fallback_ranges": np.asarray([[1, 3]], dtype=np.int64),
        }
        t_new = np.linspace(0, 3, 7, dtype=np.float64)

        with_disps = interpolated_metadata_with_disps(data, t_new)
        without_disps = interpolated_metadata_without_disps(data, t_new)

        for metadata in (with_disps, without_disps):
            self.assertFalse(bool(metadata["slam_valid"]))
            np.testing.assert_array_equal(
                metadata["slam_fallback_ranges"],
                np.asarray([[2, 6]], dtype=np.int64),
            )


if __name__ == "__main__":
    unittest.main()
