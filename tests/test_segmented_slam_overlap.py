from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import json

import numpy as np

from scripts.segmented_demo_pipeline import ChunkMeta, merge_cam_space, merge_slam


class SegmentedSlamOverlapTests(unittest.TestCase):

    @staticmethod
    def _write_run(seq_dir: Path, process_start: int, process_end: int, offset: float):
        slam_dir = seq_dir / "SLAM"
        slam_dir.mkdir(parents=True)
        length = process_end - process_start
        global_frames = np.arange(process_start, process_end, dtype=np.float32)
        traj = np.zeros((length, 7), dtype=np.float32)
        traj[:, 0] = global_frames * 0.1 + offset
        traj[:, 6] = 1.0  # xyzw identity quaternion
        np.savez(
            slam_dir / f"hawor_slam_w_scale_0_{length}.npz",
            traj=traj,
            disps=np.ones((length, 2, 2), dtype=np.float32),
            tstamp=np.arange(length, dtype=np.int64),
            scale=np.asarray(1.0, dtype=np.float32),
            img_focal=np.asarray(600.0, dtype=np.float32),
            img_center=np.asarray([320.0, 240.0], dtype=np.float32),
        )

    def test_all_shared_context_poses_align_and_blend_independent_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "chunk_0"
            second = root / "chunk_1"
            self._write_run(first, 0, 24, offset=0.0)
            self._write_run(second, 16, 40, offset=5.0)

            metas = [
                ChunkMeta(str(first), str(first), 0, 20, 0, 24),
                ChunkMeta(str(second), str(second), 20, 40, 16, 40),
            ]
            output = root / "merged" / "SLAM"
            path = merge_slam(metas, str(output), "keep_last")
            data = dict(np.load(path))

        expected_x = np.arange(40, dtype=np.float32) * 0.1
        np.testing.assert_allclose(data["traj"][:, 0], expected_x, atol=1e-5)
        np.testing.assert_array_equal(data["overlap_frames"], np.arange(16, 24))
        self.assertTrue(bool(data["overlap_aligned"]))


class SegmentedCameraSpaceOverlapTests(unittest.TestCase):

    @staticmethod
    def _write_run(seq_dir: Path, length: int, translation: float):
        hand_dir = seq_dir / "cam_space" / "1"
        hand_dir.mkdir(parents=True)
        identity = np.eye(3, dtype=np.float32)
        payload = {
            "init_root_orient": np.tile(identity, (1, length, 1, 1)).tolist(),
            "init_hand_pose": np.tile(identity, (1, length, 15, 1, 1)).tolist(),
            "init_trans": np.full((1, length, 3), translation, dtype=np.float32).tolist(),
            "init_betas": np.full((1, length, 10), translation, dtype=np.float32).tolist(),
        }
        (hand_dir / f"0_{length - 1}.json").write_text(json.dumps(payload))

    def test_camera_pose_crossfade_uses_both_context_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "chunk_0"
            second = root / "chunk_1"
            self._write_run(first, length=6, translation=0.0)
            self._write_run(second, length=6, translation=10.0)
            metas = [
                ChunkMeta(str(first), str(first), 0, 4, 0, 6),
                ChunkMeta(str(second), str(second), 4, 8, 2, 8),
            ]

            output = root / "merged" / "cam_space"
            merge_cam_space(metas, str(output), "keep_last")
            chunks = [
                json.loads(path.read_text())
                for path in sorted((output / "1").glob("*.json"))
            ]

        translations = np.concatenate(
            [np.asarray(data["init_trans"], dtype=np.float32)[0, :, 0] for data in chunks]
        )
        np.testing.assert_allclose(
            translations,
            np.asarray([0.0, 0.0, 2.5, 4.0, 6.0, 7.5, 10.0, 10.0]),
            atol=1e-5,
        )
        roots = np.concatenate(
            [np.asarray(data["init_root_orient"], dtype=np.float32)[0] for data in chunks]
        )
        np.testing.assert_allclose(
            roots,
            np.tile(np.eye(3, dtype=np.float32), (8, 1, 1)),
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
