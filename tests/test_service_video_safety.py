from __future__ import annotations

import os
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

from service.hawor_video_processor_for_service import (
    HaWoRProcessorConfig,
    HaWoRVideoProcessorForService,
    VideoTimingInfo,
    _PeakRssTracker,
)


class ServiceTimestampSafetyTests(unittest.TestCase):

    def _processor(self, **overrides):
        values = {
            "project_dir": Path(__file__).resolve().parents[1],
            "run_post_steps": False,
            "run_world_space": False,
            "run_visualizations": False,
        }
        values.update(overrides)
        return HaWoRVideoProcessorForService(config=HaWoRProcessorConfig(**values))

    @staticmethod
    def _timing(min_pts: float, *, cfr: bool = True, b_frames: bool = False):
        return VideoTimingInfo(
            codec_name="hevc",
            fps=Fraction(50, 1),
            is_cfr=cfr,
            has_b_frames=b_frames,
            min_pts_seconds=min_pts,
            min_dts_seconds=min_pts,
        )

    def test_safe_input_is_used_without_copy(self):
        processor = self._processor()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            video.write_bytes(b"video")
            log = root / "process.log"
            with (
                mock.patch.object(processor, "_probe_video_timing", return_value=self._timing(0.0)),
                mock.patch.object(processor, "_run_command") as run_command,
            ):
                selected = processor._prepare_video_for_processing(
                    video,
                    root,
                    log,
                    os.environ.copy(),
                    _PeakRssTracker(),
                )

        self.assertEqual(selected, video)
        run_command.assert_not_called()

    def test_negative_cfr_input_is_exactly_retimestamped_and_verified(self):
        processor = self._processor()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            video.write_bytes(b"video")
            log = root / "process.log"

            def fake_run(cmd, *_args):
                Path(cmd[-1]).write_bytes(b"normalized")

            with (
                mock.patch.object(
                    processor,
                    "_probe_video_timing",
                    side_effect=[self._timing(-0.82), self._timing(0.0)],
                ),
                mock.patch.object(processor, "_run_command", side_effect=fake_run) as run_command,
            ):
                selected = processor._prepare_video_for_processing(
                    video,
                    root,
                    log,
                    os.environ.copy(),
                    _PeakRssTracker(),
                )
                command = run_command.call_args.args[0]
                log_text = log.read_text()

        self.assertEqual(selected.name, "_normalized_input.mp4")
        self.assertIn("-c:v", command)
        self.assertEqual(command[command.index("-c:v") + 1], "copy")
        self.assertIn("-bsf:v", command)
        self.assertIn("setts=pts=N*1/(50*TB)", command[command.index("-bsf:v") + 1])
        self.assertIn("normalized decoder timeline verified", log_text)

    def test_b_frame_input_preserves_packet_order_with_timestamp_offset(self):
        processor = self._processor()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            video.write_bytes(b"video")

            def fake_run(cmd, *_args):
                Path(cmd[-1]).write_bytes(b"normalized")

            with (
                mock.patch.object(
                    processor,
                    "_probe_video_timing",
                    side_effect=[
                        self._timing(-0.04, b_frames=True),
                        self._timing(0.0, b_frames=True),
                    ],
                ),
                mock.patch.object(processor, "_run_command", side_effect=fake_run) as run_command,
            ):
                processor._prepare_video_for_processing(
                    video,
                    root,
                    root / "process.log",
                    os.environ.copy(),
                    _PeakRssTracker(),
                )
                command = run_command.call_args.args[0]

        self.assertNotIn("-bsf:v", command)
        self.assertEqual(
            command[command.index("-avoid_negative_ts") + 1],
            "make_zero",
        )

    def test_every_pipeline_stage_receives_the_prepared_video(self):
        processor = self._processor(
            cleanup_intermediate=False,
            run_post_steps=True,
            run_world_space=True,
            run_visualizations=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            prepared = root / "scratch" / "_normalized_input.mp4"
            video.write_bytes(b"video")
            prepared.parent.mkdir()
            prepared.write_bytes(b"normalized")

            def create_merged(_video, work_dir, *_args):
                slam_dir = work_dir / "merged" / "SLAM"
                slam_dir.mkdir(parents=True)
                (slam_dir / "hawor_slam_w_scale_0_1.npz").write_bytes(b"trajectory")

            def create_cam_video(seq_dir, _video, *_args):
                path = seq_dir / "cam_space_visualization_50fps.mp4"
                path.write_bytes(b"cam")
                return path

            def create_world_video(seq_dir, _video, *_args):
                path = seq_dir / "world_space_visualization_50fps.mp4"
                path.write_bytes(b"world")
                return path

            with (
                mock.patch.object(processor, "_prepare_video_for_processing", return_value=prepared),
                mock.patch.object(
                    processor,
                    "_run_segmented_pipeline",
                    side_effect=create_merged,
                ) as segmented,
                mock.patch.object(processor, "_build_world_space_res") as world_res,
                mock.patch.object(processor, "_run_interpolation") as interpolation,
                mock.patch.object(
                    processor,
                    "_run_cam_space_visualization",
                    side_effect=create_cam_video,
                ) as cam_video,
                mock.patch.object(
                    processor,
                    "_run_world_space_visualization",
                    side_effect=create_world_video,
                ) as world_video,
            ):
                result = processor.process_video(
                    video,
                    root / "output",
                    scratch_dir=root / "scratch",
                )

        self.assertEqual(segmented.call_args.args[0], prepared)
        self.assertEqual(world_res.call_args.args[1], prepared)
        self.assertEqual(interpolation.call_args.args[1], prepared)
        self.assertEqual(cam_video.call_args.args[1], prepared)
        self.assertEqual(world_video.call_args.args[1], prepared)
        self.assertEqual(result.video_path, video.resolve())


if __name__ == "__main__":
    unittest.main()
