from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from service.config import load_service_config
from service.hawor_video_processor_for_service import (
    COPY_BACK_DISPARITY_ARTIFACTS,
    PEAK_RSS_ACCEPTANCE_BYTES,
    HaWoRProcessorConfig,
    HaWoRVideoProcessorForService,
    _PeakRssTracker,
    _read_process_tree_rss_bytes,
)
from service.job_manager import JobManager, JobRecord, VideoItem


class ServiceConfigurationTests(unittest.TestCase):

    def test_defaults_environment_overrides_and_invalid_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            mounts = Path(tmp) / "mounts"
            with mock.patch.dict(os.environ, {}, clear=True):
                defaults = load_service_config(cache_dir=cache, mount_root=mounts)
            self.assertEqual(defaults.decord_num_threads, 1)
            self.assertEqual(defaults.decord_recycle_after, 4096)

            with mock.patch.dict(
                os.environ,
                {
                    "HAWOR_DECORD_NUM_THREADS": "2",
                    "HAWOR_DECORD_RECYCLE_AFTER": "96",
                },
                clear=True,
            ):
                overridden = load_service_config(cache_dir=cache, mount_root=mounts)
            self.assertEqual(overridden.decord_num_threads, 2)
            self.assertEqual(overridden.decord_recycle_after, 96)

            for name, value in (
                ("HAWOR_DECORD_NUM_THREADS", "0"),
                ("HAWOR_DECORD_NUM_THREADS", "invalid"),
                ("HAWOR_DECORD_RECYCLE_AFTER", "-1"),
                ("HAWOR_DECORD_RECYCLE_AFTER", "invalid"),
            ):
                with self.subTest(name=name, value=value):
                    with mock.patch.dict(os.environ, {name: value}, clear=True):
                        with self.assertRaisesRegex(ValueError, "positive integer"):
                            load_service_config(cache_dir=cache, mount_root=mounts)

    def test_cli_and_health_expose_decoder_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            # api_server creates its default ASGI app at import time, so give that
            # import writable roots as well as the explicit app below.
            with mock.patch.dict(
                os.environ,
                {
                    "HAWOR_CACHE_DIR": str(Path(tmp) / "import-cache"),
                    "HAWOR_MOUNT_ROOT": str(Path(tmp) / "import-mounts"),
                },
            ):
                from service.api_server import _build_arg_parser, create_app

            args = _build_arg_parser().parse_args(
                ["--decord-num-threads", "2", "--decord-recycle-after", "80"]
            )
            self.assertEqual(args.decord_num_threads, 2)
            self.assertEqual(args.decord_recycle_after, 80)

            config = load_service_config(
                cache_dir=Path(tmp) / "cache",
                mount_root=Path(tmp) / "mounts",
                decord_num_threads=2,
                decord_recycle_after=80,
            )
            app = create_app(config)
            health_endpoint = next(
                route.endpoint for route in app.routes
                if getattr(route, "path", None) == "/healthz"
            )
            health = health_endpoint()
        self.assertEqual(health["decord_num_threads"], 2)
        self.assertEqual(health["decord_recycle_after"], 80)


class ServiceMemoryTelemetryTests(unittest.TestCase):

    def _processor(self):
        return HaWoRVideoProcessorForService(
            config=HaWoRProcessorConfig(
                project_dir=Path(__file__).resolve().parents[1],
                run_post_steps=False,
                run_world_space=False,
                run_visualizations=False,
            )
        )

    def test_process_tree_sampler_and_command_log(self):
        self.assertGreater(_read_process_tree_rss_bytes(os.getpid()), 0)
        processor = self._processor()
        tracker = _PeakRssTracker()
        child_code = (
            "import subprocess,sys,time;"
            "p=subprocess.Popen([sys.executable,'-c',"
            "\"import time; x=bytearray(24*1024*1024); time.sleep(0.25)\"]);"
            "p.wait();time.sleep(0.05)"
        )
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "process.log"
            processor._run_command(
                [sys.executable, "-c", child_code],
                log_path,
                os.environ.copy(),
                tracker,
            )
            processor._write_peak_rss_summary(log_path, tracker.peak_rss_bytes)
            log_text = log_path.read_text()

        self.assertGreater(tracker.peak_rss_bytes, 24 * 1024 ** 2)
        self.assertIn("command_peak_rss_bytes=", log_text)
        self.assertIn(f"peak_rss_bytes={tracker.peak_rss_bytes}", log_text)
        self.assertIn("acceptance=PASS", log_text)

    def test_processor_child_environment_is_memory_bounded_and_overridable(self):
        processor = self._processor()
        with mock.patch.dict(os.environ, {}, clear=True):
            defaults = processor._build_env(3)
        self.assertEqual(defaults["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(defaults["HAWOR_NUM_THREADS"], "1")
        self.assertEqual(defaults["HAWOR_DECORD_NUM_THREADS"], "1")
        self.assertEqual(defaults["HAWOR_DECORD_RECYCLE_AFTER"], "4096")

        with mock.patch.dict(os.environ, {"HAWOR_NUM_THREADS": "4"}, clear=True):
            overridden = processor._build_env(3)
        self.assertEqual(overridden["HAWOR_NUM_THREADS"], "4")

    def test_acceptance_gate_reports_over_budget_without_killing(self):
        self.assertEqual(PEAK_RSS_ACCEPTANCE_BYTES, 20 * 1024 ** 3)
        processor = self._processor()
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "process.log"
            processor._write_peak_rss_summary(log_path, PEAK_RSS_ACCEPTANCE_BYTES)
            log_text = log_path.read_text()
        self.assertIn("acceptance=FAIL", log_text)

    def test_api_result_contains_video_peak_rss(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_service_config(
                cache_dir=Path(tmp) / "cache",
                mount_root=Path(tmp) / "mounts",
                max_workers=1,
            )
            manager = JobManager(config)
            item = VideoItem(
                video="clip.mp4",
                status="SUCCEEDED",
                result_dir="/results/clip",
                log_path="/results/clip/process.log",
                peak_rss_bytes=123456789,
            )
            manager._jobs["job"] = JobRecord(
                job_id="job",
                status="SUCCEEDED",
                stage="SUCCEEDED",
                input_dir="/input",
                output_dir="/results",
                videos_total=1,
                items=[item],
            )
            try:
                result = manager.get_result("job")
            finally:
                manager.shutdown()
        self.assertEqual(result["items"][0]["peak_rss_bytes"], 123456789)

    def test_video_runner_updates_peak_telemetry_on_success_and_failure(self):
        class FakeProcessor:

            def __init__(self, *, fail=False):
                self.fail = fail

            def process_video(self, *args, peak_rss_callback, **kwargs):
                peak_rss_callback(111)
                if self.fail:
                    peak_rss_callback(222)
                    raise RuntimeError("synthetic failure")
                return SimpleNamespace(peak_rss_bytes=333)

        with tempfile.TemporaryDirectory() as tmp:
            config = load_service_config(
                cache_dir=Path(tmp) / "cache",
                mount_root=Path(tmp) / "mounts",
                max_workers=1,
            )
            manager = JobManager(config)
            manager._jobs["job"] = JobRecord(
                job_id="job",
                status="RUNNING",
                stage="PROCESSING",
                input_dir="/input",
                output_dir="/results",
                videos_total=1,
                items=[
                    VideoItem(
                        video="clip.mp4",
                        status="PENDING",
                        result_dir="/results/clip",
                        log_path="/results/clip/process.log",
                    )
                ],
            )
            try:
                manager._process_single_video(
                    "job",
                    FakeProcessor(),
                    Path("clip.mp4"),
                    Path("/results/clip"),
                    Path(tmp) / "scratch",
                    False,
                    True,
                )
                self.assertEqual(
                    manager.get_job("job")["items"][0]["peak_rss_bytes"], 333
                )

                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    manager._process_single_video(
                        "job",
                        FakeProcessor(fail=True),
                        Path("clip.mp4"),
                        Path("/results/clip"),
                        Path(tmp) / "scratch",
                        False,
                        True,
                    )
                self.assertEqual(
                    manager.get_job("job")["items"][0]["peak_rss_bytes"], 222
                )
            finally:
                manager.shutdown()


class ServiceOutputArtifactTests(unittest.TestCase):

    def _processor(self):
        return HaWoRVideoProcessorForService(
            config=HaWoRProcessorConfig(
                project_dir=Path(__file__).resolve().parents[1],
                run_post_steps=False,
                run_world_space=False,
                run_visualizations=False,
            )
        )

    def test_disparity_cleanup_only_deletes_transient_exports(self):
        self.assertFalse(COPY_BACK_DISPARITY_ARTIFACTS)
        processor = self._processor()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            slam_dir = output_dir / "SLAM"
            slam_dir.mkdir()
            transient_npz = slam_dir / "hawor_slam_w_scale_disps_0_10_50fps.npz"
            transient_mkv = slam_dir / "hawor_slam_w_scale_disps_0_10_50fps_uint16.mkv"
            retained_slam = slam_dir / "hawor_slam_w_scale_0_10_50fps.npz"
            retained_video = slam_dir / "diagnostic.mkv"
            for path in (
                transient_npz,
                transient_mkv,
                retained_slam,
                retained_video,
            ):
                path.write_bytes(b"x")

            processor._delete_disparity_artifacts(output_dir)

            self.assertFalse(transient_npz.exists())
            self.assertFalse(transient_mkv.exists())
            self.assertTrue(retained_slam.exists())
            self.assertTrue(retained_video.exists())

    def test_process_video_does_not_copy_back_disparity_exports(self):
        processor = self._processor()

        def create_merged_output(_video, work_dir, *_args):
            slam_dir = work_dir / "merged" / "SLAM"
            slam_dir.mkdir(parents=True)
            (slam_dir / "hawor_slam_w_scale_disps_0_10_50fps.npz").write_bytes(b"disp")
            (slam_dir / "hawor_slam_w_scale_disps_0_10_50fps_uint16.mkv").write_bytes(b"disp")
            (slam_dir / "hawor_slam_w_scale_0_10_50fps.npz").write_bytes(b"traj")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            video.write_bytes(b"video")
            output_dir = root / "output"
            scratch_dir = root / "scratch"

            with mock.patch.object(
                processor,
                "_run_segmented_pipeline",
                side_effect=create_merged_output,
            ):
                processor.process_video(
                    video,
                    output_dir,
                    scratch_dir=scratch_dir,
                )

            slam_dir = output_dir / "SLAM"
            self.assertFalse(
                (slam_dir / "hawor_slam_w_scale_disps_0_10_50fps.npz").exists()
            )
            self.assertFalse(
                (
                    slam_dir
                    / "hawor_slam_w_scale_disps_0_10_50fps_uint16.mkv"
                ).exists()
            )
            self.assertTrue(
                (slam_dir / "hawor_slam_w_scale_0_10_50fps.npz").exists()
            )


if __name__ == "__main__":
    unittest.main()
