from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from service.config import load_service_config
from service.job_manager import JobManager


class LocalApiJobTests(unittest.TestCase):

    def _manager(self, root: Path) -> JobManager:
        config = load_service_config(
            cache_dir=root / "cache",
            mount_root=root / "mounts",
            max_workers=1,
        )
        return JobManager(config)

    def _create_job(
        self,
        manager: JobManager,
        input_url: str,
        *,
        endpoint=None,
        access_key=None,
        secret_key=None,
        skip_processed=False,
    ):
        return manager.create_job(
            input_url=input_url,
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            region=None,
            force_path_style=False,
            use_listobject_v2=False,
            vis_mode="off",
            overwrite=False,
            skip_processed=skip_processed,
        )

    def test_api_request_accepts_local_path_without_s3_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.dict(
                os.environ,
                {
                    "HAWOR_CACHE_DIR": str(root / "import-cache"),
                    "HAWOR_MOUNT_ROOT": str(root / "import-mounts"),
                },
            ):
                from service.api_server import AnnotateRequest, create_app

            input_dir = root / "videos"
            request = AnnotateRequest(input_url=str(input_dir))
            config = load_service_config(
                cache_dir=root / "cache",
                mount_root=root / "mounts",
            )
            fake_manager = mock.Mock()
            fake_manager.create_job.return_value = {"job_id": "local-job"}
            with mock.patch(
                "service.api_server.JobManager",
                return_value=fake_manager,
            ):
                app = create_app(config)
            annotate_endpoint = next(
                route.endpoint
                for route in app.routes
                if getattr(route, "path", None) == "/v1/annotate"
            )
            response = annotate_endpoint(request)

        self.assertIsNone(request.endpoint)
        self.assertIsNone(request.access_key)
        self.assertIsNone(request.secret_key)
        self.assertEqual(response, {"job_id": "local-job"})
        fake_manager.create_job.assert_called_once_with(
            input_url=str(input_dir),
            endpoint=None,
            access_key=None,
            secret_key=None,
            region=None,
            force_path_style=False,
            use_listobject_v2=False,
            vis_mode="off",
            overwrite=False,
            skip_processed=False,
        )

    def test_local_job_skips_mount_and_uses_annotations_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "videos"
            input_dir.mkdir()
            (input_dir / "clip.mp4").write_bytes(b"test")
            manager = self._manager(root)
            try:
                with (
                    mock.patch.object(manager._mount_manager, "mount") as mount,
                    mock.patch("service.job_manager.threading.Thread") as thread,
                ):
                    result = self._create_job(manager, str(input_dir))

                mount.assert_not_called()
                thread.return_value.start.assert_called_once_with()
                self.assertEqual(result["input_dir"], str(input_dir.resolve()))
                self.assertEqual(
                    result["output_dir"],
                    str((input_dir / "annotations").resolve()),
                )
                self.assertEqual(result["videos_total"], 1)
                self.assertEqual(result["items"][0]["video"], "clip.mp4")
                self.assertIsNone(manager._jobs[result["job_id"]].mount_handle)
            finally:
                manager.shutdown()

    def test_local_job_preserves_skip_processed_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "videos"
            input_dir.mkdir()
            (input_dir / "clip.mp4").write_bytes(b"test")
            done_dir = input_dir / "annotations" / "clip"
            done_dir.mkdir(parents=True)
            (done_dir / "process.done").write_text("x")
            manager = self._manager(root)
            try:
                with mock.patch("service.job_manager.threading.Thread"):
                    result = self._create_job(
                        manager,
                        str(input_dir),
                        skip_processed=True,
                    )
                self.assertEqual(result["videos_skipped"], 1)
                self.assertEqual(result["items"][0]["status"], "SKIPPED")
            finally:
                manager.shutdown()

    def test_s3_input_still_requires_connection_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(Path(tmp))
            complete = {
                "endpoint": "https://objects.example",
                "access_key": "ak",
                "secret_key": "sk",
            }
            try:
                for missing in complete:
                    values = dict(complete)
                    values[missing] = None
                    with self.subTest(missing=missing):
                        with self.assertRaisesRegex(
                            ValueError,
                            rf"{missing} is required for s3:// input_url",
                        ):
                            self._create_job(
                                manager,
                                "s3://bucket/prefix",
                                **values,
                            )
            finally:
                manager.shutdown()

    def test_non_s3_url_scheme_is_not_treated_as_a_local_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(Path(tmp))
            try:
                with self.assertRaisesRegex(
                    ValueError,
                    "s3:// URL or a local directory path",
                ):
                    self._create_job(manager, "https://example.test/videos")
            finally:
                manager.shutdown()


if __name__ == "__main__":
    unittest.main()
