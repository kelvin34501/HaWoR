from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import interpolation


class _PartialWriter:

    def __init__(self, max_write=7, error=None):
        self.data = bytearray()
        self.max_write = max_write
        self.error = error
        self.closed = False

    def write(self, data):
        if self.error is not None:
            raise self.error
        size = min(len(data), self.max_write)
        self.data.extend(data[:size])
        return size

    def close(self):
        self.closed = True


class _FakeProcess:

    def __init__(self, writer=None, returncode=0, stderr=b""):
        self.stdin = writer or _PartialWriter()
        self.returncode = returncode
        self.stderr = stderr
        self.kill_count = 0
        self.wait_count = 0

    def kill(self):
        self.kill_count += 1

    def wait(self):
        self.wait_count += 1
        return self.returncode


class InterpolationDispsTests(unittest.TestCase):

    def _legacy_payload(self, disps):
        disp = np.nan_to_num(disps, nan=0.0, posinf=0.0, neginf=0.0)
        return np.ascontiguousarray(
            np.clip(
                np.round(disp * interpolation.DISPS_U16_SCALE),
                0.0,
                65535.0,
            ).astype(np.uint16)
        ).tobytes()

    def _write_disps_npz(self, folder, disps):
        path = os.path.join(folder, "slam_disps_0_3_50fps.npz")
        np.savez(path, disps=disps)
        return path

    def test_frame_quantization_matches_legacy_payload(self):
        rng = np.random.default_rng(42)
        disps = rng.normal(2.0, 5.0, (17, 11, 13)).astype(np.float32)
        disps.ravel()[:10] = [
            np.nan,
            np.inf,
            -np.inf,
            -0.0,
            0.0,
            np.nextafter(np.float32(0), np.float32(1)),
            6.55345,
            6.55355,
            65535.0 / interpolation.DISPS_U16_SCALE,
            100.0,
        ]

        streamed = b"".join(
            interpolation._quantize_disps_frame(frame).tobytes()
            for frame in disps
        )

        self.assertEqual(streamed, self._legacy_payload(disps))
        quantized = interpolation._quantize_disps_frame(disps[0])
        self.assertEqual(quantized.dtype, np.uint16)
        self.assertTrue(quantized.flags.c_contiguous)

    def test_write_all_handles_partial_writes(self):
        writer = _PartialWriter(max_write=3)
        data = np.arange(12, dtype=np.uint16).reshape(3, 4)

        interpolation._write_all(writer, data)

        self.assertEqual(bytes(writer.data), data.tobytes())

    def test_video_conversion_streams_frames_with_unchanged_command(self):
        disps = np.array(
            [
                [[np.nan, 0.0], [1.0, np.inf]],
                [[-np.inf, -1.0], [6.55345, 6.55355]],
                [[0.25, 0.5], [2.0, 100.0]],
            ],
            dtype=np.float32,
        )
        process = _FakeProcess(writer=_PartialWriter(max_write=5))
        popen_calls = []

        def fake_popen(cmd, **kwargs):
            popen_calls.append((cmd, kwargs))
            return process

        with tempfile.TemporaryDirectory() as folder:
            disps_path = self._write_disps_npz(folder, disps)
            output = io.StringIO()
            with (
                mock.patch.object(
                    interpolation.shutil,
                    "which",
                    return_value="/test/ffmpeg",
                ),
                mock.patch.object(
                    interpolation.subprocess,
                    "Popen",
                    side_effect=fake_popen,
                ),
                contextlib.redirect_stdout(output),
            ):
                result = interpolation.disps_npz_to_uint16_video(
                    disps_path,
                    fps=47.5,
                )

            expected_video = os.path.splitext(disps_path)[0] + "_uint16.mkv"
            self.assertEqual(result, expected_video)

        self.assertEqual(bytes(process.stdin.data), self._legacy_payload(disps))
        self.assertTrue(process.stdin.closed)
        self.assertEqual(process.kill_count, 0)
        self.assertEqual(process.wait_count, 1)
        self.assertEqual(
            popen_calls[0][0],
            [
                "/test/ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray16le",
                "-s",
                "2x2",
                "-r",
                "47.5",
                "-i",
                "-",
                "-an",
                "-c:v",
                "ffv1",
                expected_video,
            ],
        )
        self.assertIs(popen_calls[0][1]["stdin"], interpolation.subprocess.PIPE)

    def test_encoder_failure_preserves_diagnostic_and_return_value(self):
        disps = np.zeros((1, 2, 2), dtype=np.float32)
        process = _FakeProcess(returncode=1, stderr=b"encoder failed\n")

        def fake_popen(cmd, **kwargs):
            kwargs["stderr"].write(process.stderr)
            return process

        with tempfile.TemporaryDirectory() as folder:
            disps_path = self._write_disps_npz(folder, disps)
            output = io.StringIO()
            with (
                mock.patch.object(
                    interpolation.shutil,
                    "which",
                    return_value="/test/ffmpeg",
                ),
                mock.patch.object(
                    interpolation.subprocess,
                    "Popen",
                    side_effect=fake_popen,
                ),
                contextlib.redirect_stdout(output),
            ):
                result = interpolation.disps_npz_to_uint16_video(disps_path)

        self.assertIsNone(result)
        self.assertIn("Skip disps video: ffmpeg failed", output.getvalue())
        self.assertIn("encoder failed", output.getvalue())
        self.assertEqual(process.kill_count, 0)
        self.assertEqual(process.wait_count, 1)

    def test_stream_failure_kills_and_reaps_encoder(self):
        disps = np.zeros((1, 2, 2), dtype=np.float32)
        process = _FakeProcess(
            writer=_PartialWriter(error=BrokenPipeError("closed"))
        )

        with tempfile.TemporaryDirectory() as folder:
            disps_path = self._write_disps_npz(folder, disps)
            with (
                mock.patch.object(
                    interpolation.shutil,
                    "which",
                    return_value="/test/ffmpeg",
                ),
                mock.patch.object(
                    interpolation.subprocess,
                    "Popen",
                    return_value=process,
                ),
            ):
                with self.assertRaisesRegex(BrokenPipeError, "closed"):
                    interpolation.disps_npz_to_uint16_video(disps_path)

        self.assertEqual(process.kill_count, 1)
        self.assertEqual(process.wait_count, 1)


if __name__ == "__main__":
    unittest.main()
