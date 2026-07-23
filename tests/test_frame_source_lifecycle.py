from __future__ import annotations

import gc
import os
import pickle
import sys
import types
import unittest
import weakref
from unittest import mock

import numpy as np

import lib.pipeline.frame_source as frame_source_module
from lib.pipeline.frame_source import FrameSource


class _FakeArray:

    def __init__(self, value):
        self._value = np.asarray(value)

    def asnumpy(self):
        return self._value.copy()


class _FakeDecord(types.ModuleType):

    def __init__(self):
        super().__init__("decord")
        indices = np.arange(220, dtype=np.uint8)
        self.frames = np.empty((220, 2, 3, 3), dtype=np.uint8)
        self.frames[..., 0] = indices[:, None, None]
        self.frames[..., 1] = (indices * 2)[:, None, None]
        self.frames[..., 2] = (indices * 3)[:, None, None]
        self.opens = []
        self.batch_sizes = []
        self.seeks = []
        self.alive = set()
        self.fail_indices = set()
        self.VideoReader = self._video_reader

    @staticmethod
    def cpu(index):
        return ("cpu", index)

    def _video_reader(self, path, *, ctx, num_threads):
        ident = len(self.opens) + 1
        self.opens.append({
            "id": ident,
            "path": path,
            "ctx": ctx,
            "num_threads": num_threads,
        })
        self.alive.add(ident)
        parent = self

        class Reader:

            def __len__(self):
                return len(parent.frames)

            @staticmethod
            def get_avg_fps():
                return 30.0

            def __getitem__(self, index):
                index = int(index)
                if index in parent.fail_indices:
                    raise RuntimeError("synthetic decode failure")
                return _FakeArray(parent.frames[index])

            def get_batch(self, indices):
                indices = [int(index) for index in indices]
                if any(index in parent.fail_indices for index in indices):
                    raise RuntimeError("synthetic batch failure")
                parent.batch_sizes.append(len(indices))
                return _FakeArray(parent.frames[indices])

            @staticmethod
            def seek(index):
                parent.seeks.append(int(index))

        reader = Reader()
        weakref.finalize(reader, self.alive.discard, ident)
        return reader


class FrameSourceLifecycleTests(unittest.TestCase):

    def setUp(self):
        owner_ref = frame_source_module._ACTIVE_READER_OWNER
        owner = owner_ref() if owner_ref is not None else None
        if owner is not None:
            owner.close()
        frame_source_module._ACTIVE_READER_PID = None
        frame_source_module._ACTIVE_READER_OWNER = None
        self.decord = _FakeDecord()
        self.decord_patch = mock.patch.dict(sys.modules, {"decord": self.decord})
        self.decord_patch.start()

    def tearDown(self):
        owner_ref = frame_source_module._ACTIVE_READER_OWNER
        owner = owner_ref() if owner_ref is not None else None
        if owner is not None:
            owner.close()
        self.decord_patch.stop()
        gc.collect()

    def test_defaults_and_sequential_reads_cross_recycle_boundaries(self):
        with mock.patch.dict(
            os.environ,
            {"HAWOR_DECORD_NUM_THREADS": "", "HAWOR_DECORD_RECYCLE_AFTER": ""},
            clear=False,
        ):
            os.environ.pop("HAWOR_DECORD_NUM_THREADS")
            os.environ.pop("HAWOR_DECORD_RECYCLE_AFTER")
            source = FrameSource("video.mp4", color="rgb")

        self.assertEqual(source.num_threads, 1)
        self.assertEqual(source.recycle_after, 64)
        decoded = [source[index] for index in range(150)]
        self.assertEqual(len(self.decord.opens), 3)
        self.assertEqual(self.decord.seeks, [0] * 9)
        self.assertEqual(decoded[129][0, 0].tolist(), self.decord.frames[129, 0, 0].tolist())
        self.assertEqual(len(self.decord.alive), 1)

        source.close()
        self.assertIsNone(source._video_reader)
        self.assertEqual(len(self.decord.alive), 0)
        source.close()

    def test_window_fancy_batch_and_color_access_across_recycles(self):
        source = FrameSource(
            "video.mp4", start=10, end=210, color="bgr", recycle_after=64
        )
        picks = np.concatenate((np.arange(130), np.array([199, 3, 64, 129])))
        subset = source[picks]
        decoded = list(subset)
        self.assertEqual(len(decoded), len(picks))
        native = 10 + int(picks[-1])
        self.assertEqual(
            decoded[-1][0, 0].tolist(),
            self.decord.frames[native, 0, 0, ::-1].tolist(),
        )

        batch = source.get_batch(np.arange(150))
        self.assertEqual(batch.shape, (150, 2, 3, 3))
        self.assertTrue(all(size <= 64 for size in self.decord.batch_sizes))
        self.assertGreaterEqual(len(self.decord.batch_sizes), 3)

        rgb = FrameSource("video.mp4", color="rgb")
        rgb_pixel = rgb[7][0, 0].copy()
        bgr = FrameSource("video.mp4", color="bgr")
        bgr_pixel = bgr[7][0, 0].copy()
        self.assertEqual(rgb_pixel.tolist(), bgr_pixel[::-1].tolist())

    def test_only_one_reader_is_active_per_process(self):
        first = FrameSource("first.mp4")
        first_reader = weakref.ref(first._video_reader)
        second = FrameSource("second.mp4")

        self.assertIsNone(first._video_reader)
        self.assertIsNone(first_reader())
        self.assertEqual(len(self.decord.alive), 1)
        self.assertIsNotNone(second._video_reader)

        first[0]
        self.assertIsNone(second._video_reader)
        self.assertIsNotNone(first._video_reader)
        self.assertEqual(len(self.decord.alive), 1)

    def test_pid_change_and_spawn_serialization_discard_reader(self):
        with mock.patch.object(frame_source_module.os, "getpid", return_value=101):
            source = FrameSource("video.mp4")
            inherited_reader = weakref.ref(source._video_reader)
            payload = pickle.dumps(source)

        restored = pickle.loads(payload)
        self.assertIsNone(restored._video_reader)
        self.assertIsNone(restored._reader_pid)

        with mock.patch.object(frame_source_module.os, "getpid", return_value=202):
            source[0]
        self.assertIsNone(inherited_reader())
        self.assertEqual(source._reader_pid, 202)

    def test_decode_exception_releases_reader(self):
        source = FrameSource("video.mp4")
        self.decord.fail_indices.add(17)
        with self.assertRaisesRegex(RuntimeError, "synthetic decode failure"):
            source[17]
        self.assertIsNone(source._video_reader)
        self.assertEqual(len(self.decord.alive), 0)

    def test_environment_and_explicit_settings_are_validated(self):
        with mock.patch.dict(
            os.environ,
            {
                "HAWOR_DECORD_NUM_THREADS": "3",
                "HAWOR_DECORD_RECYCLE_AFTER": "17",
            },
        ):
            source = FrameSource("video.mp4")
        self.assertEqual(source.num_threads, 3)
        self.assertEqual(source.recycle_after, 17)

        for kwargs in (
            {"num_threads": 0},
            {"num_threads": "bad"},
            {"recycle_after": 0},
            {"recycle_after": "bad"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    FrameSource("video.mp4", **kwargs)


if __name__ == "__main__":
    unittest.main()
