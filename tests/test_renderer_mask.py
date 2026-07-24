from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from lib.vis.renderer import Renderer


class RendererMaskTests(unittest.TestCase):

    def test_mask_only_path_matches_render_multiple_mask(self):
        rgba = torch.tensor(
            [
                [
                    [[0.1, 0.2, 0.3, 0.0], [0.4, 0.5, 0.6, 0.25]],
                    [[0.7, 0.8, 0.9, -0.1], [1.0, 0.0, 0.5, 1.0]],
                ]
            ],
            dtype=torch.float32,
        )
        renderer = object.__new__(Renderer)

        with mock.patch.object(
            renderer, "_render_multiple_rgba", return_value=rgba
        ) as render_rgba:
            mask_only = renderer.render_mask(
                "verts", "faces", "colors", "cameras", "lights"
            )
            _, legacy_mask = renderer.render_multiple(
                "verts", "faces", "colors", "cameras", "lights"
            )

        self.assertTrue(np.array_equal(mask_only, legacy_mask))
        self.assertEqual(mask_only.dtype, np.bool_)
        self.assertTrue(
            np.array_equal(
                mask_only,
                np.array([[False, True], [False, True]], dtype=np.bool_),
            )
        )
        self.assertEqual(render_rgba.call_count, 2)

    def test_mask_only_path_reuses_host_buffer(self):
        rgba = torch.tensor(
            [[[[0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 1.0]]]],
            dtype=torch.float32,
        )
        renderer = object.__new__(Renderer)
        host_buffer = torch.empty((1, 2), dtype=torch.bool)

        with mock.patch.object(
            renderer, "_render_multiple_rgba", return_value=rgba
        ):
            mask = renderer.render_mask(
                "verts",
                "faces",
                "colors",
                "cameras",
                "lights",
                host_buffer=host_buffer,
            )

        self.assertTrue(np.shares_memory(mask, host_buffer.numpy()))
        self.assertTrue(
            np.array_equal(mask, np.array([[False, True]], dtype=np.bool_))
        )

    def test_mask_only_path_rejects_incompatible_host_buffer(self):
        rgba = torch.zeros((1, 1, 2, 4), dtype=torch.float32)
        renderer = object.__new__(Renderer)

        with mock.patch.object(
            renderer, "_render_multiple_rgba", return_value=rgba
        ):
            with self.assertRaisesRegex(ValueError, "CPU bool tensor"):
                renderer.render_mask(
                    "verts",
                    "faces",
                    "colors",
                    "cameras",
                    "lights",
                    host_buffer=torch.empty((1, 2), dtype=torch.float32),
                )

    def test_mask_threshold_is_bit_identical_on_host_and_device(self):
        alpha = torch.tensor(
            [
                [float("-inf"), -1.0, -0.0],
                [0.0, torch.finfo(torch.float32).tiny, float("inf")],
            ],
            dtype=torch.float32,
        )
        host_mask = alpha.cpu().numpy() > 0
        device_mask = (alpha > 0).cpu().numpy()

        self.assertTrue(np.array_equal(host_mask, device_mask))


if __name__ == "__main__":
    unittest.main()
