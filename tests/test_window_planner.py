from __future__ import annotations

import unittest

from lib.pipeline.window_planner import compute_windows, processing_windows


class TemporalWindowPlannerTests(unittest.TestCase):

    def _safe_windows(self, n_frames: int):
        owned = compute_windows(
            n_frames,
            chunk_frames=3000,
            min_last_frames=1500,
            max_chunk_frames=3001,
            left_context_frames=16,
            right_context_frames=16,
            alignment_frames=16,
        )
        processed = processing_windows(
            owned,
            n_frames,
            left_context_frames=16,
            right_context_frames=16,
        )
        return owned, processed

    def test_short_video_remains_one_unchanged_window(self):
        owned, processed = self._safe_windows(1823)
        self.assertEqual(owned, [(0, 1823)])
        self.assertEqual(processed, owned)

    def test_long_windows_are_aligned_contextual_and_capped(self):
        n_frames = 12000
        owned, processed = self._safe_windows(n_frames)

        self.assertEqual(owned[0][0], 0)
        self.assertEqual(owned[-1][1], n_frames)
        self.assertTrue(all(left[1] == right[0] for left, right in zip(owned, owned[1:])))
        self.assertTrue(all(end % 16 == 0 for _, end in owned[:-1]))
        self.assertTrue(all(end - start <= 3001 for start, end in processed))

        for index, ((owned_start, owned_end), (process_start, process_end)) in enumerate(
            zip(owned, processed)
        ):
            expected_start = 0 if index == 0 else owned_start - 16
            expected_end = n_frames if index == len(owned) - 1 else owned_end + 16
            self.assertEqual((process_start, process_end), (expected_start, expected_end))

    def test_just_over_cap_rebalances_without_a_tiny_tail(self):
        owned, processed = self._safe_windows(3039)
        self.assertEqual(owned, [(0, 1536), (1536, 3039)])
        self.assertGreaterEqual(owned[-1][1] - owned[-1][0], 1500)
        self.assertTrue(all(end - start <= 3001 for start, end in processed))

    def test_many_video_lengths_preserve_all_window_invariants(self):
        for n_frames in range(1, 15000, 37):
            with self.subTest(n_frames=n_frames):
                owned, processed = self._safe_windows(n_frames)
                self.assertEqual(owned[0][0], 0)
                self.assertEqual(owned[-1][1], n_frames)
                self.assertTrue(
                    all(
                        left[1] == right[0]
                        for left, right in zip(owned, owned[1:])
                    )
                )
                self.assertTrue(
                    all(end % 16 == 0 for _, end in owned[:-1])
                )
                self.assertTrue(
                    all(0 < end - start <= 3001 for start, end in processed)
                )

    def test_historical_defaults_retain_one_frame_right_overlap(self):
        owned = compute_windows(5000, 3000, 0, max_chunk_frames=3001)
        processed = processing_windows(owned, 5000)
        self.assertEqual(owned, [(0, 3000), (3000, 5000)])
        self.assertEqual(processed, [(0, 3001), (3000, 5000)])

    def test_rejects_context_that_cannot_fit(self):
        with self.assertRaisesRegex(ValueError, "too small"):
            compute_windows(
                100,
                100,
                0,
                max_chunk_frames=32,
                left_context_frames=16,
                right_context_frames=16,
                alignment_frames=16,
            )


if __name__ == "__main__":
    unittest.main()
