import unittest

from lib.pipeline.droid_slam_fallback import run_with_unmasked_fallback


EMPTY_GRAPH_MESSAGE = "not enough values to unpack (expected 2, got 0)"


def raise_backend_empty_graph(
    message=EMPTY_GRAPH_MESSAGE,
    source_directory="/tmp/droid_slam",
):
    factor_namespace = {}
    exec(
        compile(
            "def add_proximity_factors():\n"
            f"    raise ValueError({message!r})\n",
            f"{source_directory}/factor_graph.py",
            "exec",
        ),
        factor_namespace,
    )

    backend_namespace = {
        "add_proximity_factors": factor_namespace["add_proximity_factors"]
    }
    exec(
        compile(
            "def __call__():\n"
            "    add_proximity_factors()\n",
            f"{source_directory}/droid_backend.py",
            "exec",
        ),
        backend_namespace,
    )
    backend_namespace["__call__"]()


class DroidSlamFallbackTest(unittest.TestCase):
    def test_successful_primary_does_not_run_fallback(self):
        events = []

        result = run_with_unmasked_fallback(
            primary=lambda: ("primary-droid", "primary-traj"),
            fallback=lambda: events.append("fallback"),
            cleanup=lambda: events.append("cleanup"),
        )

        self.assertEqual(result, ("primary-droid", "primary-traj"))
        self.assertEqual(events, [])

    def test_exact_backend_empty_graph_error_runs_one_fallback(self):
        events = []

        result = run_with_unmasked_fallback(
            primary=raise_backend_empty_graph,
            fallback=lambda: (
                events.append("fallback"),
                ("fallback-droid", "fallback-traj"),
            )[1],
            cleanup=lambda: events.append("cleanup"),
        )

        self.assertEqual(result, ("fallback-droid", "fallback-traj"))
        self.assertEqual(events, ["cleanup", "fallback"])

    def test_same_message_from_another_origin_is_not_caught(self):
        fallback_calls = []

        def primary():
            raise ValueError(EMPTY_GRAPH_MESSAGE)

        with self.assertRaises(ValueError) as caught:
            run_with_unmasked_fallback(
                primary=primary,
                fallback=lambda: fallback_calls.append(True),
                cleanup=lambda: None,
            )

        self.assertEqual(str(caught.exception), EMPTY_GRAPH_MESSAGE)
        self.assertEqual(fallback_calls, [])

    def test_different_message_from_backend_is_not_caught(self):
        with self.assertRaisesRegex(ValueError, "different failure"):
            run_with_unmasked_fallback(
                primary=lambda: raise_backend_empty_graph("different failure"),
                fallback=lambda: self.fail("fallback must not run"),
                cleanup=lambda: self.fail("cleanup must not run"),
            )

    def test_matching_frame_names_outside_droid_slam_are_not_caught(self):
        with self.assertRaises(ValueError) as caught:
            run_with_unmasked_fallback(
                primary=lambda: raise_backend_empty_graph(
                    source_directory="/tmp/foreign"
                ),
                fallback=lambda: self.fail("fallback must not run"),
                cleanup=lambda: self.fail("cleanup must not run"),
            )

        self.assertEqual(str(caught.exception), EMPTY_GRAPH_MESSAGE)

    def test_non_value_error_is_not_caught(self):
        error = RuntimeError("later trajectory failure")

        def primary():
            raise error

        with self.assertRaises(RuntimeError) as caught:
            run_with_unmasked_fallback(
                primary=primary,
                fallback=lambda: self.fail("fallback must not run"),
                cleanup=lambda: self.fail("cleanup must not run"),
            )

        self.assertIs(caught.exception, error)

    def test_fallback_failure_propagates_without_retry(self):
        events = []
        fallback_error = RuntimeError("unmasked fallback failed")

        def fallback():
            events.append("fallback")
            raise fallback_error

        with self.assertRaises(RuntimeError) as caught:
            run_with_unmasked_fallback(
                primary=raise_backend_empty_graph,
                fallback=fallback,
                cleanup=lambda: events.append("cleanup"),
            )

        self.assertIs(caught.exception, fallback_error)
        self.assertEqual(events, ["cleanup", "fallback"])

    def test_second_empty_graph_error_runs_terminal_fallback(self):
        events = []

        def fallback():
            events.append("fallback")
            raise_backend_empty_graph()

        result = run_with_unmasked_fallback(
            primary=raise_backend_empty_graph,
            fallback=fallback,
            cleanup=lambda: events.append("cleanup"),
            terminal_fallback=lambda: (
                events.append("terminal"),
                (None, "constant-traj"),
            )[1],
        )

        self.assertEqual(result, (None, "constant-traj"))
        self.assertEqual(
            events,
            ["cleanup", "fallback", "cleanup", "terminal"],
        )

    def test_second_empty_graph_error_propagates_without_terminal_fallback(self):
        events = []

        def fallback():
            events.append("fallback")
            raise_backend_empty_graph()

        with self.assertRaisesRegex(ValueError, "not enough values to unpack"):
            run_with_unmasked_fallback(
                primary=raise_backend_empty_graph,
                fallback=fallback,
                cleanup=lambda: events.append("cleanup"),
            )

        self.assertEqual(events, ["cleanup", "fallback"])

    def test_unrelated_second_value_error_does_not_run_terminal_fallback(self):
        events = []

        with self.assertRaisesRegex(ValueError, "different failure"):
            run_with_unmasked_fallback(
                primary=raise_backend_empty_graph,
                fallback=lambda: raise_backend_empty_graph("different failure"),
                cleanup=lambda: events.append("cleanup"),
                terminal_fallback=lambda: events.append("terminal"),
            )

        self.assertEqual(events, ["cleanup"])


if __name__ == "__main__":
    unittest.main()
