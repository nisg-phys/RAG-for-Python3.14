"""Tests for the Opik tracing helpers.

Fully hermetic: opik.track / opik_context.update_current_trace are mocked
out, so these never make a network call and never depend on OPIK_API_KEY
being set. They check the contract in ragbot.observability.opik_tracing:

- tracing can be switched off entirely via RAGBOT_ENABLE_OPIK=false, with
  @track becoming a true no-op (same function object back, zero overhead)
- update_trace_metadata never raises, even when there's no active trace to
  annotate (which is exactly what happens if it's ever called outside a
  @track-decorated function, or with tracing disabled) - annotating a trace
  must never be able to break the pipeline it's observing
"""

import importlib
import os
import unittest
from unittest.mock import patch


class OpikTracingToggleTests(unittest.TestCase):
    def _reload(self):
        # Settings and opik_tracing both cache the enabled flag at import
        # time, so both need reloading for an env var change to take effect.
        from ragbot.config import settings as settings_module
        importlib.reload(settings_module)
        from ragbot.observability import opik_tracing
        importlib.reload(opik_tracing)
        return opik_tracing

    def setUp(self):
        self._orig_enable = os.environ.get("RAGBOT_ENABLE_OPIK")

    def tearDown(self):
        if self._orig_enable is None:
            os.environ.pop("RAGBOT_ENABLE_OPIK", None)
        else:
            os.environ["RAGBOT_ENABLE_OPIK"] = self._orig_enable
        self._reload()

    def test_track_is_untouched_when_tracing_disabled(self):
        os.environ["RAGBOT_ENABLE_OPIK"] = "false"
        opik_tracing = self._reload()

        def add(a, b):
            return a + b

        decorated = opik_tracing.track(name="unit-test")(add)
        # Disabled tracing must hand back the exact same function object -
        # zero wrapping, zero overhead, zero dependency on Opik.
        self.assertIs(decorated, add)

    def test_langchain_tracer_is_none_when_tracing_disabled(self):
        os.environ["RAGBOT_ENABLE_OPIK"] = "false"
        opik_tracing = self._reload()
        self.assertIsNone(opik_tracing.get_langchain_tracer())

    def test_update_trace_metadata_is_a_no_op_when_tracing_disabled(self):
        os.environ["RAGBOT_ENABLE_OPIK"] = "false"
        opik_tracing = self._reload()
        with patch("opik.opik_context.update_current_trace") as mock_update:
            opik_tracing.update_trace_metadata(metadata={"anything": 1})
        mock_update.assert_not_called()

    def test_update_trace_metadata_swallows_missing_trace_error(self):
        os.environ["RAGBOT_ENABLE_OPIK"] = "true"
        opik_tracing = self._reload()
        from opik.exceptions import OpikException

        with patch(
            "opik.opik_context.update_current_trace",
            side_effect=OpikException("no trace"),
        ):
            # Must not propagate - this is exactly the exception Opik raises
            # when called with no active trace in context.
            opik_tracing.update_trace_metadata(metadata={"anything": 1})

    def test_track_delegates_to_opik_when_tracing_enabled(self):
        os.environ["RAGBOT_ENABLE_OPIK"] = "true"
        opik_tracing = self._reload()

        def add(a, b):
            return a + b

        with patch("opik.track") as mock_opik_track:
            mock_opik_track.return_value = lambda f: f
            decorated = opik_tracing.track(name="unit-test")(add)

        mock_opik_track.assert_called_once()
        _, kwargs = mock_opik_track.call_args
        self.assertEqual(kwargs["name"], "unit-test")
        self.assertEqual(kwargs["project_name"], opik_tracing.settings.opik_project_name)
        self.assertEqual(decorated(2, 3), 5)


if __name__ == "__main__":
    unittest.main()
