import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ragbot.observability.telemetry import build_summary, record_event


class ObservabilityTests(unittest.TestCase):
    def test_build_summary_aggregates_request_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("ragbot.observability.telemetry.OBSERVABILITY_DIR", Path(temp_dir)), patch(
                "ragbot.observability.telemetry.EVENTS_PATH",
                Path(temp_dir) / "events.jsonl",
            ):
                record_event(
                    "rag_query",
                    {
                        "total_latency_ms": 10.0,
                        "retrieval": {"retrieval_latency_ms": 4.0, "returned_chunks": 5},
                        "generation": {"generation_latency_ms": 6.0},
                    },
                )
                record_event(
                    "rag_query",
                    {
                        "total_latency_ms": 20.0,
                        "retrieval": {"retrieval_latency_ms": 7.0, "returned_chunks": 3},
                        "generation": {"generation_latency_ms": 13.0},
                    },
                )
                record_event("ingestion", {"chunks_created": 12, "dataset_hash": "abc"})

                summary = build_summary()

        self.assertEqual(2, summary["request_metrics"]["total_requests"])
        self.assertEqual(15.0, summary["request_metrics"]["avg_total_latency_ms"])
        self.assertEqual(20.0, summary["request_metrics"]["p95_total_latency_ms"])
        self.assertEqual(4.0, summary["request_metrics"]["avg_returned_chunks"])
        self.assertEqual("abc", summary["latest_ingestion"]["dataset_hash"])


if __name__ == "__main__":
    unittest.main()
