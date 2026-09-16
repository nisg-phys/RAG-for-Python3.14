import unittest

from ragbot.evaluation.metrics import compute_query_metrics, normalize_eval_specs, summarize_metrics


class EvaluationMetricsTests(unittest.TestCase):
    def test_compute_query_metrics_uses_deterministic_keyword_scoring(self):
        result = compute_query_metrics(
            query="What is a Python list?",
            answer="A list is ordered and mutable.",
            chunk_texts=[
                "Python list objects are ordered and mutable sequences.",
                "Tuples are immutable.",
            ],
            expected_keywords=["list", "ordered", "mutable"],
            pipeline_metrics={
                "total_latency_ms": 12.0,
                "retrieval": {"retrieval_latency_ms": 4.0},
                "generation": {"generation_latency_ms": 8.0},
            },
        )

        self.assertEqual(1.0, result["retrieval_hit_rate_at_k"])
        self.assertEqual(1.0, result["reciprocal_rank_at_k"])
        self.assertEqual(0.5, result["context_precision_at_k"])
        self.assertEqual(1.0, result["answer_keyword_recall"])

    def test_summarize_metrics_aggregates_query_results(self):
        summary = summarize_metrics(
            [
                {
                    "retrieval_hit_rate_at_k": 1.0,
                    "reciprocal_rank_at_k": 1.0,
                    "context_precision_at_k": 0.5,
                    "answer_keyword_recall": 1.0,
                    "answer_fallback_used": False,
                    "latency_ms": 10.0,
                    "retrieval_latency_ms": 4.0,
                    "generation_latency_ms": 6.0,
                },
                {
                    "retrieval_hit_rate_at_k": 0.0,
                    "reciprocal_rank_at_k": 0.0,
                    "context_precision_at_k": 0.0,
                    "answer_keyword_recall": 0.5,
                    "answer_fallback_used": True,
                    "latency_ms": 30.0,
                    "retrieval_latency_ms": 10.0,
                    "generation_latency_ms": 20.0,
                },
            ]
        )

        self.assertEqual(2, summary.queries_evaluated)
        self.assertEqual(0.5, summary.retrieval_hit_rate_at_k)
        self.assertEqual(0.5, summary.mean_reciprocal_rank_at_k)
        self.assertEqual(0.25, summary.mean_context_precision_at_k)
        self.assertEqual(0.75, summary.mean_answer_keyword_recall)
        self.assertEqual(0.5, summary.answer_fallback_rate)
        self.assertEqual(20.0, summary.avg_total_latency_ms)

    def test_normalize_eval_specs_supports_eval_data_py_shape(self):
        normalized = normalize_eval_specs(
            [
                {
                    "question": "What is the walrus operator?",
                    "ground_truth": "It assigns a value inside an expression using := syntax.",
                    "source": "python-docs",
                }
            ]
        )

        self.assertEqual("What is the walrus operator?", normalized[0]["query"])
        self.assertEqual("python-docs", normalized[0]["source"])
        self.assertEqual(
            "It assigns a value inside an expression using := syntax.",
            normalized[0]["ground_truth"],
        )
        self.assertIn("assigns", normalized[0]["expected_keywords"])
        self.assertIn("expression", normalized[0]["expected_keywords"])


if __name__ == "__main__":
    unittest.main()
