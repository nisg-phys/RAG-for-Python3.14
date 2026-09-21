"""Tests for ragbot.pipeline.rag_pipeline.RAGPipeline.

RAGPipeline.__init__ builds a real Pinecone connection and real LLM clients,
so these tests bypass __init__ (RAGPipeline.__new__) and wire up fakes for
just the collaborators each unit under test needs - the same pattern already
used in tests/test_hybrid_retriever.py. No network calls, no API keys
required beyond what importing ragbot.config.settings already needs from the
repo's .env.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ragbot.pipeline.rag_pipeline import RAGPipeline


def _result(chunk_id, source, page_content="text", vector_score=0.0, bm25_score=0.0, rrf_score=0.0):
    return {
        "doc": SimpleNamespace(page_content=page_content, metadata={"chunk_id": chunk_id, "source": source}),
        "vector_score": vector_score,
        "bm25_score": bm25_score,
        "rrf_score": rrf_score,
    }


class MergeRankedResultsTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = RAGPipeline.__new__(RAGPipeline)

    def test_doc_ranked_in_both_lists_outranks_single_list_doc(self):
        list_a = [_result("shared", "s1.rst"), _result("only_a", "s2.rst")]
        list_b = [_result("shared", "s1.rst"), _result("only_b", "s3.rst")]

        merged = self.pipeline._merge_ranked_results([list_a, list_b], top_k=3)

        self.assertEqual("shared", merged[0]["doc"].metadata["chunk_id"])

    def test_dedups_by_chunk_id_keeping_max_scores(self):
        list_a = [_result("dup", "s1.rst", vector_score=0.9, bm25_score=0.1)]
        list_b = [_result("dup", "s1.rst", vector_score=0.2, bm25_score=0.8)]

        merged = self.pipeline._merge_ranked_results([list_a, list_b], top_k=5)

        self.assertEqual(1, len(merged))
        self.assertEqual(0.9, merged[0]["vector_score"])
        self.assertEqual(0.8, merged[0]["bm25_score"])

    def test_respects_top_k(self):
        list_a = [_result(f"c{i}", "s.rst") for i in range(5)]

        merged = self.pipeline._merge_ranked_results([list_a], top_k=2)

        self.assertEqual(2, len(merged))


class RunSourcesAndFormattingTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = RAGPipeline.__new__(RAGPipeline)
        self.pipeline.use_query_rewriting = False
        self.pipeline._opik_tracer = None
        from ragbot.prompts.rag_prompt import rag_prompt

        self.pipeline.prompt = rag_prompt

    def test_run_returns_deduped_sources_and_formatted_answer(self):
        retrieved = [
            _result("c1", "data/logging.rst", page_content="chunk one"),
            _result("c2", "data/logging.rst", page_content="chunk two"),
            _result("c3", "data/typing.rst", page_content="chunk three"),
        ]
        self.pipeline.retriever = SimpleNamespace(retrieve=lambda query, k: retrieved)
        self.pipeline.llm = SimpleNamespace(
            invoke=lambda prompt, config=None: SimpleNamespace(
                content="Answer\nLogging uses the logging module."
            )
        )

        with patch("ragbot.pipeline.rag_pipeline.record_event"), patch(
            "ragbot.pipeline.rag_pipeline.update_trace_metadata"
        ):
            result = self.pipeline.run("How is logging done in python?")

        self.assertEqual(["data/logging.rst", "data/typing.rst"], result["sources"])
        self.assertIn("## Answer", result["answer"])

    def test_run_raises_when_no_documents_ingested(self):
        self.pipeline.retriever = None

        with self.assertRaises(RuntimeError):
            self.pipeline.run("anything")


if __name__ == "__main__":
    unittest.main()
