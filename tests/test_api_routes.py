"""Tests for the /query and /health API routes.

ragbot.api.routes instantiates a real RAGPipeline at import time (`rag =
RAGPipeline()`), which would otherwise open a real Pinecone connection and
real LLM clients. So ragbot.pipeline.rag_pipeline is replaced with a fake
module (exposing a fake RAGPipeline) in sys.modules before ragbot.main /
ragbot.api.routes are imported, mirroring the sys.modules-faking pattern in
tests/test_hybrid_retriever.py. This locks in the QueryResponse contract
(answer + sources + metrics) added when source citations were wired through
to the API.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient


class _FakeRAGPipeline:
    last_query = None

    def __init__(self):
        pass

    def run(self, query):
        _FakeRAGPipeline.last_query = query
        return {
            "answer": "## Answer\nLogging uses the logging module.",
            "retrieved_chunks": [],
            "sources": ["data/logging.rst", "data/typing.rst"],
            "metrics": {
                "trace_id": "abc123",
                "query": query,
                "rewriting_enabled": False,
                "total_latency_ms": 1.0,
                "retrieval": {
                    "requested_top_k": 5,
                    "returned_chunks": 2,
                    "unique_sources": 2,
                    "avg_vector_score": 0.5,
                    "avg_bm25_score": 0.5,
                    "avg_rrf_score": 0.5,
                    "retrieval_latency_ms": 0.5,
                },
                "generation": {
                    "prompt_chars": 100,
                    "answer_chars": 40,
                    "generation_latency_ms": 0.5,
                },
                "source_paths": ["data/logging.rst", "data/typing.rst"],
            },
        }

    def reload_documents(self):
        pass


def _import_app_with_fake_pipeline():
    fake_module = SimpleNamespace(RAGPipeline=_FakeRAGPipeline)
    sys.modules.pop("ragbot.main", None)
    sys.modules.pop("ragbot.api.routes", None)
    with patch.dict(sys.modules, {"ragbot.pipeline.rag_pipeline": fake_module}):
        from ragbot.main import app
    return app


class QueryRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _import_app_with_fake_pipeline()
        cls.client = TestClient(cls.app)

    def test_query_response_includes_answer_sources_and_metrics(self):
        response = self.client.post("/query", json={"query": "How is logging done in python?"})

        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual("## Answer\nLogging uses the logging module.", body["answer"])
        self.assertEqual(["data/logging.rst", "data/typing.rst"], body["sources"])
        self.assertIn("metrics", body)
        self.assertEqual("abc123", body["metrics"]["trace_id"])

    def test_query_route_forwards_the_question_to_the_pipeline(self):
        self.client.post("/query", json={"query": "what is a decorator?"})

        self.assertEqual("what is a decorator?", _FakeRAGPipeline.last_query)

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(200, response.status_code)
        self.assertEqual({"status": "ok"}, response.json())

    def test_query_requires_a_query_field(self):
        response = self.client.post("/query", json={})

        self.assertEqual(422, response.status_code)


if __name__ == "__main__":
    unittest.main()
