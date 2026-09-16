import re
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class _FakeBM25Okapi:
    def __init__(self, *args, **kwargs):
        pass


class HybridRetrieverRerankTests(unittest.TestCase):
    def test_rerank_prefers_higher_query_overlap(self):
        with patch.dict(
            sys.modules,
            {
                "rank_bm25": SimpleNamespace(BM25Okapi=_FakeBM25Okapi),
                "ragbot.config.settings": SimpleNamespace(
                    settings=SimpleNamespace(retrieval_candidate_pool=12)
                ),
            },
        ):
            from ragbot.retrievers.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.token_pattern = re.compile(r"[a-zA-Z0-9_]+")

        results = [
            {
                "doc": SimpleNamespace(
                    page_content="Pattern matching uses match and case statements.",
                    metadata={"chunk_id": "a"},
                ),
                "rrf_score": 0.0200,
                "bm25_score": 1.5,
                "vector_score": 0.1,
            },
            {
                "doc": SimpleNamespace(
                    page_content="General overview of Python data structures.",
                    metadata={"chunk_id": "b"},
                ),
                "rrf_score": 0.0210,
                "bm25_score": 0.5,
                "vector_score": 0.2,
            },
        ]

        reranked = retriever._rerank_results(
            "How does pattern matching work in Python?",
            results,
            k=2,
        )

        self.assertEqual("a", reranked[0]["doc"].metadata["chunk_id"])
        self.assertGreater(reranked[0]["lexical_overlap"], reranked[1]["lexical_overlap"])
        self.assertGreater(reranked[0]["rrf_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
