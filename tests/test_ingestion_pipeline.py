"""Tests for ragbot.pipeline.ingestion_pipeline.IngestionPipeline.ingest.

IngestionPipeline.__init__ builds a real PineconeStore, so these tests
bypass it (IngestionPipeline.__new__) and inject a fake vectorstore plus a
no-op text splitter, following the same bypass-__init__ pattern used in
tests/test_hybrid_retriever.py and tests/test_rag_pipeline.py. S3 chunk
persistence (download_chunks/upload_chunks) is mocked, and the test runs
inside a temp cwd so the "storage/chunks.pkl" side-effect write doesn't
touch the real repo.

These exercise the incremental-ingestion diffing logic: new / changed /
unchanged / removed sources are detected by comparing per-source content
hashes against the previously persisted chunk corpus, and only new/changed
sources are re-chunked, re-embedded, and upserted.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from ragbot.pipeline.ingestion_pipeline import IngestionPipeline


class _FakeVectorstore:
    def __init__(self):
        self.deleted_ids = []
        self.added_documents = None
        self.added_ids = None
        self.index = SimpleNamespace(describe_index_stats=lambda: {"totalVectorCount": 0})

    def delete_ids(self, ids):
        self.deleted_ids.extend(ids)

    def add_documents(self, documents, ids=None):
        self.added_documents = documents
        self.added_ids = ids


def _make_pipeline():
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline.vectorstore = _FakeVectorstore()
    # Identity splitter: keep the diffing-logic tests independent of
    # RecursiveCharacterTextSplitter's chunking behavior, which isn't what's
    # under test here.
    pipeline.text_splitter = SimpleNamespace(split_documents=lambda docs: list(docs))
    return pipeline


class IngestionPipelineDiffTests(unittest.TestCase):
    def setUp(self):
        self._orig_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        self._tmp.cleanup()

    def test_first_ingestion_treats_all_sources_as_new(self):
        pipeline = _make_pipeline()
        documents = [Document(page_content="A", metadata={"source": "a.txt"})]

        with patch("ragbot.pipeline.ingestion_pipeline.download_chunks", return_value=None), patch(
            "ragbot.pipeline.ingestion_pipeline.upload_chunks"
        ) as mock_upload, patch("ragbot.pipeline.ingestion_pipeline.record_event"), patch(
            "ragbot.pipeline.ingestion_pipeline.update_trace_metadata"
        ):
            metrics = pipeline.ingest(documents)

        self.assertEqual(1, metrics.sources_new)
        self.assertEqual(0, metrics.sources_changed)
        self.assertEqual(0, metrics.sources_unchanged)
        self.assertEqual(0, metrics.sources_removed)
        self.assertEqual(1, metrics.chunks_created)
        self.assertEqual(0, metrics.chunks_deleted)
        self.assertEqual(1, len(pipeline.vectorstore.added_documents))
        mock_upload.assert_called_once()

    def test_unchanged_source_is_retained_without_reembedding(self):
        pipeline = _make_pipeline()
        existing_hash = pipeline._source_hash("A")
        existing_chunks = [
            Document(
                page_content="A",
                metadata={"source": "a.txt", "chunk_id": "existing-id", "source_hash": existing_hash},
            )
        ]
        documents = [Document(page_content="A", metadata={"source": "a.txt"})]

        with patch(
            "ragbot.pipeline.ingestion_pipeline.download_chunks", return_value=existing_chunks
        ), patch("ragbot.pipeline.ingestion_pipeline.upload_chunks"), patch(
            "ragbot.pipeline.ingestion_pipeline.record_event"
        ), patch("ragbot.pipeline.ingestion_pipeline.update_trace_metadata"):
            metrics = pipeline.ingest(documents)

        self.assertEqual(0, metrics.sources_new)
        self.assertEqual(0, metrics.sources_changed)
        self.assertEqual(1, metrics.sources_unchanged)
        self.assertEqual(0, metrics.chunks_created)
        self.assertIsNone(pipeline.vectorstore.added_documents)

    def test_changed_source_is_reembedded_and_old_chunk_deleted(self):
        pipeline = _make_pipeline()
        stale_hash = pipeline._source_hash("OLD CONTENT")
        existing_chunks = [
            Document(
                page_content="OLD CONTENT",
                metadata={"source": "a.txt", "chunk_id": "stale-id", "source_hash": stale_hash},
            )
        ]
        documents = [Document(page_content="NEW CONTENT", metadata={"source": "a.txt"})]

        with patch(
            "ragbot.pipeline.ingestion_pipeline.download_chunks", return_value=existing_chunks
        ), patch("ragbot.pipeline.ingestion_pipeline.upload_chunks"), patch(
            "ragbot.pipeline.ingestion_pipeline.record_event"
        ), patch("ragbot.pipeline.ingestion_pipeline.update_trace_metadata"):
            metrics = pipeline.ingest(documents)

        self.assertEqual(0, metrics.sources_new)
        self.assertEqual(1, metrics.sources_changed)
        self.assertEqual(1, metrics.chunks_created)
        self.assertEqual(1, metrics.chunks_deleted)
        self.assertIn("stale-id", pipeline.vectorstore.deleted_ids)

    def test_source_missing_from_input_is_removed(self):
        pipeline = _make_pipeline()
        existing_chunks = [
            Document(
                page_content="GONE",
                metadata={"source": "removed.txt", "chunk_id": "gone-id", "source_hash": "irrelevant"},
            )
        ]

        with patch(
            "ragbot.pipeline.ingestion_pipeline.download_chunks", return_value=existing_chunks
        ), patch("ragbot.pipeline.ingestion_pipeline.upload_chunks"), patch(
            "ragbot.pipeline.ingestion_pipeline.record_event"
        ), patch("ragbot.pipeline.ingestion_pipeline.update_trace_metadata"):
            metrics = pipeline.ingest([])

        self.assertEqual(1, metrics.sources_removed)
        self.assertEqual(1, metrics.chunks_deleted)
        self.assertIn("gone-id", pipeline.vectorstore.deleted_ids)


if __name__ == "__main__":
    unittest.main()
