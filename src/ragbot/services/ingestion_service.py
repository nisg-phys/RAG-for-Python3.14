from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ragbot.pipeline.ingestion_pipeline import IngestionPipeline
from ragbot.utils.logger import get_logger
from ragbot.vectorstore.pinecone_store import PineconeStore

logger = get_logger("ingestion_service")
"""Loads documents from disk and drives the ingest/delete-index operations
exposed via the /ingest and /index API endpoints (see ragbot.api.routes)."""

DATA_DIR = "data"


class _FallbackTextLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        return [
            SimpleNamespace(
                page_content=Path(self.file_path).read_text(encoding="utf-8"),
                metadata={"source": self.file_path},
            )
        ]


class _MissingPDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        raise ImportError(
            "PyPDFLoader is unavailable. Install project dependencies to ingest PDF files."
        )


def _get_supported_loaders():
    try:
        from langchain_community.document_loaders import PyPDFLoader, TextLoader

        return {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
        }
    except ImportError:
        logger.warning(
            "langchain_community is unavailable; falling back to basic text loading only."
        )
        return {
            ".txt": _FallbackTextLoader,
            ".pdf": _MissingPDFLoader,
        }


def load_documents(data_dir: str = DATA_DIR):
    """Load documents from the data directory."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    supported_loaders = _get_supported_loaders()
    documents = []
    discovered_files = []

    for suffix in supported_loaders:
        files = sorted(path for path in root.rglob(f"*{suffix}") if path.is_file())
        discovered_files.extend(files)
        logger.info("Discovered %s %s files under %s", len(files), suffix, data_dir)

    for file_path in discovered_files:
        loader_cls = supported_loaders[file_path.suffix.lower()]
        logger.info("Loading document %s", file_path)
        loader = loader_cls(str(file_path))
        documents.extend(loader.load())

    return documents


def run_ingestion(data_dir: str = DATA_DIR) -> dict:
    """Load documents from `data_dir` and upsert new/changed ones into Pinecone.

    Documents whose content hasn't changed since the last run are skipped
    (no re-embedding, no re-upload); documents that changed or were removed
    have their stale vectors deleted from Pinecone. See
    IngestionPipeline.ingest for the comparison logic.

    Used by both `scripts/ingest.py` (CLI) and `POST /ingest` (API).
    """
    logger.info("Loading documents...")
    documents = load_documents(data_dir)
    logger.info(f"Loaded {len(documents)} documents")

    pipeline = IngestionPipeline()
    metrics = pipeline.ingest(documents)

    return {
        "documents_loaded": metrics.documents_loaded,
        "chunks_created": metrics.chunks_created,
        "chunks_deleted": metrics.chunks_deleted,
        "sources_new": metrics.sources_new,
        "sources_changed": metrics.sources_changed,
        "sources_unchanged": metrics.sources_unchanged,
        "sources_removed": metrics.sources_removed,
    }


def delete_index() -> None:
    """Delete every vector in the configured Pinecone index."""
    PineconeStore().delete_all()
