import hashlib
import pickle
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbot.config.settings import settings
from ragbot.ingestion.s3_storage import download_chunks, upload_chunks
from ragbot.observability.models import IngestionMetrics
from ragbot.observability.opik_tracing import track, update_trace_metadata
from ragbot.observability.telemetry import record_event
from ragbot.utils.logger import get_logger
from ragbot.vectorstore.pinecone_store import PineconeStore

logger = get_logger("ingestion_pipeline")
"""This class is get called by scripts/ingest.py to handle document chunking and ingestion into Pinecone. In return this module calls the PineconeStore class from ragbot.vectorstore.pinecone_store. It uses RecursiveCharacterTextSplitter for chunking and PineconeStore for storage."""
class IngestionPipeline:
    """
    Handles document chunking and ingestion into Pinecone.
    """

    def __init__(self):

        self.vectorstore = PineconeStore()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
    def _make_chunk_id(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _source_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Adding dataset fingerprint to prevent silent desync again.

    def _dataset_hash(self, chunks):
        combined = "".join([c.page_content for c in chunks])
        return hashlib.sha256(combined.encode()).hexdigest()

    def _group_by_source(self, documents):
        grouped: dict[str, list] = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            grouped.setdefault(source, []).append(doc)
        return grouped

    @track(name="ingestion.ingest", tags=["ingestion"], capture_input=False, capture_output=False)
    def ingest(self, documents):
        """
        Split new/changed documents into chunks and upsert them into Pinecone.

        Documents are compared to the previously ingested corpus by a
        per-source content hash: unchanged sources are skipped entirely (no
        re-chunking, no re-embedding, no upsert). Sources whose content
        changed have their old chunks deleted from Pinecone by id and
        replaced; sources that disappeared from the input entirely (deleted
        files) have their old chunks deleted too.
        """
        logger.info(f"Starting ingestion for {len(documents)} documents")

        existing_chunks = download_chunks() or []
        existing_by_source: dict[str, list] = {}
        existing_hash_by_source: dict[str, str] = {}
        for chunk in existing_chunks:
            source = chunk.metadata.get("source", "unknown")
            existing_by_source.setdefault(source, []).append(chunk)
            source_hash = chunk.metadata.get("source_hash")
            if source_hash:
                existing_hash_by_source[source] = source_hash

        grouped_documents = self._group_by_source(documents)
        current_sources = set(grouped_documents.keys())
        removed_sources = set(existing_by_source.keys()) - current_sources

        new_sources, changed_sources, unchanged_sources = [], [], []
        source_hashes = {}
        for source, docs in grouped_documents.items():
            source_hash = self._source_hash("".join(doc.page_content for doc in docs))
            source_hashes[source] = source_hash

            if source not in existing_hash_by_source:
                new_sources.append(source)
            elif existing_hash_by_source[source] != source_hash:
                changed_sources.append(source)
            else:
                unchanged_sources.append(source)

        logger.info(
            "Sources: %s new, %s changed, %s unchanged, %s removed",
            len(new_sources), len(changed_sources), len(unchanged_sources), len(removed_sources),
        )

        # Drop stale vectors for anything that changed content or disappeared.
        ids_to_delete = [
            chunk.metadata.get("chunk_id")
            for source in list(changed_sources) + list(removed_sources)
            for chunk in existing_by_source.get(source, [])
            if chunk.metadata.get("chunk_id")
        ]
        if ids_to_delete:
            self.vectorstore.delete_ids(ids_to_delete)

        # Only (re-)chunk and embed sources that are new or changed.
        documents_to_process = [
            doc
            for source in new_sources + changed_sources
            for doc in grouped_documents[source]
        ]
        new_chunks = self.text_splitter.split_documents(documents_to_process)

        ids = []
        for c in new_chunks:
            cid = self._make_chunk_id(c.page_content)
            c.metadata["chunk_id"] = cid
            c.metadata["source_hash"] = source_hashes[c.metadata.get("source", "unknown")]
            ids.append(cid)

        if new_chunks:
            logger.info(f"Upserting {len(ids)} chunks into Pinecone (idempotent expected)...")
            self.vectorstore.add_documents(documents=new_chunks, ids=ids)
            index = self.vectorstore.index
            stats = index.describe_index_stats()
            logger.info("Pinecone index stats after upsert: %s", stats)
        else:
            logger.info("No new or changed documents; nothing to embed or upsert")

        # Full corpus = untouched chunks from unchanged sources + freshly embedded chunks.
        retained_chunks = [
            chunk for source in unchanged_sources for chunk in existing_by_source[source]
        ]
        all_chunks = retained_chunks + new_chunks

        dataset_hash = self._dataset_hash(all_chunks)

        #Persist the full chunk corpus for BM25 retrieval
        Path("storage").mkdir(exist_ok=True)
        with open("storage/chunks.pkl", "wb") as f:
            pickle.dump(all_chunks, f)
        logger.info("Chunks persisted to storage/chunks.pkl")

        upload_chunks(all_chunks)
        logger.info("Chunks uploaded to S3")

        avg_chunk_length = (
            round(sum(len(chunk.page_content) for chunk in new_chunks) / len(new_chunks), 4)
            if new_chunks
            else 0.0
        )
        metrics = IngestionMetrics(
            files_discovered=len(current_sources),
            documents_loaded=len(documents),
            chunks_created=len(new_chunks),
            chunks_deleted=len(ids_to_delete),
            chunks_reused=len(retained_chunks),
            chunks_total=len(all_chunks),
            sources_new=len(new_sources),
            sources_changed=len(changed_sources),
            sources_unchanged=len(unchanged_sources),
            sources_removed=len(removed_sources),
            avg_chunks_per_document=(
                round(len(new_chunks) / len(documents_to_process), 4) if documents_to_process else 0.0
            ),
            avg_chunk_length_chars=avg_chunk_length,
            dataset_hash=dataset_hash,
        )
        record_event("ingestion", metrics)
        logger.info("Ingestion metrics recorded: %s", metrics.model_dump())
        update_trace_metadata(metadata=metrics.model_dump())

        return metrics

