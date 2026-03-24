import hashlib
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragbot.ingestion.s3_storage import upload_chunks
from ragbot.vectorstore.pinecone_store import PineconeStore
from ragbot.config.settings import settings
from ragbot.utils.logger import get_logger
import pickle

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

    # Adding dataset fingerprint to prevent silent desync again. 
    
    def _dataset_hash(self, chunks):
        combined = "".join([c.page_content for c in chunks])
        return hashlib.sha256(combined.encode()).hexdigest()

    def ingest(self, documents):
        """
        Split documents into chunks and store them in Pinecone.
        """
        logger.info(f"Starting ingestion for {len(documents)} documents")


        chunks = self.text_splitter.split_documents(documents)

        ids = []
        for c in chunks:
            cid = self._make_chunk_id(c.page_content)
            c.metadata["chunk_id"] = cid   # ← ADD THIS
            ids.append(cid)

        # Dataset fingerprint
        dataset_hash = self._dataset_hash(chunks)
        for c in chunks:
            c.metadata["dataset_hash"] = dataset_hash

        logger.info(f"Upserting {len(ids)} chunks into Pinecone (idempotent expected)...")    

        self.vectorstore.add_documents(documents=chunks, ids=ids)
        index= self.vectorstore.index
        stats= index.describe_index_stats()
        print(stats)

        logger.info(f"Created {len(chunks)} chunks")

        #Persist the chunks for BM25 retrieval
        Path("storage").mkdir(exist_ok=True)
        with open("storage/chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
        logger.info("Chunks persisted to storage/chunks.pkl")

        upload_chunks(chunks)
        logger.info("Chunks uploaded to S3")


        return chunks

    