from langchain_core.documents import Document
from ragbot.pipeline.ingestion_pipeline import IngestionPipeline

docs = [
    Document(page_content="This is a test document.")
]

pipeline = IngestionPipeline()
pipeline.ingest(docs)