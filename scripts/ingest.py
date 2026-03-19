import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_core.documents import Document

from ragbot.pipeline.ingestion_pipeline import IngestionPipeline
from ragbot.utils.logger import get_logger

logger = get_logger("document_loader")
""" This script loads documents from the folder specified by DATA_DIR, process them  using the IngestionPipeline class in pipeline/ingestion_pipeline.py, and stores the processed chunks in Pinecone. """

DATA_DIR = "data"


def load_documents(data_dir: str):
    """Load documents from the data directory."""

    documents = []
     #LOAD DOCUMENTS FROM .TXT FILES
    logger.info(f"Loading .txt documents from {data_dir}...")
    txt_loader= DirectoryLoader(DATA_DIR, glob="*txt", loader_cls= TextLoader)
    txt_documents = txt_loader.load()

    #LOAD DOCUMENTS FROM .PDF FILES
    logger.info(f"Loading .pdf documents from {data_dir}...")
    pdf_loader = DirectoryLoader(DATA_DIR, glob="*.pdf", loader_cls= PyPDFLoader)
    pdf_docs = pdf_loader.load()

    documents = txt_documents + pdf_docs
    return documents


def main():

    logger.info("Loading documents...")

    documents = load_documents(DATA_DIR)

    logger.info(f"Loaded {len(documents)} documents")

    pipeline = IngestionPipeline()
    #Chunks are created for the 
    keyword_docs= pipeline.ingest(documents)

if __name__ == "__main__":
    main()