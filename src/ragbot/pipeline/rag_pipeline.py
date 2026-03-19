import pickle
from urllib import response
from pathlib import Path
from langchain_groq import ChatGroq
from ragbot.ingestion.s3_storage import download_chunks

from pydantic import SecretStr
from ragbot.pipeline.ingestion_pipeline import IngestionPipeline
from ragbot.retrievers.hybrid_retriever import HybridRetriever
from ragbot.vectorstore.pinecone_store import PineconeStore
from ragbot.config.settings import settings
from ragbot.prompts.rag_prompt import rag_prompt
from ragbot.utils.logger import get_logger

from scripts.ingest import DATA_DIR, load_documents
from ragbot.utils.formatter import format_to_markdown


logger = get_logger("rag_pipeline")
"""This class is the core of the RAG pipeline. It initializes the vector store, retriever, and LLM, and defines the run() method which takes a query, retrieves relevant documents, builds a prompt, and gets a response from the LLM."""
def load_chunks():
    path = "storage/chunks.pkl"

    if not Path(path).exists():
        raise RuntimeError(
            "Chunks not found. Run ingestion first: python scripts/ingest.py"
        )

    with open(path, "rb") as f:
        return pickle.load(f)

class RAGPipeline:
    """
    Handles retrieval + LLM generation.
    """

    def __init__(self):

        # Vector store
        self.vectorstore = PineconeStore()

        # --- LOAD PERSISTED CHUNKS ---
        chunks = download_chunks()

        if chunks is None:
            raise RuntimeError("Chunks not found in S3. Run ingestion first.")
        self.documents= chunks
        logger.info(f"Loaded {len(chunks)} chunks for hybrid retrieval")
        
        self.retriever = HybridRetriever(
            self.vectorstore,
            self.documents
        )

        # Groq LLM
        self.llm = ChatGroq(
            api_key=SecretStr(settings.groq_api_key),
            model=settings.llm_model
        )

        # Prompt template
        self.prompt  = rag_prompt

    def run(self, query: str)-> str:

        logger.info(f"Received query: {query}")

        # Retrieve documents
        docs = self.retriever.retrieve(query)

        logger.info(f"{len(docs)} documents retrieved after hybrid merge")
        # Combine context
        sources = [doc.metadata.get("source", "unknown") for doc in docs]
        logger.info(f"Retrieved sources: {sources}")

        context = "\n\n".join([d.page_content for d in docs])

        logger.info("Context constructed")
        # Build prompt
        prompt = self.prompt.invoke(
            {
                "context": context,
                "question": query
            }
        )

        # Call LLM
        logger.info("Calling LLM with constructed prompt")
        response = self.llm.invoke(prompt)

        logger.info("LLM response generated")
          # Remove newlines
        clean_response = str(response.content).replace('\n', ' ')
        clean_response = format_to_markdown(clean_response)

        return clean_response
        

       