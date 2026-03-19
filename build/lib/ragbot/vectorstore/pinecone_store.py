from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from pydantic import SecretStr
from ragbot.config.settings import settings
from ragbot.utils.logger import get_logger

logger = get_logger("pinecone_store")
""""""
class PineconeStore:
    """
    Wrapper around Pinecone vector database.
    Handles index connection, embeddings, and retrieval.
    """

    def __init__(self):

        # Initialize Pinecone
        self.pc = Pinecone(api_key=settings.pinecone_api_key)

        # Connect to index
        self.index = self.pc.Index(settings.pinecone_index_name)

        # Embedding model
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key= SecretStr(settings.openai_api_key),
        )

        # LangChain vector store
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings
        )

    def add_documents(self, documents):
        """
        Add document chunks to the Pinecone index.
        """
        logger.info(f"Embedding {len(documents)} documents")
        self.vectorstore.add_documents(documents, batch_size=100)
        logger.info("Documents successfully stored in Pinecone")

    def similarity_search(self, query: str, k: int ):
        """
        Retrieve most similar documents.
        """
        k = settings.top_k
        logger.info(f"Vector search query: {query}")

        results = self.vectorstore.similarity_search(query, k=k)

        logger.info(f"Retrieved {len(results)} documents")

        return results
        
    def as_retriever(self):
        """
        Convert vectorstore into a LangChain retriever.
        """
        return self.vectorstore.as_retriever(
            search_kwargs={"k": settings.top_k}
        )