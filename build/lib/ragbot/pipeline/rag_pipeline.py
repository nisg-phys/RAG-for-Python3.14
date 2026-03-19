from langchain_groq import ChatGroq

from pydantic import SecretStr
from ragbot.retrievers.hybrid_retriever import HybridRetriever
from ragbot.vectorstore.pinecone_store import PineconeStore
from ragbot.config.settings import settings
from ragbot.prompts.rag_prompt import rag_prompt
from ragbot.utils.logger import get_logger


logger = get_logger("rag_pipeline")
"""This class is the core of the RAG pipeline. It initializes the vector store, retriever, and LLM, and defines the run() method which takes a query, retrieves relevant documents, builds a prompt, and gets a response from the LLM."""

class RAGPipeline:
    """
    Handles retrieval + LLM generation.
    """

    def __init__(self):

        # Vector store
        self.vectorstore = PineconeStore()

        self.documents=[]

        # Retriever
        #self.retriever = self.vectorstore.as_retriever()
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

        return str(response.content)