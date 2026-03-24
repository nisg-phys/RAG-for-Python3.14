import pickle
from pathlib import Path
from langchain_groq import ChatGroq
from ragbot.ingestion.s3_storage import download_chunks
from fastapi.responses import PlainTextResponse

from pydantic import SecretStr
from ragbot.pipeline.ingestion_pipeline import IngestionPipeline
from ragbot.retrievers.hybrid_retriever import HybridRetriever
from ragbot.retrievers.query_rewriter import QueryRewriter
from ragbot.vectorstore.pinecone_store import PineconeStore
from ragbot.config.settings import settings
from ragbot.prompts.rag_prompt import rag_prompt
from ragbot.utils.logger import get_logger
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

        self.query_rewriter = QueryRewriter(self.llm)
        self.use_query_rewriting = False

        # Prompt template
        self.prompt  = rag_prompt

    def _merge_ranked_results(self, ranked_results, top_k=5):
        scores = {}
        docs_by_chunk_id = {}
        vector_scores = {}
        bm25_scores = {}

        for results in ranked_results:
            for rank, result in enumerate(results):
                doc = result["doc"]
                chunk_id = doc.metadata.get("chunk_id", doc.page_content)
                docs_by_chunk_id[chunk_id] = doc
                vector_scores[chunk_id] = max(
                    vector_scores.get(chunk_id, float("-inf")),
                    float(result.get("vector_score", 0.0)),
                )
                bm25_scores[chunk_id] = max(
                    bm25_scores.get(chunk_id, float("-inf")),
                    float(result.get("bm25_score", 0.0)),
                )
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rank + 60)

        sorted_chunk_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
        return [
            {
                "doc": docs_by_chunk_id[chunk_id],
                "vector_score": 0.0 if vector_scores.get(chunk_id, float("-inf")) == float("-inf") else vector_scores[chunk_id],
                "bm25_score": 0.0 if bm25_scores.get(chunk_id, float("-inf")) == float("-inf") else bm25_scores[chunk_id],
                "rrf_score": scores[chunk_id],
            }
            for chunk_id in sorted_chunk_ids[:top_k]
        ]

    def _retrieve_documents(self, query: str, top_k=5):
        if not self.use_query_rewriting:
            return self.retriever.retrieve(query, k=top_k)

        rewritten_queries = self.query_rewriter.rewrite(query)
        all_queries = [query] + rewritten_queries
        logger.info("Using query rewriting with %s total queries", len(all_queries))

        ranked_results = []
        for index, current_query in enumerate(all_queries):
            current_top_k = top_k if index == 0 else 3
            logger.info(
                "Retrieving for query variant %s/%s: '%s' with k=%s",
                index + 1,
                len(all_queries),
                current_query,
                current_top_k,
            )
            ranked_results.append(self.retriever.retrieve(current_query, k=current_top_k))

        return self._merge_ranked_results(ranked_results, top_k=top_k)

    def run(self, query: str)-> str:

        logger.info(f"Received query: {query}")

        # Retrieve documents
        results = self._retrieve_documents(query)

        logger.info(f"{len(results)} documents retrieved after hybrid merge")
        # Combine context
        sources = [result["doc"].metadata.get("source", "unknown") for result in results]
        logger.info(f"Retrieved sources: {sources}")

        context = "\n\n".join(
            [
                f"[chunk_id={result['doc'].metadata.get('chunk_id', 'unknown')}]\n{result['doc'].page_content}"
                for result in results
            ]
        )

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
        clean_response = str(response.content)
        clean_response = format_to_markdown(clean_response)

        logger.info("LLM response generated")
        # TODO: Return {"answer": clean_response, "retrieved_chunks": results} once the API response model is updated.
        return clean_response
        

       
