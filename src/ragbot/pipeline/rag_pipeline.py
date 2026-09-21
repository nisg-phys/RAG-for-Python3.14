import pickle
import time
from pathlib import Path
from uuid import uuid4

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ragbot.retrievers.hybrid_retriever import HybridRetriever
from ragbot.retrievers.query_rewriter import QueryRewriter
from ragbot.vectorstore.pinecone_store import PineconeStore
from ragbot.config.settings import settings
from ragbot.ingestion.s3_storage import download_chunks
from ragbot.observability.models import GenerationMetrics, QueryTelemetry, RetrievalMetrics
from ragbot.observability.opik_tracing import get_langchain_tracer, track, update_trace_metadata
from ragbot.observability.telemetry import record_event
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
            logger.warning(
                "No chunks found in S3 yet; starting without a retriever. "
                "Call POST /ingest to populate the index before querying."
            )
            self.documents = []
            self.retriever = None
        else:
            self.documents = chunks
            logger.info(f"Loaded {len(chunks)} chunks for hybrid retrieval")
            self.retriever = HybridRetriever(
                self.vectorstore,
                self.documents
            )

        self.llm = self._build_llm()

        # Reused across requests: nests as a child span under whichever
        # @track-decorated trace is active when it's invoked (see run()).
        self._opik_tracer = get_langchain_tracer()

        self.query_rewriter = QueryRewriter(self.llm, opik_tracer=self._opik_tracer)
        self.use_query_rewriting = settings.use_query_rewriting

        # Prompt template
        self.prompt  = rag_prompt

    def _build_llm(self):
        provider = settings.llm_provider.lower()
        logger.info("Initializing LLM provider=%s model=%s", provider, settings.llm_model)

        if provider == "openai":
            return ChatOpenAI(
                api_key=SecretStr(settings.openai_api_key),
                model=settings.llm_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )

        if provider == "groq":
            groq_model = settings.groq_llm_model or settings.llm_model
            return ChatGroq(
                api_key=SecretStr(settings.groq_api_key),
                model=groq_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )

        raise ValueError(f"Unsupported llm_provider: {settings.llm_provider}")

    def reload_documents(self):
        """Re-download chunks from S3 and rebuild the BM25 retriever.

        Called after a live /ingest run so this process picks up newly
        ingested chunks without needing a restart.
        """
        chunks = download_chunks()
        if chunks is None:
            raise RuntimeError("Chunks not found in S3. Run ingestion first.")

        self.documents = chunks
        self.retriever = HybridRetriever(self.vectorstore, self.documents)
        logger.info(f"Reloaded {len(chunks)} chunks for hybrid retrieval")

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

    @track(name="rag.retrieve_documents", type="tool")
    def _retrieve_documents(self, query: str, top_k=5):
        if self.retriever is None:
            raise RuntimeError("No documents ingested yet. Call POST /ingest first.")

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

    def _build_query_metrics(self, *, trace_id: str, query: str, results, retrieval_latency_ms: float, generation_latency_ms: float, prompt_chars: int, answer_chars: int) -> QueryTelemetry:
        vector_scores = [float(result.get("vector_score", 0.0)) for result in results]
        bm25_scores = [float(result.get("bm25_score", 0.0)) for result in results]
        rrf_scores = [float(result.get("rrf_score", 0.0)) for result in results]
        sources = [result["doc"].metadata.get("source", "unknown") for result in results]
        total_latency_ms = round(retrieval_latency_ms + generation_latency_ms, 4)

        return QueryTelemetry(
            trace_id=trace_id,
            query=query,
            rewriting_enabled=self.use_query_rewriting,
            total_latency_ms=total_latency_ms,
            retrieval=RetrievalMetrics(
                requested_top_k=settings.top_k,
                returned_chunks=len(results),
                unique_sources=len(set(sources)),
                avg_vector_score=round(sum(vector_scores) / len(vector_scores), 4) if vector_scores else 0.0,
                avg_bm25_score=round(sum(bm25_scores) / len(bm25_scores), 4) if bm25_scores else 0.0,
                avg_rrf_score=round(sum(rrf_scores) / len(rrf_scores), 4) if rrf_scores else 0.0,
                retrieval_latency_ms=round(retrieval_latency_ms, 4),
            ),
            generation=GenerationMetrics(
                prompt_chars=prompt_chars,
                answer_chars=answer_chars,
                generation_latency_ms=round(generation_latency_ms, 4),
            ),
            source_paths=sources,
        )

    @track(name="rag.run", tags=["rag_query"])
    def run(self, query: str):
        trace_id = uuid4().hex[:12]

        logger.info(f"Received query: {query}")

        # Retrieve documents
        retrieval_start = time.perf_counter()
        results = self._retrieve_documents(query)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        logger.info(f"{len(results)} documents retrieved after hybrid merge")
        # Combine context
        sources = [result["doc"].metadata.get("source", "unknown") for result in results]
        logger.info(f"Retrieved sources: {sources}")
        unique_sources = list(dict.fromkeys(sources))

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
        generation_start = time.perf_counter()
        llm_config = {"callbacks": [self._opik_tracer]} if self._opik_tracer else {}
        response = self.llm.invoke(prompt, config=llm_config)
        generation_latency_ms = (time.perf_counter() - generation_start) * 1000
        clean_response = str(response.content)
        clean_response = format_to_markdown(clean_response)

        metrics = self._build_query_metrics(
            trace_id=trace_id,
            query=query,
            results=results,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            prompt_chars=len(str(prompt)),
            answer_chars=len(clean_response),
        )
        record_event("rag_query", metrics)
        update_trace_metadata(metadata={"internal_trace_id": trace_id, **metrics.model_dump()}, tags=["rag_query"])

        logger.info("LLM response generated")
        logger.info("Trace %s completed in %.2f ms", trace_id, metrics.total_latency_ms)
        return {
            "answer": clean_response,
            "retrieved_chunks": results,
            "sources": unique_sources,
            "metrics": metrics.model_dump(),
        }
        

       
