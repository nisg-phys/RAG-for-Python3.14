from pydantic import BaseModel, Field


class RetrievalMetrics(BaseModel):
    requested_top_k: int
    returned_chunks: int
    unique_sources: int
    avg_vector_score: float = 0.0
    avg_bm25_score: float = 0.0
    avg_rrf_score: float = 0.0
    retrieval_latency_ms: float


class GenerationMetrics(BaseModel):
    prompt_chars: int
    answer_chars: int
    generation_latency_ms: float


class QueryTelemetry(BaseModel):
    trace_id: str
    query: str
    rewriting_enabled: bool
    total_latency_ms: float
    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    source_paths: list[str] = Field(default_factory=list)


class IngestionMetrics(BaseModel):
    files_discovered: int
    documents_loaded: int
    chunks_created: int
    chunks_deleted: int
    chunks_reused: int
    chunks_total: int
    sources_new: int
    sources_changed: int
    sources_unchanged: int
    sources_removed: int
    avg_chunks_per_document: float
    avg_chunk_length_chars: float
    dataset_hash: str


class EvaluationSummary(BaseModel):
    queries_evaluated: int
    retrieval_hit_rate_at_k: float
    mean_reciprocal_rank_at_k: float
    mean_context_precision_at_k: float
    mean_answer_keyword_recall: float
    answer_fallback_rate: float
    avg_total_latency_ms: float
    avg_retrieval_latency_ms: float
    avg_generation_latency_ms: float
