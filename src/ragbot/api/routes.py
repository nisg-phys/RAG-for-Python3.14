from typing import Annotated

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ragbot.observability.models import QueryTelemetry
from ragbot.observability.telemetry import build_summary
from ragbot.services.ingestion_service import delete_index, run_ingestion
from ragbot.utils.logger import get_logger
from ragbot.pipeline.rag_pipeline import RAGPipeline

logger = get_logger("api")


router = APIRouter()

rag = RAGPipeline()


class QueryRequest(BaseModel):
    query: Annotated[str,Field(..., description="Ask anything about python 3.14 documents", examples=["How is the error handled in python?"])]


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    metrics: QueryTelemetry


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    chunks_deleted: int
    sources_new: int
    sources_changed: int
    sources_unchanged: int
    sources_removed: int


class DeleteIndexResponse(BaseModel):
    status: str

@router.get("/")
def hello():
    return{"message":"RAG Bot that will answer everything about python 3.14 with codes and examples."}


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    logger.info(f"API query received: {request.query}")
    result = rag.run(request.query)
    logger.info("Response returned to client")
    return QueryResponse(answer=result["answer"], sources=result["sources"], metrics=result["metrics"])
@router.get("/health")
def health():
    logger.info("Health check requested")
    return {"status": "ok"}


@router.get("/observability/summary")
def observability_summary():
    logger.info("Observability summary requested")
    return build_summary()


@router.post("/ingest", response_model=IngestResponse)
def ingest():
    logger.info("Ingestion requested")
    result = run_ingestion()
    rag.reload_documents()
    logger.info("Ingestion complete, retriever reloaded")
    return IngestResponse(
        status="ok",
        documents_loaded=result["documents_loaded"],
        chunks_created=result["chunks_created"],
        chunks_deleted=result["chunks_deleted"],
        sources_new=result["sources_new"],
        sources_changed=result["sources_changed"],
        sources_unchanged=result["sources_unchanged"],
        sources_removed=result["sources_removed"],
    )


@router.delete("/index", response_model=DeleteIndexResponse)
def delete_index_route():
    logger.info("Index deletion requested")
    delete_index()
    return DeleteIndexResponse(status="ok")
