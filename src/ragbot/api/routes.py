from typing import Annotated
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ragbot.pipeline.rag_pipeline import RAGPipeline
from ragbot.utils.logger import get_logger

logger = get_logger("api")


router = APIRouter()

rag = RAGPipeline()


class QueryRequest(BaseModel):
    query: Annotated[str,Field(..., description="Ask anything about python 3.14 documents", examples=["How is the error handled in python?"])]


class QueryResponse(BaseModel):
    answer: str


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    logger.info(f"API query received: {request.query}")
    answer = rag.run(request.query)
    logger.info("Response returned to client")
    return QueryResponse(answer= answer)
@router.get("/health")
def health():
    logger.info("Health check requested")
    return {"status": "ok"}