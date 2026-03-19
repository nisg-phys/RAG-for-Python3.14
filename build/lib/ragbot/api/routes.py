from fastapi import APIRouter
from pydantic import BaseModel

from ragbot.pipeline.rag_pipeline import RAGPipeline


router = APIRouter()

rag = RAGPipeline()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    answer = rag.run(request.query)
    return QueryResponse(answer=answer)
@router.get("/health")
def health():
    return {"status": "ok"}