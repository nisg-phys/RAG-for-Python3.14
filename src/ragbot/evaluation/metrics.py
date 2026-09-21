from __future__ import annotations

import re
from typing import Any

from ragbot.observability.models import EvaluationSummary


FALLBACK_ANSWER = "I am sorry, but I don't have enough information to answer that question."
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "uses",
    "using",
    "via",
    "what",
    "while",
    "with",
}


def derive_expected_keywords(text: str, max_keywords: int = 8) -> list[str]:
    keywords = []
    seen = set()

    for token in TOKEN_PATTERN.findall(text.lower()):
        if token in STOPWORDS or token.isdigit() or len(token) <= 2:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= max_keywords:
            break

    return keywords


def normalize_eval_specs(raw_specs: list[Any]) -> list[dict[str, Any]]:
    normalized = []
    for item in raw_specs:
        if isinstance(item, str):
            normalized.append({"query": item, "expected_keywords": []})
            continue

        query = item.get("query", item.get("question", ""))
        ground_truth = item.get("ground_truth")
        expected_keywords = [keyword.lower() for keyword in item.get("expected_keywords", [])]
        if not expected_keywords and ground_truth:
            expected_keywords = derive_expected_keywords(ground_truth)

        normalized.append(
            {
                "query": query,
                "expected_keywords": expected_keywords,
                "ground_truth": ground_truth,
                "source": item.get("source"),
            }
        )
    return normalized


def _keyword_hits(text: str, expected_keywords: list[str]) -> set[str]:
    lowered = text.lower()
    tokens = set(TOKEN_PATTERN.findall(lowered))
    hits = set()

    for keyword in expected_keywords:
        if " " in keyword:
            if keyword in lowered:
                hits.add(keyword)
        elif keyword in tokens:
            hits.add(keyword)

    return hits


def _first_relevant_rank(chunk_texts: list[str], expected_keywords: list[str]) -> int | None:
    if not expected_keywords:
        return None

    for index, chunk_text in enumerate(chunk_texts, start=1):
        if _keyword_hits(chunk_text, expected_keywords):
            return index
    return None


def compute_query_metrics(
    *,
    query: str,
    answer: str,
    chunk_texts: list[str],
    expected_keywords: list[str],
    ground_truth: str | None = None,
    pipeline_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    relevant_chunks = [
        chunk_text for chunk_text in chunk_texts if _keyword_hits(chunk_text, expected_keywords)
    ]
    first_rank = _first_relevant_rank(chunk_texts, expected_keywords)
    matched_answer_keywords = _keyword_hits(answer, expected_keywords)
    fallback_used = answer.strip() == FALLBACK_ANSWER

    retrieval_hit_rate = 1.0 if relevant_chunks else 0.0
    reciprocal_rank = round(1 / first_rank, 4) if first_rank else 0.0
    context_precision = round(len(relevant_chunks) / len(chunk_texts), 4) if chunk_texts else 0.0
    answer_keyword_recall = (
        round(len(matched_answer_keywords) / len(expected_keywords), 4)
        if expected_keywords
        else 0.0
    )

    pipeline_metrics = pipeline_metrics or {}
    retrieval_metrics = pipeline_metrics.get("retrieval", {})
    generation_metrics = pipeline_metrics.get("generation", {})

    return {
        "query": query,
        "expected_keywords": expected_keywords,
        "ground_truth": ground_truth,
        "retrieval_hit_rate_at_k": retrieval_hit_rate,
        "reciprocal_rank_at_k": reciprocal_rank,
        "context_precision_at_k": context_precision,
        "answer_keyword_recall": answer_keyword_recall,
        "answer_fallback_used": fallback_used,
        "answer": answer,
        "retrieved_chunks": len(chunk_texts),
        "latency_ms": round(float(pipeline_metrics.get("total_latency_ms", 0.0)), 4),
        "retrieval_latency_ms": round(float(retrieval_metrics.get("retrieval_latency_ms", 0.0)), 4),
        "generation_latency_ms": round(float(generation_metrics.get("generation_latency_ms", 0.0)), 4),
    }


def summarize_metrics(results: list[dict[str, Any]]) -> EvaluationSummary:
    total = len(results)
    if total == 0:
        return EvaluationSummary(
            queries_evaluated=0,
            retrieval_hit_rate_at_k=0.0,
            mean_reciprocal_rank_at_k=0.0,
            mean_context_precision_at_k=0.0,
            mean_answer_keyword_recall=0.0,
            answer_fallback_rate=0.0,
            avg_total_latency_ms=0.0,
            avg_retrieval_latency_ms=0.0,
            avg_generation_latency_ms=0.0,
        )

    return EvaluationSummary(
        queries_evaluated=total,
        retrieval_hit_rate_at_k=round(
            sum(result["retrieval_hit_rate_at_k"] for result in results) / total,
            4,
        ),
        mean_reciprocal_rank_at_k=round(
            sum(result["reciprocal_rank_at_k"] for result in results) / total,
            4,
        ),
        mean_context_precision_at_k=round(
            sum(result["context_precision_at_k"] for result in results) / total,
            4,
        ),
        mean_answer_keyword_recall=round(
            sum(result["answer_keyword_recall"] for result in results) / total,
            4,
        ),
        answer_fallback_rate=round(
            sum(1.0 if result["answer_fallback_used"] else 0.0 for result in results) / total,
            4,
        ),
        avg_total_latency_ms=round(
            sum(result["latency_ms"] for result in results) / total,
            4,
        ),
        avg_retrieval_latency_ms=round(
            sum(result["retrieval_latency_ms"] for result in results) / total,
            4,
        ),
        avg_generation_latency_ms=round(
            sum(result["generation_latency_ms"] for result in results) / total,
            4,
        ),
    )
