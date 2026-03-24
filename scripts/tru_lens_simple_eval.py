"""
Simplified TruLens-style evaluation for the existing RAG pipeline.

This script evaluates the current app stack:
- Groq chat model for generation and LLM-as-a-judge metrics
- OpenAI embeddings for Pinecone retrieval
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from langchain_groq import ChatGroq
from pydantic import SecretStr

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ragbot.config.settings import settings
from ragbot.pipeline.rag_pipeline import RAGPipeline
from ragbot.utils.logger import get_logger

logger = get_logger("simple_trulens_eval")

QUERIES_PATH = Path("evaluation/queries.json")
RESULTS_NO_REWRITE_PATH = Path("evaluation/simple_trulens_no_rewrite.json")
RESULTS_WITH_REWRITE_PATH = Path("evaluation/simple_trulens_with_rewrite.json")
COMPARISON_PATH = Path("evaluation/simple_trulens_comparison.json")


class SimpleTruLensEvaluator:
    """
    Post-hoc evaluator for the current RAG pipeline.

    TruLens is used through its provider interface so the pipeline itself does
    not need instrumentation changes.
    """

    def __init__(self, reset_database: bool = False):
        self.pipeline = RAGPipeline()
        self.provider = self._create_provider()
        self.tru_session = self._create_tru_session()

        if reset_database and self.tru_session is not None:
            self.tru_session.reset_database()
            logger.info("TruLens database reset")
        elif reset_database:
            logger.warning(
                "Requested TruLens database reset, but no TruLens session is available."
            )

        logger.info(
            "Evaluator configured with Groq judge model '%s' and OpenAI embedding model '%s'",
            settings.llm_model,
            settings.embedding_model,
        )

    def _create_provider(self) -> Any:
        """Use Groq directly as the evaluation judge."""
        llm = ChatGroq(
            api_key=SecretStr(settings.groq_api_key),
            model=settings.llm_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        logger.info("Using Groq judge model for evaluation metrics")
        return SimpleGroqProvider(llm)

    def _create_tru_session(self) -> Any | None:
        """Create a TruLens session when the package is available."""
        try:
            from trulens.core import TruSession

            return TruSession()
        except Exception as exc:
            logger.warning("TruLens session unavailable: %s", exc)
            return None

    def _evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """Evaluate how relevant the answer is to the query."""
        try:
            score, metadata = self.provider.relevance_with_cot_reasons(query, answer)
            logger.debug(
                "Answer relevance: %.3f - %s",
                score,
                str(metadata)[:160],
            )
            return self._coerce_score(score)
        except Exception as exc:
            logger.error("Error evaluating answer relevance: %s", exc)
            return 0.0

    def _evaluate_context_relevance(self, query: str, contexts: list[str]) -> float:
        """Evaluate how relevant the retrieved contexts are to the query."""
        if not contexts:
            return 0.0

        scores: list[float] = []
        for context in contexts:
            try:
                score, _metadata = self.provider.context_relevance_with_cot_reasons(
                    query, context
                )
                scores.append(self._coerce_score(score))
            except AttributeError:
                try:
                    score, _metadata = self.provider.qs_relevance_with_cot_reasons(
                        query, context
                    )
                    scores.append(self._coerce_score(score))
                except Exception as exc:
                    logger.error("Error evaluating context relevance: %s", exc)
                    scores.append(0.0)
            except Exception as exc:
                logger.error("Error evaluating context relevance: %s", exc)
                scores.append(0.0)

        return float(np.mean(scores)) if scores else 0.0

    def _evaluate_groundedness(
        self, query: str, answer: str, contexts: list[str]
    ) -> float:
        """Evaluate how well the answer is grounded in the retrieved contexts."""
        if not contexts:
            return 0.0

        combined_context = "\n\n".join(contexts)

        try:
            score, _metadata = (
                self.provider.groundedness_measure_with_cot_reasons_consider_answerability(
                    combined_context,
                    answer,
                    query,
                )
            )
            return self._coerce_score(score)
        except AttributeError:
            try:
                score = self.provider.groundedness_measure_with_cot_reasons(
                    combined_context,
                    answer,
                )
                if isinstance(score, tuple):
                    score = score[0]
                return self._coerce_score(score)
            except Exception as exc:
                logger.error("Error evaluating groundedness: %s", exc)
                return 0.0
        except Exception as exc:
            logger.error("Error evaluating groundedness: %s", exc)
            return 0.0

    @staticmethod
    def _coerce_score(score: Any) -> float:
        """Normalize provider outputs into a [0, 1] float score."""
        try:
            return max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            return 0.0

    def load_queries(self) -> list[str]:
        """Load evaluation queries."""
        with open(QUERIES_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_results(self, results: list[dict[str, Any]], output_path: Path) -> None:
        """Save results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    def evaluate(
        self,
        use_query_rewriting: bool = False,
        output_path: Path = RESULTS_NO_REWRITE_PATH,
    ) -> list[dict[str, Any]]:
        """Run evaluation on the pipeline."""
        self.pipeline.use_query_rewriting = use_query_rewriting
        queries = self.load_queries()
        results: list[dict[str, Any]] = []

        logger.info("Starting evaluation with %s queries", len(queries))
        logger.info("Query rewriting: %s", use_query_rewriting)

        for idx, query in enumerate(queries):
            logger.info("\nEvaluating query %s/%s: %s", idx + 1, len(queries), query)

            try:
                answer = self.pipeline.run(query)
                retrieved_results = self.pipeline._retrieve_documents(query)
                contexts = [
                    result["doc"].page_content
                    for result in retrieved_results
                    if result.get("doc") is not None
                ]

                logger.info("Computing answer relevance")
                answer_relevance = self._evaluate_answer_relevance(query, answer)

                logger.info("Computing context relevance")
                context_relevance = self._evaluate_context_relevance(query, contexts)

                logger.info("Computing groundedness")
                groundedness = self._evaluate_groundedness(query, answer, contexts)

                results.append(
                    {
                        "query": query,
                        "answer": answer,
                        "answer_relevance": round(answer_relevance, 3),
                        "context_relevance": round(context_relevance, 3),
                        "groundedness": round(groundedness, 3),
                        "num_contexts": len(contexts),
                        "llm_model": settings.llm_model,
                        "embedding_model": settings.embedding_model,
                    }
                )

                logger.info(
                    "Results: AR=%.3f, CR=%.3f, G=%.3f",
                    answer_relevance,
                    context_relevance,
                    groundedness,
                )

            except Exception as exc:
                logger.error("Error evaluating query '%s': %s", query, exc, exc_info=True)
                results.append(
                    {
                        "query": query,
                        "error": str(exc),
                        "answer": None,
                        "answer_relevance": None,
                        "context_relevance": None,
                        "groundedness": None,
                        "llm_model": settings.llm_model,
                        "embedding_model": settings.embedding_model,
                    }
                )

        self.save_results(results, output_path)
        logger.info("\nSaved results to %s", output_path)
        self._log_summary(results, use_query_rewriting)
        return results

    def _log_summary(
        self, results: list[dict[str, Any]], use_query_rewriting: bool
    ) -> None:
        """Log summary statistics."""
        valid = [result for result in results if "error" not in result]

        if not valid:
            logger.warning("No valid results")
            return

        ar_scores = [
            result["answer_relevance"]
            for result in valid
            if result["answer_relevance"] is not None
        ]
        cr_scores = [
            result["context_relevance"]
            for result in valid
            if result["context_relevance"] is not None
        ]
        g_scores = [
            result["groundedness"]
            for result in valid
            if result["groundedness"] is not None
        ]

        logger.info("\n%s", "=" * 70)
        logger.info(
            "EVALUATION SUMMARY (Query Rewriting: %s)", use_query_rewriting
        )
        logger.info("%s", "=" * 70)
        logger.info("Total Queries: %s", len(results))
        logger.info("Successful: %s", len(valid))
        logger.info("Failed: %s", len(results) - len(valid))

        if ar_scores:
            logger.info("\nAnswer Relevance:")
            logger.info("   Mean:  %.3f", np.mean(ar_scores))
            logger.info("   Std:   %.3f", np.std(ar_scores))
            logger.info("   Range: %.3f - %.3f", np.min(ar_scores), np.max(ar_scores))

        if cr_scores:
            logger.info("\nContext Relevance:")
            logger.info("   Mean:  %.3f", np.mean(cr_scores))
            logger.info("   Std:   %.3f", np.std(cr_scores))
            logger.info("   Range: %.3f - %.3f", np.min(cr_scores), np.max(cr_scores))

        if g_scores:
            logger.info("\nGroundedness:")
            logger.info("   Mean:  %.3f", np.mean(g_scores))
            logger.info("   Std:   %.3f", np.std(g_scores))
            logger.info("   Range: %.3f - %.3f", np.min(g_scores), np.max(g_scores))

        logger.info("%s\n", "=" * 70)

    def save_comparison(
        self,
        baseline_results: list[dict[str, Any]],
        rewritten_results: list[dict[str, Any]],
    ) -> None:
        """Compare baseline vs query rewriting results."""
        baseline_map = {result["query"]: result for result in baseline_results}
        rewritten_map = {result["query"]: result for result in rewritten_results}

        comparison = []
        for query in self.load_queries():
            before = baseline_map.get(query, {})
            after = rewritten_map.get(query, {})

            def calc_delta(metric: str) -> float | None:
                before_value = before.get(metric)
                after_value = after.get(metric)
                if before_value is not None and after_value is not None:
                    return round(after_value - before_value, 3)
                return None

            comparison.append(
                {
                    "query": query,
                    "answer_relevance": {
                        "before": before.get("answer_relevance"),
                        "after": after.get("answer_relevance"),
                        "delta": calc_delta("answer_relevance"),
                    },
                    "context_relevance": {
                        "before": before.get("context_relevance"),
                        "after": after.get("context_relevance"),
                        "delta": calc_delta("context_relevance"),
                    },
                    "groundedness": {
                        "before": before.get("groundedness"),
                        "after": after.get("groundedness"),
                        "delta": calc_delta("groundedness"),
                    },
                }
            )

        self.save_results(comparison, COMPARISON_PATH)
        logger.info("Saved comparison to %s", COMPARISON_PATH)
        self._log_comparison(comparison)

    def _log_comparison(self, comparison: list[dict[str, Any]]) -> None:
        """Log comparison summary."""
        ar_deltas = [
            item["answer_relevance"]["delta"]
            for item in comparison
            if item["answer_relevance"]["delta"] is not None
        ]
        cr_deltas = [
            item["context_relevance"]["delta"]
            for item in comparison
            if item["context_relevance"]["delta"] is not None
        ]
        g_deltas = [
            item["groundedness"]["delta"]
            for item in comparison
            if item["groundedness"]["delta"] is not None
        ]

        logger.info("\n%s", "=" * 70)
        logger.info("QUERY REWRITING IMPACT")
        logger.info("%s", "=" * 70)

        if ar_deltas:
            improved = sum(1 for delta in ar_deltas if delta > 0)
            logger.info("\nAnswer Relevance:")
            logger.info("   Avg Delta: %+.3f", np.mean(ar_deltas))
            logger.info("   Improved:  %s/%s queries", improved, len(ar_deltas))

        if cr_deltas:
            improved = sum(1 for delta in cr_deltas if delta > 0)
            logger.info("\nContext Relevance:")
            logger.info("   Avg Delta: %+.3f", np.mean(cr_deltas))
            logger.info("   Improved:  %s/%s queries", improved, len(cr_deltas))

        if g_deltas:
            improved = sum(1 for delta in g_deltas if delta > 0)
            logger.info("\nGroundedness:")
            logger.info("   Avg Delta: %+.3f", np.mean(g_deltas))
            logger.info("   Improved:  %s/%s queries", improved, len(g_deltas))

        logger.info("%s\n", "=" * 70)


class SimpleGroqProvider:
    """Local fallback evaluator when the TruLens LangChain provider is unavailable."""

    def __init__(self, llm: ChatGroq):
        self.llm = llm

    def _call_llm(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return str(response.content).strip()

    def _score_prompt(self, prompt: str) -> tuple[float, dict[str, str]]:
        result = self._call_llm(prompt)
        try:
            reasoning, score_text = result.split("SCORE:", maxsplit=1)
            reasoning = reasoning.replace("REASONING:", "").strip()
            score = max(0.0, min(1.0, float(score_text.strip())))
            return score, {"reasoning": reasoning}
        except (ValueError, IndexError):
            return 0.0, {"reasoning": f"Could not parse judge output: {result}"}

    def relevance_with_cot_reasons(
        self, question: str, response: str
    ) -> tuple[float, dict[str, str]]:
        prompt = f"""Rate how relevant this answer is to the question on a scale of 0.0 to 1.0.

Question: {question}
Answer: {response}

Provide:
1. Your reasoning
2. A score between 0.0 (not relevant) and 1.0 (highly relevant)

Format: REASONING: <your reasoning> | SCORE: <number>"""
        return self._score_prompt(prompt)

    def context_relevance_with_cot_reasons(
        self, question: str, context: str
    ) -> tuple[float, dict[str, str]]:
        prompt = f"""Rate how relevant this context is to answering the question on a scale of 0.0 to 1.0.

Question: {question}
Context: {context}

Provide:
1. Your reasoning
2. A score between 0.0 (not relevant) and 1.0 (highly relevant)

Format: REASONING: <your reasoning> | SCORE: <number>"""
        return self._score_prompt(prompt)

    def groundedness_measure_with_cot_reasons_consider_answerability(
        self, source: str, statement: str, question: str
    ) -> tuple[float, dict[str, str]]:
        prompt = f"""Rate how well the answer is grounded in the source context on a scale of 0.0 to 1.0.

Question: {question}
Source Context: {source}
Answer: {statement}

Give a higher score only when the answer is supported by the source context.

Provide:
1. Your reasoning
2. A score between 0.0 (not grounded) and 1.0 (fully grounded)

Format: REASONING: <your reasoning> | SCORE: <number>"""
        return self._score_prompt(prompt)


def main() -> None:
    """Main evaluation script."""
    evaluator = SimpleTruLensEvaluator(reset_database=False)

    logger.info("\n%s", "🔍" * 35)
    logger.info("PHASE 1: Baseline (No Query Rewriting)")
    logger.info("%s\n", "🔍" * 35)
    results_baseline = evaluator.evaluate(
        use_query_rewriting=False,
        output_path=RESULTS_NO_REWRITE_PATH,
    )

    logger.info("\n%s", "🔍" * 35)
    logger.info("PHASE 2: With Query Rewriting")
    logger.info("%s\n", "🔍" * 35)
    results_rewrite = evaluator.evaluate(
        use_query_rewriting=True,
        output_path=RESULTS_WITH_REWRITE_PATH,
    )

    logger.info("\n%s", "📊" * 35)
    logger.info("PHASE 3: Comparison Analysis")
    logger.info("%s\n", "📊" * 35)
    evaluator.save_comparison(results_baseline, results_rewrite)

    logger.info("\nEvaluation complete!")
    logger.info("   - Baseline results: %s", RESULTS_NO_REWRITE_PATH)
    logger.info("   - Rewrite results:  %s", RESULTS_WITH_REWRITE_PATH)
    logger.info("   - Comparison:       %s\n", COMPARISON_PATH)


if __name__ == "__main__":
    main()
