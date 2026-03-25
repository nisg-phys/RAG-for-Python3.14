"""
Simplified TruLens evaluation that works with your existing RAG pipeline.
This version doesn't require pipeline modifications.
"""

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from trulens.core import TruSession, Select
from trulens.core.feedback import Feedback
from trulens.core.app import App
from trulens.core.schema import Record
from trulens.feedback import Groundedness
from ragbot.pipeline.rag_pipeline import RAGPipeline
from ragbot.utils.logger import get_logger
from trulens.providers.google import Gemini 

logger = get_logger("simple_trulens_eval")

QUERIES_PATH = Path("evaluation/queries.json")
RESULTS_NO_REWRITE_PATH = Path("evaluation/simple_trulens_no_rewrite.json")
RESULTS_WITH_REWRITE_PATH = Path("evaluation/simple_trulens_with_rewrite.json")
COMPARISON_PATH = Path("evaluation/simple_trulens_comparison.json")


class SimpleTruLensEvaluator:
    """
    Simplified TruLens evaluator that works with any RAG pipeline.
    Does not require pipeline instrumentation - evaluates results post-hoc.
    """

    def __init__(self, reset_database=False):
        self.tru = Tru()
        if reset_database:
            self.tru.reset_database()
            logger.info("TruLens database reset")

        self.pipeline = RAGPipeline()

        # Setup feedback provider
        try:
            from trulens_eval.feedback.provider import OpenAI
            self.provider = OpenAI()
            logger.info("Using OpenAI for feedback evaluation")
        except Exception as e:
            logger.warning(f"OpenAI not available: {e}. Using fallback provider.")
            self.provider = self._create_fallback_provider()

        # Setup groundedness checker
        self.grounded = Groundedness(groundedness_provider=self.provider)

    def _create_fallback_provider(self):
        """Fallback to Groq-based evaluation."""
        from langchain_groq import ChatGroq
        from pydantic import SecretStr
        from ragbot.config.settings import settings

        class SimpleGroqProvider:
            def __init__(self):
                self.llm = ChatGroq(
                    api_key=SecretStr(settings.groq_api_key),
                    model=settings.llm_model,
                )

            def _call_llm(self, prompt):
                response = self.llm.invoke(prompt)
                return str(response.content).strip()

            def relevance_with_cot_reasons(self, question: str, response: str) -> tuple:
                """Evaluate answer relevance to question."""
                prompt = f"""Rate how relevant this answer is to the question on a scale of 0.0 to 1.0.

Question: {question}
Answer: {response}

Provide:
1. Your reasoning
2. A score between 0.0 (not relevant) and 1.0 (highly relevant)

Format: REASONING: <your reasoning> | SCORE: <number>"""
                
                result = self._call_llm(prompt)
                try:
                    reasoning = result.split("SCORE:")[0].replace("REASONING:", "").strip()
                    score = float(result.split("SCORE:")[1].strip())
                    return max(0.0, min(1.0, score)), {"reasoning": reasoning}
                except:
                    return 0.5, {"reasoning": "Could not parse response"}

            def qs_relevance_with_cot_reasons(self, question: str, context: str) -> tuple:
                """Evaluate context relevance to question."""
                prompt = f"""Rate how relevant this context is to answering the question on a scale of 0.0 to 1.0.

Question: {question}
Context: {context}

Provide:
1. Your reasoning
2. A score between 0.0 (not relevant) and 1.0 (highly relevant)

Format: REASONING: <your reasoning> | SCORE: <number>"""
                
                result = self._call_llm(prompt)
                try:
                    reasoning = result.split("SCORE:")[0].replace("REASONING:", "").strip()
                    score = float(result.split("SCORE:")[1].strip())
                    return max(0.0, min(1.0, score)), {"reasoning": reasoning}
                except:
                    return 0.5, {"reasoning": "Could not parse response"}

        return SimpleGroqProvider()

    def _evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """Evaluate how relevant the answer is to the query."""
        try:
            score, metadata = self.provider.relevance_with_cot_reasons(query, answer)
            logger.debug(f"Answer relevance: {score:.3f} - {metadata.get('reasoning', '')[:100]}")
            return float(score)
        except Exception as e:
            logger.error(f"Error evaluating answer relevance: {e}")
            return 0.0

    def _evaluate_context_relevance(self, query: str, contexts: list) -> float:
        """Evaluate how relevant the retrieved contexts are to the query."""
        if not contexts:
            return 0.0

        scores = []
        for context in contexts:
            try:
                score, metadata = self.provider.qs_relevance_with_cot_reasons(query, context)
                scores.append(float(score))
            except Exception as e:
                logger.error(f"Error evaluating context relevance: {e}")
                scores.append(0.0)

        return float(np.mean(scores)) if scores else 0.0

    def _evaluate_groundedness(self, answer: str, contexts: list) -> float:
        """Evaluate how well the answer is grounded in the contexts."""
        if not contexts:
            return 0.0

        try:
            # Combine contexts
            combined_context = "\n\n".join(contexts)

            # Use TruLens groundedness measure
            score = self.grounded.groundedness_measure_with_cot_reasons(
                combined_context, answer
            )

            if isinstance(score, tuple):
                score = score[0]

            return float(score)
        except Exception as e:
            logger.error(f"Error evaluating groundedness: {e}")
            return 0.0

    def load_queries(self):
        """Load evaluation queries."""
        with open(QUERIES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_results(self, results, output_path):
        """Save results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    def evaluate(self, use_query_rewriting=False, output_path=RESULTS_NO_REWRITE_PATH):
        """
        Run evaluation on the pipeline.
        """
        self.pipeline.use_query_rewriting = use_query_rewriting
        queries = self.load_queries()
        results = []

        logger.info(f"Starting evaluation with {len(queries)} queries")
        logger.info(f"Query rewriting: {use_query_rewriting}")

        for idx, query in enumerate(queries):
            logger.info(f"\nEvaluating query {idx + 1}/{len(queries)}: {query}")

            try:
                # Run pipeline
                answer = self.pipeline.run(query)

                # Get retrieved documents for evaluation
                retrieved_results = self.pipeline._retrieve_documents(query)
                contexts = [
                    result["doc"].page_content
                    for result in retrieved_results
                ]

                # Evaluate metrics
                logger.info("Computing answer relevance...")
                answer_relevance = self._evaluate_answer_relevance(query, answer)

                logger.info("Computing context relevance...")
                context_relevance = self._evaluate_context_relevance(query, contexts)

                logger.info("Computing groundedness...")
                groundedness = self._evaluate_groundedness(answer, contexts)

                results.append({
                    "query": query,
                    "answer": answer,
                    "answer_relevance": round(answer_relevance, 3),
                    "context_relevance": round(context_relevance, 3),
                    "groundedness": round(groundedness, 3),
                    "num_contexts": len(contexts),
                })

                logger.info(f"Results: AR={answer_relevance:.3f}, CR={context_relevance:.3f}, G={groundedness:.3f}")

            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {e}", exc_info=True)
                results.append({
                    "query": query,
                    "error": str(e),
                    "answer": None,
                    "answer_relevance": None,
                    "context_relevance": None,
                    "groundedness": None,
                })

        # Save results
        self.save_results(results, output_path)
        logger.info(f"\nSaved results to {output_path}")

        # Calculate summary
        self._log_summary(results, use_query_rewriting)

        return results

    def _log_summary(self, results, use_query_rewriting):
        """Log summary statistics."""
        valid = [r for r in results if "error" not in r]

        if not valid:
            logger.warning("No valid results")
            return

        ar_scores = [r["answer_relevance"] for r in valid if r["answer_relevance"] is not None]
        cr_scores = [r["context_relevance"] for r in valid if r["context_relevance"] is not None]
        g_scores = [r["groundedness"] for r in valid if r["groundedness"] is not None]

        logger.info(f"\n{'='*70}")
        logger.info(f"EVALUATION SUMMARY (Query Rewriting: {use_query_rewriting})")
        logger.info(f"{'='*70}")
        logger.info(f"Total Queries: {len(results)}")
        logger.info(f"Successful: {len(valid)}")
        logger.info(f"Failed: {len(results) - len(valid)}")

        if ar_scores:
            logger.info(f"\n📊 Answer Relevance:")
            logger.info(f"   Mean:  {np.mean(ar_scores):.3f}")
            logger.info(f"   Std:   {np.std(ar_scores):.3f}")
            logger.info(f"   Range: {np.min(ar_scores):.3f} - {np.max(ar_scores):.3f}")

        if cr_scores:
            logger.info(f"\n📊 Context Relevance:")
            logger.info(f"   Mean:  {np.mean(cr_scores):.3f}")
            logger.info(f"   Std:   {np.std(cr_scores):.3f}")
            logger.info(f"   Range: {np.min(cr_scores):.3f} - {np.max(cr_scores):.3f}")

        if g_scores:
            logger.info(f"\n📊 Groundedness:")
            logger.info(f"   Mean:  {np.mean(g_scores):.3f}")
            logger.info(f"   Std:   {np.std(g_scores):.3f}")
            logger.info(f"   Range: {np.min(g_scores):.3f} - {np.max(g_scores):.3f}")

        logger.info(f"{'='*70}\n")

    def save_comparison(self, baseline_results, rewritten_results):
        """Compare baseline vs query rewriting results."""
        baseline_map = {r["query"]: r for r in baseline_results}
        rewritten_map = {r["query"]: r for r in rewritten_results}

        comparison = []
        for query in self.load_queries():
            before = baseline_map.get(query, {})
            after = rewritten_map.get(query, {})

            def calc_delta(metric):
                b = before.get(metric)
                a = after.get(metric)
                if b is not None and a is not None:
                    return round(a - b, 3)
                return None

            comparison.append({
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
            })

        self.save_results(comparison, COMPARISON_PATH)
        logger.info(f"Saved comparison to {COMPARISON_PATH}")

        # Log comparison summary
        self._log_comparison(comparison)

    def _log_comparison(self, comparison):
        """Log comparison summary."""
        ar_deltas = [c["answer_relevance"]["delta"] for c in comparison if c["answer_relevance"]["delta"] is not None]
        cr_deltas = [c["context_relevance"]["delta"] for c in comparison if c["context_relevance"]["delta"] is not None]
        g_deltas = [c["groundedness"]["delta"] for c in comparison if c["groundedness"]["delta"] is not None]

        logger.info(f"\n{'='*70}")
        logger.info(f"QUERY REWRITING IMPACT")
        logger.info(f"{'='*70}")

        if ar_deltas:
            improved = sum(1 for d in ar_deltas if d > 0)
            logger.info(f"\n📈 Answer Relevance:")
            logger.info(f"   Avg Delta: {np.mean(ar_deltas):+.3f}")
            logger.info(f"   Improved:  {improved}/{len(ar_deltas)} queries")

        if cr_deltas:
            improved = sum(1 for d in cr_deltas if d > 0)
            logger.info(f"\n📈 Context Relevance:")
            logger.info(f"   Avg Delta: {np.mean(cr_deltas):+.3f}")
            logger.info(f"   Improved:  {improved}/{len(cr_deltas)} queries")

        if g_deltas:
            improved = sum(1 for d in g_deltas if d > 0)
            logger.info(f"\n📈 Groundedness:")
            logger.info(f"   Avg Delta: {np.mean(g_deltas):+.3f}")
            logger.info(f"   Improved:  {improved}/{len(g_deltas)} queries")

        logger.info(f"{'='*70}\n")


def main():
    """Main evaluation script."""
    evaluator = SimpleTruLensEvaluator(reset_database=False)

    # Phase 1: Baseline (no query rewriting)
    logger.info("\n" + "🔍" * 35)
    logger.info("PHASE 1: Baseline (No Query Rewriting)")
    logger.info("🔍" * 35 + "\n")

    results_baseline = evaluator.evaluate(
        use_query_rewriting=False,
        output_path=RESULTS_NO_REWRITE_PATH,
    )

    # Phase 2: With query rewriting
    logger.info("\n" + "🔍" * 35)
    logger.info("PHASE 2: With Query Rewriting")
    logger.info("🔍" * 35 + "\n")

    results_rewrite = evaluator.evaluate(
        use_query_rewriting=True,
        output_path=RESULTS_WITH_REWRITE_PATH,
    )

    # Phase 3: Comparison
    logger.info("\n" + "📊" * 35)
    logger.info("PHASE 3: Comparison Analysis")
    logger.info("📊" * 35 + "\n")

    evaluator.save_comparison(results_baseline, results_rewrite)

    logger.info("\n✅ Evaluation complete!")
    logger.info(f"   - Baseline results: {RESULTS_NO_REWRITE_PATH}")
    logger.info(f"   - Rewrite results:  {RESULTS_WITH_REWRITE_PATH}")
    logger.info(f"   - Comparison:       {COMPARISON_PATH}\n")


if __name__ == "__main__":
    main()