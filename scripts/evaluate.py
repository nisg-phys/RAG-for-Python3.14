import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langchain_groq import ChatGroq
from pydantic import SecretStr

from ragbot.config.settings import settings
from ragbot.pipeline.rag_pipeline import RAGPipeline
from ragbot.utils.logger import get_logger


logger = get_logger("evaluation")

QUERIES_PATH = Path("evaluation/queries.json")
RESULTS_PATH = Path("evaluation/results.json")
RESULTS_NO_REWRITE_PATH = Path("evaluation/results_no_rewrite.json")
RESULTS_WITH_REWRITE_PATH = Path("evaluation/results_with_rewrite.json")
COMPARISON_PATH = Path("evaluation/comparison.json")
USE_QUERY_REWRITING = False


class EvaluationRunner:
    def __init__(self):
        self.pipeline = RAGPipeline()
        self.judge_llm = ChatGroq(
            api_key=SecretStr(settings.groq_api_key),
            model=settings.llm_model,
        )

    def _ask_judge(self, prompt: str) -> str:
        response = self.judge_llm.invoke(prompt)
        return str(response.content).strip()

    def judge_relevance(self, query, chunk_text):
        prompt = f"""
You are evaluating retrieval relevance for a RAG system.

Query:
{query}

Chunk:
{chunk_text}

Is this chunk relevant to answering the query?
Answer with exactly one word: YES or NO
""".strip()
        verdict = self._ask_judge(prompt).upper()
        return verdict.startswith("YES")

    def judge_coverage(self, query, chunk_texts):
        context = "\n\n".join(chunk_texts)
        prompt = f"""
You are evaluating context sufficiency for a RAG system.

Query:
{query}

Context:
{context}

Is the context sufficient to answer the query?
Answer with exactly one word: FULL, PARTIAL, or NONE
""".strip()
        verdict = self._ask_judge(prompt).upper()
        if verdict.startswith("FULL"):
            return "FULL"
        if verdict.startswith("PARTIAL"):
            return "PARTIAL"
        return "NONE"

    def judge_groundedness(self, query, answer, context):
        prompt = f"""
You are evaluating answer groundedness for a RAG system.

Query:
{query}

Answer:
{answer}

Context:
{context}

Is the answer fully supported by the context?
Answer with exactly one word: YES or NO
""".strip()
        verdict = self._ask_judge(prompt).upper()
        return verdict.startswith("YES")

    def load_queries(self):
        with open(QUERIES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_results(self, results, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    def _normalize_output(self, query, output):
        if isinstance(output, dict):
            answer = output.get("answer", "")
            retrieved_chunks = output.get("retrieved_chunks", [])
            return answer, retrieved_chunks

        logger.info("Pipeline returned a string answer; retrieving chunks separately for evaluation")
        answer = output
        retrieved_chunks = self.pipeline._retrieve_documents(query)
        return answer, retrieved_chunks

    def _extract_chunk_texts(self, retrieved_chunks):
        chunk_texts = []

        for chunk in retrieved_chunks:
            if isinstance(chunk, dict):
                if "text" in chunk:
                    chunk_texts.append(chunk["text"])
                elif "doc" in chunk and hasattr(chunk["doc"], "page_content"):
                    chunk_texts.append(chunk["doc"].page_content)
            elif hasattr(chunk, "page_content"):
                chunk_texts.append(chunk.page_content)

        return chunk_texts

    def evaluate(self, use_query_rewriting=USE_QUERY_REWRITING, output_path=RESULTS_PATH):
        self.pipeline.use_query_rewriting = use_query_rewriting
        queries = self.load_queries()
        results = []

        for query in queries:
            logger.info(f"Evaluating query: {query}")
            output = self.pipeline.run(query)

            answer, retrieved_chunks = self._normalize_output(query, output)
            chunk_texts = self._extract_chunk_texts(retrieved_chunks)

            relevant_count = 0
            for chunk_text in chunk_texts:
                if self.judge_relevance(query, chunk_text):
                    relevant_count += 1

            total_chunks = len(chunk_texts)
            precision_at_k = relevant_count / total_chunks if total_chunks else 0.0
            coverage = self.judge_coverage(query, chunk_texts)
            grounded = self.judge_groundedness(query, answer, "\n\n".join(chunk_texts))

            results.append(
                {
                    "query": query,
                    "precision_at_k": precision_at_k,
                    "coverage": coverage,
                    "grounded": grounded,
                    "groundedness": grounded,
                    "answer": answer,
                }
            )

        self.save_results(results, output_path)
        logger.info(f"Saved evaluation results to {output_path}")
        return results

    def save_comparison(self, baseline_results, rewritten_results):
        baseline_by_query = {result["query"]: result for result in baseline_results}
        rewritten_by_query = {result["query"]: result for result in rewritten_results}

        comparison = []
        for query in self.load_queries():
            before = baseline_by_query.get(query, {})
            after = rewritten_by_query.get(query, {})
            comparison.append(
                {
                    "query": query,
                    "precision_at_k": {
                        "before": before.get("precision_at_k"),
                        "after": after.get("precision_at_k"),
                    },
                    "coverage": {
                        "before": before.get("coverage"),
                        "after": after.get("coverage"),
                    },
                    "groundedness": {
                        "before": before.get("groundedness", before.get("grounded")),
                        "after": after.get("groundedness", after.get("grounded")),
                    },
                }
            )

        self.save_results(comparison, COMPARISON_PATH)
        logger.info(f"Saved comparison results to {COMPARISON_PATH}")


def main():
    runner = EvaluationRunner()
    results_no_rewrite = runner.evaluate(
        use_query_rewriting=False,
        output_path=RESULTS_NO_REWRITE_PATH,
    )
    results_with_rewrite = runner.evaluate(
        use_query_rewriting=True,
        output_path=RESULTS_WITH_REWRITE_PATH,
    )
    runner.save_comparison(results_no_rewrite, results_with_rewrite)


if __name__ == "__main__":
    main()
