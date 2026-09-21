import json
import importlib.util
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ragbot.evaluation.metrics import compute_query_metrics, normalize_eval_specs, summarize_metrics
from ragbot.config.settings import settings
from ragbot.observability.telemetry import record_event
from ragbot.pipeline.rag_pipeline import RAGPipeline
from ragbot.utils.logger import get_logger


logger = get_logger("evaluation")

EVAL_DATA_PATH = Path("evaluation/eval_data.py")
LEGACY_EVAL_DATA_PATH = Path("scripts/eval_data.py")
QUERIES_PATH = Path("evaluation/queries.json")
RESULTS_PATH = Path("evaluation/results.json")
RESULTS_NO_REWRITE_PATH = Path("evaluation/results_no_rewrite.json")
RESULTS_WITH_REWRITE_PATH = Path("evaluation/results_with_rewrite.json")
COMPARISON_PATH = Path("evaluation/comparison.json")
USE_QUERY_REWRITING = False


class EvaluationRunner:
    def __init__(self):
        self.pipeline = RAGPipeline()

    def _load_eval_data_module(self, path: Path):
        spec = importlib.util.spec_from_file_location("ragbot_eval_data", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load evaluation dataset module from {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def load_queries(self):
        for candidate in (EVAL_DATA_PATH, LEGACY_EVAL_DATA_PATH):
            if candidate.exists():
                logger.info("Loading evaluation dataset from %s", candidate)
                module = self._load_eval_data_module(candidate)
                return normalize_eval_specs(getattr(module, "eval_dataset", []))

        logger.info("Loading evaluation dataset from %s", QUERIES_PATH)
        with open(QUERIES_PATH, "r", encoding="utf-8") as f:
            return normalize_eval_specs(json.load(f))

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

        for query_spec in queries:
            query = query_spec["query"]
            expected_keywords = query_spec.get("expected_keywords", [])
            ground_truth = query_spec.get("ground_truth")
            logger.info(f"Evaluating query: {query}")
            output = self.pipeline.run(query)

            answer, retrieved_chunks = self._normalize_output(query, output)
            chunk_texts = self._extract_chunk_texts(retrieved_chunks)
            results.append(
                compute_query_metrics(
                    query=query,
                    answer=answer,
                    chunk_texts=chunk_texts,
                    expected_keywords=expected_keywords,
                    ground_truth=ground_truth,
                    pipeline_metrics=output.get("metrics", {}) if isinstance(output, dict) else {},
                )
            )

        summary = summarize_metrics(results)
        payload = {
            "use_query_rewriting": use_query_rewriting,
            "summary": summary.model_dump(),
            "queries": results,
        }
        self.save_results(payload, output_path)
        record_event("evaluation", payload["summary"])
        logger.info(f"Saved evaluation results to {output_path}")
        return payload

    def save_comparison(self, baseline_results, rewritten_results):
        before = baseline_results["summary"]
        after = rewritten_results["summary"]
        comparison = {
            "before": before,
            "after": after,
            "delta": {
                metric: round(after[metric] - before[metric], 4)
                for metric in before
                if metric != "queries_evaluated"
            },
        }

        self.save_results(comparison, COMPARISON_PATH)
        logger.info(f"Saved comparison results to {COMPARISON_PATH}")


def main():
    runner = EvaluationRunner()
    results_no_rewrite = runner.evaluate(
        use_query_rewriting=False,
        output_path=RESULTS_NO_REWRITE_PATH,
    )

    if not settings.use_query_rewriting:
        runner.save_results(results_no_rewrite, RESULTS_PATH)
        logger.info(
            "Query rewriting is disabled in settings; skipped rewritten evaluation outputs."
        )
        return

    results_with_rewrite = runner.evaluate(
        use_query_rewriting=True,
        output_path=RESULTS_WITH_REWRITE_PATH,
    )
    runner.save_comparison(results_no_rewrite, results_with_rewrite)
    runner.save_results(results_no_rewrite, RESULTS_PATH)


if __name__ == "__main__":
    main()
