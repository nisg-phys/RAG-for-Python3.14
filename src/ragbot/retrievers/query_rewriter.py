import ast

from ragbot.utils.logger import get_logger


logger = get_logger("query_rewriter")


class QueryRewriter:
    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, query: str) -> list[str]:
        prompt = f"""
You rewrite Python documentation search queries for retrieval.

Generate exactly 3 alternative rewrites for the query below.

Requirements:
- Use more explicit technical wording
- Expand abbreviations where helpful
- Include synonyms only if they are relevant
- Stay within the Python documentation domain
- Return only a valid Python list of strings

Query:
{query}
""".strip()

        try:
            response = self.llm.invoke(prompt)
            content = str(getattr(response, "content", response)).strip()
            parsed = ast.literal_eval(content)

            if not isinstance(parsed, list):
                raise ValueError("LLM did not return a list")

            rewrites = []
            seen = {query.strip().lower()}

            for item in parsed:
                if not isinstance(item, str):
                    continue
                cleaned = item.strip()
                if not cleaned:
                    continue
                normalized = cleaned.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                rewrites.append(cleaned)

            return rewrites[:3]
        except Exception as exc:
            logger.warning("Query rewriting failed for query '%s': %s", query, exc)
            return []
