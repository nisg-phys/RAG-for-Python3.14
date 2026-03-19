import re

def format_to_markdown(text: str) -> str:
    text = re.sub(r"\*\*Explanation\*\*\s*=+", "## Explanation", text)
    text = re.sub(r"\*\*Code Example\*\*\s*=+", "## Code", text)
    text = re.sub(r"\*\*Notes\*\*\s*=+", "## Notes", text)

    return text.strip()