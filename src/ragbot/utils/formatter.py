import re

SECTION_NAMES = ["Answer", "Evidence", "Code Example(s)", "Code Examples", "Notes"]


def format_to_markdown(text: str) -> str:
    """Normalize section headers to Markdown "## " headings.

    The prompt instructs the model to emit "## Answer" etc. directly, but
    models occasionally drift (plain text, bold, or a different heading
    level), so this catches those variants as a safety net.
    """
    for name in SECTION_NAMES:
        escaped = re.escape(name)
        text = re.sub(
            rf"^[ \t]*#{{0,6}}[ \t]*\*{{0,2}}{escaped}\*{{0,2}}:?[ \t]*$",
            f"## {name}",
            text,
            flags=re.MULTILINE,
        )

    return text.strip()