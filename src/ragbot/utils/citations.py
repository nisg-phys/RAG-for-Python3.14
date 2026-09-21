from pathlib import Path

PYTHON_DOCS_BASE_URL = "https://docs.python.org/3/"


def source_to_docs_url(source: str) -> str | None:
    """Map an ingested source path to its page on docs.python.org.

    The corpus under data/ mirrors the official docs' own directory layout
    (e.g. data/library/logging.txt, data/howto/logging.txt, data/about.txt),
    so the mapping is purely mechanical: drop the "data/" prefix and ".txt"
    suffix, keep everything else, append ".html". Returns None for sources
    that don't fit that shape (not a .txt file under data/), since they
    aren't part of this corpus and have no known docs.python.org page.
    """
    path = Path(source)
    parts = path.parts

    if len(parts) < 2 or parts[0] != "data" or path.suffix != ".txt":
        return None

    page_path = "/".join(parts[1:-1] + (path.stem,))
    return f"{PYTHON_DOCS_BASE_URL}{page_path}.html"
