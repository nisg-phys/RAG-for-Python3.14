"""User-editable evaluation dataset for RAGAS runs.

Each sample must contain:
- question: the input prompt sent to the RAG pipeline
- ground_truth: the expected reference answer

Optional fields:
- source: a trusted reference URL for the ground truth

Keep this file simple so non-engineers can update it safely.
"""

from __future__ import annotations


eval_dataset: list[dict[str, str]] = [
    {
        "question": "What performance improvements were made in Python 3.14?",
        "ground_truth": (
            "Python 3.14 includes an experimental JIT compiler with roughly "
            "20-50% speedups in suitable workloads, a tail-call-based "
            "interpreter with roughly 3-5% speedups, and an incremental "
            "garbage collector that reduces pause times."
        ),
    },
    {
        "question": "How does the new GIL removal affect multithreading in Python 3.14?",
        "ground_truth": (
            "The free-threaded build removes the global interpreter lock so "
            "CPU-bound threads can run in parallel across multiple cores, but "
            "extension modules and code paths must be thread-safe."
        ),
    },
    {
        "question": "How does Python 3.14 handle memory management differently?",
        "ground_truth": (
            "Python 3.14 uses an incremental garbage collector with two "
            "generations and adopts mimalloc as the default allocator for "
            "free-threaded builds."
        ),
    },
    {
        "question": "What changed for type aliases in Python 3.14?",
        "ground_truth": (
            "Type aliases are still defined with the type statement, and "
            "TypeAliasType in Python 3.14 adds support for star unpacking."
        ),
    },
    {
        "question": "What changed in tomllib error reporting in Python 3.14?",
        "ground_truth": (
            "The tomllib module still parses TOML, and Python 3.14 improves "
            "syntax error reporting to make invalid TOML input easier to debug."
        ),
    },
    {
        "question": "How does asyncio.TaskGroup differ from asyncio.gather in Python 3.14?",
        "ground_truth": (
            "asyncio.gather is unstructured and can leave sibling tasks running "
            "after a failure, while asyncio.TaskGroup is structured and cancels "
            "remaining sibling tasks when an exception occurs."
        ),
    },
    {
        "question": "What are exception groups in modern Python?",
        "ground_truth": (
            "Exception groups bundle multiple unrelated exceptions that happen "
            "at the same time, especially in concurrent code, into one object."
        ),
    },
    {
        "question": "How does ExceptionGroup differ from a regular exception?",
        "ground_truth": (
            "ExceptionGroup contains multiple exceptions and is handled with "
            "except* so matching error types can be extracted selectively, "
            "unlike regular exceptions which are handled one at a time."
        ),
    },
    {
        "question": "How do you use the walrus operator in Python?",
        "ground_truth": (
            "It allows for the assignment of a value to a variable as part of a "
            "larger expression, returning the value of the expression."
        ),
    },
    {
        "question": "How does pattern matching work in Python?",
        "ground_truth": (
            "It is a declarative way to inspect and unpack complex data "
            "structures using match and case statements."
        ),
    },
    {
        "question": "What is the difference between __slots__ and __dict__?",
        "ground_truth": (
            "__dict__ is a dynamic hash table with high memory overhead, while "
            "__slots__ uses a compact, fixed-size array to reduce memory "
            "footprint and increase attribute access speed."
        ),
    },
    {
        "question": "How do you use TypeVar in generic functions?",
        "ground_truth": (
            "Modern generic functions can use square-bracket type parameter "
            "syntax inline, which defines and scopes the type variable directly."
        ),
    },
    {
        "question": "What is the difference between Optional and Union types?",
        "ground_truth": (
            "Optional[int] is a special case of a union and is commonly written "
            "as int | None in modern type syntax."
        ),
    },
    {
        "question": "How do you use the match statement with dataclasses?",
        "ground_truth": (
            "Dataclasses define __match_args__ automatically so match statements "
            "can use positional patterns that mirror constructor fields."
        ),
    },
]
