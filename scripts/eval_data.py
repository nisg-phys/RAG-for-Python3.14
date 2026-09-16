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
        "question": "What is the purpose of the walrus operator in Python?",
        "ground_truth": (
            "It allows for the assignment of a value to a variable as part of a "
            "larger expression, returning the value of the expression [4]."
        ),
    },
    {
        "question": "How does pattern matching work in Python?",
        "ground_truth": (
            "It is a declarative way to inspect and unpack complex data "
            "structures using match and case statements [5]."
        ),
    },
    {
        "question": "What is the difference between __slots__ and __dict__?",
        "ground_truth": (
            "__dict__ is a dynamic hash table with high memory overhead, while "
            "__slots__ uses a compact, fixed-size array to reduce memory "
            "footprint and increase attribute access speed [6]."
        ),
    },
    {
        "question": "What are type aliases in Python 3.14?",
        "ground_truth": (
            "They are defined using the type statement, and in Python 3.14, "
            "TypeAliasType now supports star unpacking [7, 8]."
        ),
    },
    {
        "question": "How do you use TypeVar in generic functions?",
        "ground_truth": (
            "It is used via modern square-bracket syntax for inline generic "
            "definitions, which implicitly handles the creation and scoping of "
            "the type variable [9]."
        ),
    },
    {
        "question": "What is the difference between Optional and Union types?",
        "ground_truth": (
            "Optional[int] is a specific union form commonly written as int | "
            "None using the pipe operator syntax [10]."
        ),
    },
    {
        "question": "What performance improvements were made in Python 3.14?",
        "ground_truth": (
            "An experimental JIT compiler (20-50% speedup), a tail-call-based "
            "interpreter (3-5% speedup), and an incremental garbage collector "
            "that reduces pause times [11-14]."
        ),
    },
    {
        "question": "How does the new GIL removal affect multithreading?",
        "ground_truth": (
            "It enables true multi-core parallelism for CPU-bound tasks by "
            "allowing threads to run bytecodes simultaneously without a global "
            "lock [15, 16]."
        ),
    },
    {
        "question": "How does the tomllib module work?",
        "ground_truth": (
            "It provides native support for parsing the TOML configuration "
            "format and features improved error reporting for syntax errors in "
            "3.14 [17]."
        ),
    },
    {
        "question": "What is the difference between asyncio.gather and asyncio.TaskGroup?",
        "ground_truth": (
            "asyncio.gather is unstructured and continues sibling tasks on "
            "failure, while asyncio.TaskGroup is a structured context manager "
            "that cancels sibling tasks when an exception occurs [18]."
        ),
    },
    {
        "question": "How do you use the match statement with dataclasses?",
        "ground_truth": (
            "Dataclasses automatically define the __match_args__ attribute, "
            "allowing the match statement to use positional arguments that "
            "mirror the object's constructor [5]."
        ),
    },
    {
        "question": "What are exception groups in Python?",
        "ground_truth": (
            "They are special exceptions that wrap multiple, unrelated errors "
            "occurring simultaneously during concurrent operations [19]."
        ),
    },
    {
        "question": "How does ExceptionGroup differ from regular exceptions?",
        "ground_truth": (
            "Unlike regular exceptions, ExceptionGroup requires the except* "
            "syntax to selectively extract and handle specific types of errors "
            "within the group [20]."
        ),
    },
    {
        "question": "How does Python 3.14 handle memory management differently?",
        "ground_truth": (
            "It uses an incremental garbage collector with only two generations "
            "and adopts the mimalloc allocator as the default for free-threaded "
            "builds [21-23]."
        ),
    },
]
