"""Tests for ragbot.utils.formatter.format_to_markdown.

format_to_markdown is a safety net: the RAG prompt instructs the model to
emit "## Answer" / "## Evidence" / "## Code Example(s)" / "## Notes"
headings directly, but models occasionally drift (plain text, bold, or a
different heading level). These tests lock in that normalization and guard
against a regex regression where a greedy `\\s*` swallowed the blank line
separating sections (fixed by restricting intra-line matches to `[ \\t]*`).
"""

import unittest

from ragbot.utils.formatter import format_to_markdown


class FormatToMarkdownTests(unittest.TestCase):
    def test_normalizes_plain_text_section_labels_to_headings(self):
        text = (
            "Answer\n"
            "Logging uses the logging module.\n\n"
            "Evidence\n"
            "- loggers expose the interface\n\n"
            "Notes\n"
            "No caveats."
        )

        result = format_to_markdown(text)

        self.assertIn("## Answer", result)
        self.assertIn("## Evidence", result)
        self.assertIn("## Notes", result)

    def test_normalizes_bold_section_labels(self):
        text = "**Answer**\nSome text.\n\n**Evidence**:\n- a bullet"

        result = format_to_markdown(text)

        self.assertIn("## Answer", result)
        self.assertIn("## Evidence", result)
        self.assertNotIn("**Answer**", result)

    def test_code_example_s_heading_is_recognized(self):
        text = "Code Example(s)\n```python\nimport logging\n```"

        result = format_to_markdown(text)

        self.assertIn("## Code Example(s)", result)

    def test_already_correct_markdown_is_left_unchanged(self):
        text = "## Answer\ntext\n\n## Evidence\n- bullet"

        result = format_to_markdown(text)

        self.assertEqual(text, result)

    def test_blank_line_between_sections_is_preserved(self):
        text = "Answer\nLogging text.\n\nEvidence\n- quote one"

        result = format_to_markdown(text)

        self.assertEqual(
            "## Answer\nLogging text.\n\n## Evidence\n- quote one",
            result,
        )

    def test_does_not_touch_unrelated_lines(self):
        text = "Answer\nEvidence supports this claim directly in the body."

        result = format_to_markdown(text)

        # Only the standalone "Answer" line becomes a heading; the sentence
        # that happens to start with "Evidence" but has trailing prose is not
        # a section label and must be left alone.
        self.assertIn("## Answer", result)
        self.assertIn("Evidence supports this claim directly in the body.", result)
        self.assertNotIn("## Evidence supports", result)


if __name__ == "__main__":
    unittest.main()
