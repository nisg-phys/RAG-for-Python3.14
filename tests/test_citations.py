"""Tests for ragbot.utils.citations.source_to_docs_url.

The ingested corpus under data/ mirrors docs.python.org's own directory
layout (data/library/logging.txt, data/howto/logging.txt, data/about.txt,
...), so mapping an ingested source back to its official docs page is a
mechanical path rewrite. These tests lock that mapping in and its fallback
behavior for anything that doesn't fit the expected shape.
"""

import unittest

from ragbot.utils.citations import source_to_docs_url


class SourceToDocsUrlTests(unittest.TestCase):
    def test_maps_a_subdirectory_page(self):
        self.assertEqual(
            "https://docs.python.org/3/library/logging.html",
            source_to_docs_url("data/library/logging.txt"),
        )

    def test_maps_a_howto_page(self):
        self.assertEqual(
            "https://docs.python.org/3/howto/logging-cookbook.html",
            source_to_docs_url("data/howto/logging-cookbook.txt"),
        )

    def test_maps_a_top_level_page(self):
        self.assertEqual(
            "https://docs.python.org/3/about.html",
            source_to_docs_url("data/about.txt"),
        )

    def test_maps_a_dotted_module_name(self):
        self.assertEqual(
            "https://docs.python.org/3/library/logging.config.html",
            source_to_docs_url("data/library/logging.config.txt"),
        )

    def test_returns_none_for_non_txt_source(self):
        self.assertIsNone(source_to_docs_url("data/library/logging.pdf"))

    def test_returns_none_for_source_outside_data_dir(self):
        self.assertIsNone(source_to_docs_url("other/library/logging.txt"))

    def test_returns_none_for_bare_filename(self):
        self.assertIsNone(source_to_docs_url("logging.txt"))

    def test_returns_none_for_unknown_source_value(self):
        self.assertIsNone(source_to_docs_url("unknown"))


if __name__ == "__main__":
    unittest.main()
