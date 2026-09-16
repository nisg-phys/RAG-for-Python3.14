import tempfile
import unittest
from pathlib import Path

from ragbot.services.ingestion_service import load_documents


class RecursiveIngestionTests(unittest.TestCase):
    def test_load_documents_recurses_into_nested_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "top.txt").write_text("top level text", encoding="utf-8")
            nested_dir = root / "nested" / "deeper"
            nested_dir.mkdir(parents=True)
            (nested_dir / "inner.txt").write_text("nested text", encoding="utf-8")

            documents = load_documents(temp_dir)
            sources = {Path(doc.metadata["source"]).name for doc in documents}

            self.assertEqual(2, len(documents))
            self.assertEqual({"top.txt", "inner.txt"}, sources)


if __name__ == "__main__":
    unittest.main()
