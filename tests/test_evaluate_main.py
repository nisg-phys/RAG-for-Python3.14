import unittest
from unittest.mock import patch

import scripts.evaluate as evaluate_module


class EvaluateMainTests(unittest.TestCase):
    def test_main_skips_rewrite_outputs_when_rewriting_disabled(self):
        runner = unittest.mock.Mock()
        runner.evaluate.return_value = {"summary": {"queries_evaluated": 1}}

        with patch.object(evaluate_module, "EvaluationRunner", return_value=runner), patch.object(
            evaluate_module.settings,
            "use_query_rewriting",
            False,
        ):
            evaluate_module.main()

        runner.evaluate.assert_called_once()
        runner.save_comparison.assert_not_called()
        runner.save_results.assert_called_once()


if __name__ == "__main__":
    unittest.main()
