import unittest

from eval_stats import bootstrap_ci, bootstrap_confidence_intervals, summarize_per_question


class EvalStatsTest(unittest.TestCase):
    def test_summarize_per_question(self):
        rows = [
            {"correct": 1, "partial_correct": 1, "format_correct": 1},
            {"correct": 0, "partial_correct": 1, "format_correct": 1},
            {"correct": 0, "partial_correct": 0, "format_correct": 1},
            {"correct": 1, "partial_correct": 1, "format_correct": 0},
        ]
        summary = summarize_per_question(rows)
        self.assertEqual(summary["correct"], 2)
        self.assertEqual(summary["partial_correct"], 3)
        self.assertEqual(summary["format_correct"], 3)
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["accuracy"], 50.0)
        self.assertEqual(summary["partial_accuracy"], 75.0)
        self.assertEqual(summary["format_accuracy"], 75.0)

    def test_bootstrap_ci_is_deterministic(self):
        values = [1, 0, 1, 1, 0]
        first = bootstrap_ci(values, num_samples=100, confidence=0.9, seed=123)
        second = bootstrap_ci(values, num_samples=100, confidence=0.9, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(first["confidence"], 0.9)
        self.assertEqual(first["num_samples"], 100)
        self.assertEqual(first["seed"], 123)
        self.assertIsNotNone(first["lower"])
        self.assertIsNotNone(first["upper"])
        self.assertLessEqual(first["lower"], first["upper"])

    def test_bootstrap_can_be_disabled(self):
        interval = bootstrap_ci([1, 0, 1], num_samples=0)
        self.assertIsNone(interval["lower"])
        self.assertIsNone(interval["upper"])

    def test_bootstrap_confidence_intervals(self):
        rows = [
            {"correct": 1, "partial_correct": 1, "format_correct": 1},
            {"correct": 0, "partial_correct": 1, "format_correct": 0},
        ]
        intervals = bootstrap_confidence_intervals(rows, num_samples=10, seed=0)
        self.assertEqual(set(intervals), {"accuracy", "partial_accuracy", "format_accuracy"})


if __name__ == "__main__":
    unittest.main()
