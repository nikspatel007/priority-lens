#!/usr/bin/env python3
"""Tests for evaluation metrics module.

These tests verify the correctness of classification, regression,
and email-specific custom metrics.
"""

import unittest

# Skip tests if torch is not available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    import sys
    sys.path.insert(0, str(__file__).replace('/tests/test_metrics.py', ''))
    from src.metrics import (
        compute_classification_metrics,
        compute_regression_metrics,
        compute_urgency_metrics,
        compute_ranking_correlation,
        compute_top_k_precision,
        compute_timing_alignment,
        compute_email_metrics,
        evaluate_batch,
        MetricsAccumulator,
        ClassificationMetrics,
        RegressionMetrics,
        EmailMetrics,
        EvaluationResults,
        NUM_ACTION_TYPES,
        NUM_TIMINGS,
    )


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestClassificationMetrics(unittest.TestCase):
    """Tests for classification metrics."""

    def test_perfect_predictions(self):
        """100% accuracy should give perfect scores."""
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        predictions = targets.clone()

        metrics = compute_classification_metrics(predictions, targets, num_classes=3)

        self.assertAlmostEqual(metrics.accuracy, 1.0)
        self.assertAlmostEqual(metrics.macro_f1, 1.0)
        self.assertAlmostEqual(metrics.weighted_f1, 1.0)

    def test_completely_wrong_predictions(self):
        """0% accuracy should give zero scores."""
        targets = torch.tensor([0, 0, 0, 1, 1, 1])
        predictions = torch.tensor([1, 1, 1, 0, 0, 0])  # All wrong

        metrics = compute_classification_metrics(predictions, targets, num_classes=2)

        self.assertAlmostEqual(metrics.accuracy, 0.0)
        # Precision and recall are 0 for each class
        for cls in [0, 1]:
            self.assertAlmostEqual(metrics.precision_per_class[cls], 0.0)
            self.assertAlmostEqual(metrics.recall_per_class[cls], 0.0)

    def test_partial_accuracy(self):
        """Mixed predictions should give intermediate scores."""
        targets = torch.tensor([0, 0, 1, 1])
        predictions = torch.tensor([0, 1, 1, 1])  # 3/4 correct

        metrics = compute_classification_metrics(predictions, targets, num_classes=2)

        self.assertAlmostEqual(metrics.accuracy, 0.75)

    def test_confusion_matrix_shape(self):
        """Confusion matrix should be (num_classes, num_classes)."""
        targets = torch.randint(0, 4, (100,))
        predictions = torch.randint(0, 4, (100,))

        metrics = compute_classification_metrics(predictions, targets, num_classes=4)

        self.assertEqual(metrics.confusion_matrix.shape, (4, 4))

    def test_confusion_matrix_sum(self):
        """Confusion matrix should sum to number of samples."""
        targets = torch.randint(0, 3, (50,))
        predictions = torch.randint(0, 3, (50,))

        metrics = compute_classification_metrics(predictions, targets, num_classes=3)

        self.assertEqual(metrics.confusion_matrix.sum().item(), 50)

    def test_support_per_class(self):
        """Support should count actual occurrences of each class."""
        targets = torch.tensor([0, 0, 0, 1, 1, 2])  # 3 of class 0, 2 of class 1, 1 of class 2
        predictions = torch.randint(0, 3, (6,))

        metrics = compute_classification_metrics(predictions, targets, num_classes=3)

        self.assertEqual(metrics.support_per_class[0], 3)
        self.assertEqual(metrics.support_per_class[1], 2)
        self.assertEqual(metrics.support_per_class[2], 1)

    def test_empty_predictions(self):
        """Empty predictions should return zero metrics."""
        targets = torch.tensor([], dtype=torch.long)
        predictions = torch.tensor([], dtype=torch.long)

        metrics = compute_classification_metrics(predictions, targets, num_classes=2)

        self.assertAlmostEqual(metrics.accuracy, 0.0)

    def test_single_sample(self):
        """Single sample should work correctly."""
        targets = torch.tensor([1])
        predictions = torch.tensor([1])

        metrics = compute_classification_metrics(predictions, targets, num_classes=3)

        self.assertAlmostEqual(metrics.accuracy, 1.0)

    def test_to_dict(self):
        """to_dict should return flat dictionary."""
        targets = torch.randint(0, 3, (20,))
        predictions = torch.randint(0, 3, (20,))

        metrics = compute_classification_metrics(predictions, targets, num_classes=3)
        result = metrics.to_dict()

        self.assertIn('accuracy', result)
        self.assertIn('macro_f1', result)
        self.assertIsInstance(result['accuracy'], float)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestRegressionMetrics(unittest.TestCase):
    """Tests for regression metrics."""

    def test_perfect_predictions(self):
        """Perfect predictions should give MAE=0, R²=1."""
        targets = torch.tensor([0.1, 0.5, 0.9])
        predictions = targets.clone()

        metrics = compute_regression_metrics(predictions, targets)

        self.assertAlmostEqual(metrics.mae, 0.0, places=5)
        self.assertAlmostEqual(metrics.mse, 0.0, places=5)
        self.assertAlmostEqual(metrics.rmse, 0.0, places=5)
        self.assertAlmostEqual(metrics.r2, 1.0, places=5)

    def test_constant_predictions(self):
        """Constant predictions should give R²=0 when predicting mean."""
        targets = torch.tensor([0.0, 0.5, 1.0])
        predictions = torch.tensor([0.5, 0.5, 0.5])  # All mean

        metrics = compute_regression_metrics(predictions, targets)

        # R² = 0 when predicting mean
        self.assertAlmostEqual(metrics.r2, 0.0, places=5)

    def test_mae_calculation(self):
        """MAE should be mean of absolute differences."""
        targets = torch.tensor([0.0, 1.0])
        predictions = torch.tensor([0.2, 0.6])  # Errors: 0.2, 0.4

        metrics = compute_regression_metrics(predictions, targets)

        expected_mae = (0.2 + 0.4) / 2
        self.assertAlmostEqual(metrics.mae, expected_mae, places=5)

    def test_mse_calculation(self):
        """MSE should be mean of squared differences."""
        targets = torch.tensor([0.0, 1.0])
        predictions = torch.tensor([0.2, 0.6])  # Errors: 0.2, 0.4

        metrics = compute_regression_metrics(predictions, targets)

        expected_mse = (0.2**2 + 0.4**2) / 2
        self.assertAlmostEqual(metrics.mse, expected_mse, places=5)

    def test_rmse_is_sqrt_mse(self):
        """RMSE should be square root of MSE."""
        targets = torch.rand(100)
        predictions = targets + 0.1 * torch.randn(100)

        metrics = compute_regression_metrics(predictions, targets)

        self.assertAlmostEqual(metrics.rmse, metrics.mse ** 0.5, places=5)

    def test_empty_predictions(self):
        """Empty predictions should return zero metrics."""
        targets = torch.tensor([])
        predictions = torch.tensor([])

        metrics = compute_regression_metrics(predictions, targets)

        self.assertAlmostEqual(metrics.mae, 0.0)

    def test_to_dict(self):
        """to_dict should return dictionary with all metrics."""
        targets = torch.rand(20)
        predictions = torch.rand(20)

        metrics = compute_regression_metrics(predictions, targets)
        result = metrics.to_dict()

        self.assertIn('mae', result)
        self.assertIn('mse', result)
        self.assertIn('rmse', result)
        self.assertIn('r2', result)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestUrgencyMetrics(unittest.TestCase):
    """Tests for urgency detection metrics."""

    def test_perfect_urgency_detection(self):
        """Perfect detection should give P=R=F1=1."""
        # All urgent (action=0 or timing=0,1)
        action_targets = torch.tensor([0, 1, 2, 3])
        timing_targets = torch.tensor([0, 0, 2, 3])

        action_preds = action_targets.clone()
        timing_preds = timing_targets.clone()

        precision, recall, f1 = compute_urgency_metrics(
            action_preds, action_targets, timing_preds, timing_targets
        )

        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 1.0)

    def test_missed_urgent_emails(self):
        """Missing urgent emails should reduce recall."""
        # Ground truth: indices 0,1 are urgent
        action_targets = torch.tensor([0, 0, 3, 3])  # First two urgent (reply_now)
        timing_targets = torch.tensor([2, 2, 2, 2])

        # Predictions: miss first urgent email
        action_preds = torch.tensor([3, 0, 3, 3])
        timing_preds = torch.tensor([2, 2, 2, 2])

        precision, recall, f1 = compute_urgency_metrics(
            action_preds, action_targets, timing_preds, timing_targets
        )

        # Recall should be 0.5 (found 1 of 2 urgent)
        self.assertAlmostEqual(recall, 0.5)

    def test_false_positive_urgent(self):
        """False positives should reduce precision."""
        # Ground truth: no urgent emails
        action_targets = torch.tensor([3, 3, 3, 3])
        timing_targets = torch.tensor([2, 2, 2, 2])

        # Predictions: wrongly mark first as urgent
        action_preds = torch.tensor([0, 3, 3, 3])
        timing_preds = torch.tensor([2, 2, 2, 2])

        precision, recall, f1 = compute_urgency_metrics(
            action_preds, action_targets, timing_preds, timing_targets
        )

        # Precision should be 0 (0 true positives, 1 false positive)
        self.assertAlmostEqual(precision, 0.0)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestRankingCorrelation(unittest.TestCase):
    """Tests for ranking correlation metrics."""

    def test_perfect_correlation(self):
        """Identical rankings should give correlation of 1."""
        targets = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        predictions = targets.clone()

        spearman, kendall = compute_ranking_correlation(predictions, targets)

        self.assertAlmostEqual(spearman, 1.0, places=5)
        self.assertAlmostEqual(kendall, 1.0, places=5)

    def test_inverted_ranking(self):
        """Inverted rankings should give correlation of -1."""
        targets = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        predictions = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1])  # Reversed

        spearman, kendall = compute_ranking_correlation(predictions, targets)

        self.assertAlmostEqual(spearman, -1.0, places=5)
        self.assertAlmostEqual(kendall, -1.0, places=5)

    def test_single_sample(self):
        """Single sample should return 0."""
        targets = torch.tensor([0.5])
        predictions = torch.tensor([0.5])

        spearman, kendall = compute_ranking_correlation(predictions, targets)

        self.assertAlmostEqual(spearman, 0.0)
        self.assertAlmostEqual(kendall, 0.0)

    def test_two_samples_same_order(self):
        """Two samples in same order should have positive correlation."""
        targets = torch.tensor([0.3, 0.7])
        predictions = torch.tensor([0.2, 0.8])

        spearman, kendall = compute_ranking_correlation(predictions, targets)

        self.assertGreater(spearman, 0)
        self.assertGreater(kendall, 0)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestTopKPrecision(unittest.TestCase):
    """Tests for top-K precision."""

    def test_perfect_top_k(self):
        """Perfect ranking should give precision@k = 1."""
        targets = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])  # 4 is highest, then 3, 2, 1, 0
        predictions = targets.clone()

        result = compute_top_k_precision(predictions, targets, k_values=[1, 2, 3])

        self.assertAlmostEqual(result[1], 1.0)
        self.assertAlmostEqual(result[2], 1.0)
        self.assertAlmostEqual(result[3], 1.0)

    def test_completely_wrong_top_k(self):
        """Inverted ranking should give precision@k = 0 for small k."""
        n = 10
        targets = torch.arange(n, dtype=torch.float32)  # 9 is highest
        predictions = torch.arange(n - 1, -1, -1, dtype=torch.float32)  # 0 is highest

        result = compute_top_k_precision(predictions, targets, k_values=[1, 2])

        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], 0.0)

    def test_k_larger_than_n(self):
        """K larger than n should return 0."""
        targets = torch.tensor([0.1, 0.2, 0.3])  # Only 3 samples
        predictions = targets.clone()

        result = compute_top_k_precision(predictions, targets, k_values=[5, 10])

        self.assertAlmostEqual(result[5], 0.0)
        self.assertAlmostEqual(result[10], 0.0)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestTimingAlignment(unittest.TestCase):
    """Tests for timing alignment metrics."""

    def test_perfect_timing(self):
        """Perfect timing should give MAE=0, within_one=1."""
        targets = torch.tensor([0, 1, 2, 3, 4])
        predictions = targets.clone()

        mae, within_one = compute_timing_alignment(predictions, targets)

        self.assertAlmostEqual(mae, 0.0)
        self.assertAlmostEqual(within_one, 1.0)

    def test_off_by_one_timing(self):
        """Off by one should have MAE=1, within_one=1."""
        targets = torch.tensor([0, 1, 2, 3])
        predictions = torch.tensor([1, 2, 3, 4])  # All off by 1

        mae, within_one = compute_timing_alignment(predictions, targets)

        self.assertAlmostEqual(mae, 1.0)
        self.assertAlmostEqual(within_one, 1.0)  # All within ±1

    def test_off_by_two_timing(self):
        """Off by two should have MAE=2, within_one<1."""
        targets = torch.tensor([0, 0, 0, 0])
        predictions = torch.tensor([2, 2, 2, 2])  # All off by 2

        mae, within_one = compute_timing_alignment(predictions, targets)

        self.assertAlmostEqual(mae, 2.0)
        self.assertAlmostEqual(within_one, 0.0)  # None within ±1


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestEmailMetrics(unittest.TestCase):
    """Tests for combined email metrics."""

    def test_compute_email_metrics_returns_correct_type(self):
        """Should return EmailMetrics dataclass."""
        n = 50
        action_preds = torch.randint(0, NUM_ACTION_TYPES, (n,))
        action_targets = torch.randint(0, NUM_ACTION_TYPES, (n,))
        timing_preds = torch.randint(0, NUM_TIMINGS, (n,))
        timing_targets = torch.randint(0, NUM_TIMINGS, (n,))
        priority_preds = torch.rand(n)
        priority_targets = torch.rand(n)

        metrics = compute_email_metrics(
            action_preds, action_targets,
            timing_preds, timing_targets,
            priority_preds, priority_targets,
        )

        self.assertIsInstance(metrics, EmailMetrics)

    def test_to_dict(self):
        """to_dict should return dictionary with all email metrics."""
        n = 20
        metrics = compute_email_metrics(
            torch.randint(0, NUM_ACTION_TYPES, (n,)),
            torch.randint(0, NUM_ACTION_TYPES, (n,)),
            torch.randint(0, NUM_TIMINGS, (n,)),
            torch.randint(0, NUM_TIMINGS, (n,)),
            torch.rand(n),
            torch.rand(n),
        )

        result = metrics.to_dict()

        self.assertIn('urgency_recall', result)
        self.assertIn('spearman_correlation', result)
        self.assertIn('timing_mae', result)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestEvaluateBatch(unittest.TestCase):
    """Tests for evaluate_batch function."""

    def test_returns_evaluation_results(self):
        """Should return EvaluationResults dataclass."""
        n = 50
        results = evaluate_batch(
            torch.randint(0, NUM_ACTION_TYPES, (n,)),
            torch.randint(0, NUM_ACTION_TYPES, (n,)),
            torch.randint(0, NUM_TIMINGS, (n,)),
            torch.randint(0, NUM_TIMINGS, (n,)),
            torch.rand(n),
            torch.rand(n),
        )

        self.assertIsInstance(results, EvaluationResults)
        self.assertIsInstance(results.action_metrics, ClassificationMetrics)
        self.assertIsInstance(results.timing_metrics, ClassificationMetrics)
        self.assertIsInstance(results.priority_metrics, RegressionMetrics)
        self.assertIsInstance(results.email_metrics, EmailMetrics)

    def test_n_samples_correct(self):
        """n_samples should match input length."""
        n = 42
        results = evaluate_batch(
            torch.randint(0, NUM_ACTION_TYPES, (n,)),
            torch.randint(0, NUM_ACTION_TYPES, (n,)),
            torch.randint(0, NUM_TIMINGS, (n,)),
            torch.randint(0, NUM_TIMINGS, (n,)),
            torch.rand(n),
            torch.rand(n),
        )

        self.assertEqual(results.n_samples, n)

    def test_to_dict(self):
        """to_dict should return flat dictionary with all metrics."""
        results = evaluate_batch(
            torch.randint(0, NUM_ACTION_TYPES, (20,)),
            torch.randint(0, NUM_ACTION_TYPES, (20,)),
            torch.randint(0, NUM_TIMINGS, (20,)),
            torch.randint(0, NUM_TIMINGS, (20,)),
            torch.rand(20),
            torch.rand(20),
        )

        result_dict = results.to_dict()

        self.assertIn('n_samples', result_dict)
        self.assertIn('action_accuracy', result_dict)
        self.assertIn('timing_macro_f1', result_dict)
        self.assertIn('priority_mae', result_dict)
        self.assertIn('email_urgency_recall', result_dict)

    def test_summary_is_string(self):
        """summary should return readable string."""
        results = evaluate_batch(
            torch.randint(0, NUM_ACTION_TYPES, (20,)),
            torch.randint(0, NUM_ACTION_TYPES, (20,)),
            torch.randint(0, NUM_TIMINGS, (20,)),
            torch.randint(0, NUM_TIMINGS, (20,)),
            torch.rand(20),
            torch.rand(20),
        )

        summary = results.summary()

        self.assertIsInstance(summary, str)
        self.assertIn("EVALUATION RESULTS", summary)
        self.assertIn("ACTION CLASSIFICATION", summary)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestMetricsAccumulator(unittest.TestCase):
    """Tests for MetricsAccumulator."""

    def test_accumulate_single_batch(self):
        """Single batch should work correctly."""
        accumulator = MetricsAccumulator()

        n = 20
        accumulator.update(
            torch.randint(0, NUM_ACTION_TYPES, (n,)),
            torch.randint(0, NUM_ACTION_TYPES, (n,)),
            torch.randint(0, NUM_TIMINGS, (n,)),
            torch.randint(0, NUM_TIMINGS, (n,)),
            torch.rand(n),
            torch.rand(n),
        )

        self.assertEqual(accumulator.n_samples, n)

        results = accumulator.compute()
        self.assertEqual(results.n_samples, n)

    def test_accumulate_multiple_batches(self):
        """Multiple batches should accumulate correctly."""
        accumulator = MetricsAccumulator()

        batch_sizes = [10, 15, 25]
        for batch_size in batch_sizes:
            accumulator.update(
                torch.randint(0, NUM_ACTION_TYPES, (batch_size,)),
                torch.randint(0, NUM_ACTION_TYPES, (batch_size,)),
                torch.randint(0, NUM_TIMINGS, (batch_size,)),
                torch.randint(0, NUM_TIMINGS, (batch_size,)),
                torch.rand(batch_size),
                torch.rand(batch_size),
            )

        self.assertEqual(accumulator.n_samples, sum(batch_sizes))

        results = accumulator.compute()
        self.assertEqual(results.n_samples, sum(batch_sizes))

    def test_reset(self):
        """Reset should clear all accumulated data."""
        accumulator = MetricsAccumulator()

        accumulator.update(
            torch.randint(0, NUM_ACTION_TYPES, (20,)),
            torch.randint(0, NUM_ACTION_TYPES, (20,)),
            torch.randint(0, NUM_TIMINGS, (20,)),
            torch.randint(0, NUM_TIMINGS, (20,)),
            torch.rand(20),
            torch.rand(20),
        )

        self.assertEqual(accumulator.n_samples, 20)

        accumulator.reset()

        self.assertEqual(accumulator.n_samples, 0)

    def test_empty_accumulator(self):
        """Empty accumulator should return zero metrics."""
        accumulator = MetricsAccumulator()

        results = accumulator.compute()

        self.assertEqual(results.n_samples, 0)

    def test_gpu_tensors(self):
        """Should work with GPU tensors (moved to CPU internally)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        accumulator = MetricsAccumulator()

        n = 20
        accumulator.update(
            torch.randint(0, NUM_ACTION_TYPES, (n,)).cuda(),
            torch.randint(0, NUM_ACTION_TYPES, (n,)).cuda(),
            torch.randint(0, NUM_TIMINGS, (n,)).cuda(),
            torch.randint(0, NUM_TIMINGS, (n,)).cuda(),
            torch.rand(n).cuda(),
            torch.rand(n).cuda(),
        )

        results = accumulator.compute()
        self.assertEqual(results.n_samples, n)


if __name__ == '__main__':
    unittest.main()
