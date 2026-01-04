# Evaluation Guide

This guide explains how to evaluate the email policy network and track improvement across training stages.

## Overview

The evaluation system measures:
- **Action prediction accuracy**: How often the model predicts the correct action
- **Per-action F1 scores**: Precision, recall, and F1 for each action type
- **Priority metrics**: Mean absolute error and correlation with ground truth priority
- **Timing predictions**: Response timing accuracy (when ground truth is available)

## Quick Start

```bash
# Set up virtual environment
uv venv .venv
source .venv/bin/activate  # or: . .venv/bin/activate
uv pip install torch numpy matplotlib

# Run baseline evaluation (untrained model)
python src/evaluate.py --output eval_results/baseline.json

# After training stage 1
python src/evaluate.py --checkpoint checkpoints/stage_1.pt --output eval_results/stage_1.json

# Compare results
python src/compare_results.py eval_results/
```

## Baseline Results

The untrained model achieves approximately **50.86% accuracy** on the test set. This is due to:

1. **Class imbalance**: The dataset is heavily skewed toward "archive" actions
2. **Random initialization bias**: The model tends to predict the majority class

| Metric | Baseline |
|--------|----------|
| Action Accuracy | 50.86% |
| Macro F1 | 0.2401 |
| Reply F1 | 0.1944 |
| Forward F1 | 0.0716 |
| Archive F1 | 0.6885 |
| Delete F1 | 0.0057 |

## Action Mapping

The dataset labels are mapped to model action indices as follows:

| Dataset Label | Model Action (Index) |
|---------------|---------------------|
| REPLIED | reply_now (0) |
| FORWARDED | forward (2) |
| DELETED | delete (4) |
| ARCHIVED | archive (3) |
| AUTO_FILED | archive (3) |
| KEPT | archive (3) |
| COMPOSED | reply_now (0) |
| JUNK | delete (4) |

## Evaluation Scripts

### evaluate.py

Main evaluation script that loads test data, extracts features, runs model inference, and computes metrics.

```bash
# Full usage
python src/evaluate.py --data data/test.json --checkpoint model.pt --output results.json

# Options
--data PATH          Path to test data (default: data/test.json)
--checkpoint PATH    Path to model checkpoint (default: untrained)
--output PATH        Output path for results (default: eval_results/baseline.json)
--limit N            Limit evaluation to N samples
--batch-size N       Batch size for inference (default: 512)
```

### compare_results.py

Compare evaluation results across training stages and generate charts.

```bash
# Compare all results in directory
python src/compare_results.py eval_results/

# Compare specific files
python src/compare_results.py eval_results/baseline.json eval_results/stage_1.json

# Skip chart generation
python src/compare_results.py eval_results/ --no-chart
```

## Output Format

Evaluation results are saved as JSON with the following structure:

```json
{
  "action_accuracy": 0.5086,
  "timing_accuracy": 0.0,
  "action_precision": {"reply_now": 0.30, "forward": 0.04, ...},
  "action_recall": {"reply_now": 0.14, "forward": 0.25, ...},
  "action_f1": {"reply_now": 0.19, "forward": 0.07, ...},
  "priority_mae": 0.168,
  "priority_correlation": 0.035,
  "total_samples": 70207,
  "action_distribution_true": {...},
  "action_distribution_pred": {...},
  "confusion_matrix": {...},
  "inference_time_seconds": 0.08,
  "samples_per_second": 906704,
  "timestamp": "2026-01-03 22:13:50"
}
```

## Training Stage Targets

Based on the training pipeline goals:

| Stage | Method | Target Accuracy |
|-------|--------|-----------------|
| Baseline | Untrained | ~50% (random) |
| 1 | Supervised Fine-Tuning (SFT) | 65-70% |
| 2 | Reward Model Training | - |
| 3 | GRPO | 80-85% |
| 4 | DPO | 88-90% |
| 5 | Temporal RLHF | 92-94% |
| 6 | Rejection Sampling | 95%+ |

## Re-running Evaluation

To re-run evaluation after a training stage:

```bash
# Save checkpoint during training
torch.save({
    'model_state_dict': model.state_dict(),
    'stage': 1,
    'epoch': 10,
}, 'checkpoints/stage_1.pt')

# Run evaluation
python src/evaluate.py --checkpoint checkpoints/stage_1.pt --output eval_results/stage_1.json

# Compare with baseline
python src/compare_results.py eval_results/baseline.json eval_results/stage_1.json
```

## Test Data

The test set contains 70,207 emails from 15 Enron users. Data is split by user to prevent leakage:

- Train: 120 users, 358,665 emails
- Val: 15 users, 88,529 emails
- Test: 15 users, 70,207 emails

Test users: baughman-d, beck-s, benson-r, delainey-d, derrick-j, donohoe-t, ermis-f, geaccone-t, kaminski-v, keavey-p, king-j, lavorato-j, lucci-p, sanders-r, weldon-c
