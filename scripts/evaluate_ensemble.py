#!/usr/bin/env python3
"""Evaluate ensemble vs individual models.

Compares:
1. Individual model performance
2. Different ensemble methods (voting, averaging, weighted avg, stacking)
3. Per-class accuracy improvements

Usage:
    # Evaluate ensemble on SurrealDB data
    python scripts/evaluate_ensemble.py --checkpoint checkpoints/ensemble --database enron

    # Evaluate on JSON test data
    python scripts/evaluate_ensemble.py --checkpoint checkpoints/ensemble --test-data data/test.json

    # Train and evaluate stacking meta-classifier
    python scripts/evaluate_ensemble.py --checkpoint checkpoints/ensemble --database enron --train-stacking
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.ensemble import EnsemblePredictor, EnsembleMethod
from src.sft_training import collate_fn, IDX_TO_ACTION, NUM_ACTION_TYPES


class EmailListDataset(Dataset):
    """Simple dataset wrapping a list of (email, label) pairs."""

    def __init__(self, emails: list[dict], labels: list[int]):
        self.emails = emails
        self.labels = labels

    def __len__(self):
        return len(self.emails)

    def __getitem__(self, idx):
        return self.emails[idx], self.labels[idx]


def evaluate_individual_models(
    ensemble: EnsemblePredictor,
    emails: list[dict],
    labels: torch.Tensor,
) -> dict[str, dict]:
    """Evaluate each model in the ensemble individually.

    Args:
        ensemble: Ensemble predictor with loaded models
        emails: List of email dictionaries
        labels: Ground truth labels

    Returns:
        Dictionary mapping model names to evaluation metrics
    """
    results = {}

    for member in ensemble.members:
        # Extract features for this member
        features = ensemble._extract_features_batch(emails, member)

        # Get predictions
        with torch.no_grad():
            output = member.model(features)
            preds = output.action_logits.argmax(dim=-1)

        # Compute metrics
        labels_device = labels.to(ensemble.device)
        correct = (preds == labels_device).sum().item()
        total = len(labels)

        # Per-class accuracy
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        for i in range(len(labels)):
            label = labels[i].item()
            pred = preds[i].item()
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

        per_class_acc = {}
        for cls in range(NUM_ACTION_TYPES):
            if class_total[cls] > 0:
                per_class_acc[IDX_TO_ACTION[cls]] = class_correct[cls] / class_total[cls]
            else:
                per_class_acc[IDX_TO_ACTION[cls]] = 0.0

        results[member.name] = {
            'accuracy': correct / total,
            'total_samples': total,
            'per_class_accuracy': per_class_acc,
        }

    return results


def evaluate_ensemble_methods(
    ensemble: EnsemblePredictor,
    emails: list[dict],
    labels: torch.Tensor,
    include_stacking: bool = False,
) -> dict[str, dict]:
    """Evaluate different ensemble combination methods.

    Args:
        ensemble: Ensemble predictor
        emails: List of email dictionaries
        labels: Ground truth labels
        include_stacking: Whether to include stacking method

    Returns:
        Dictionary mapping method names to evaluation metrics
    """
    results = {}
    labels_device = labels.to(ensemble.device)

    methods = [
        EnsembleMethod.HARD_VOTING,
        EnsembleMethod.SOFT_VOTING,
        EnsembleMethod.WEIGHTED_AVG,
    ]

    if include_stacking and ensemble._meta_classifier is not None:
        methods.append(EnsembleMethod.STACKING)

    for method in methods:
        # Get predictions
        probs = ensemble.predict_proba(emails, method=method)
        preds = probs.argmax(dim=-1)

        # Compute metrics
        correct = (preds == labels_device).sum().item()
        total = len(labels)

        # Per-class accuracy
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        for i in range(len(labels)):
            label = labels[i].item()
            pred = preds[i].item()
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

        per_class_acc = {}
        for cls in range(NUM_ACTION_TYPES):
            if class_total[cls] > 0:
                per_class_acc[IDX_TO_ACTION[cls]] = class_correct[cls] / class_total[cls]
            else:
                per_class_acc[IDX_TO_ACTION[cls]] = 0.0

        results[method.value] = {
            'accuracy': correct / total,
            'total_samples': total,
            'per_class_accuracy': per_class_acc,
        }

    return results


def analyze_agreement(
    ensemble: EnsemblePredictor,
    emails: list[dict],
    labels: torch.Tensor,
) -> dict:
    """Analyze ensemble agreement and its correlation with accuracy.

    Args:
        ensemble: Ensemble predictor
        emails: List of email dictionaries
        labels: Ground truth labels

    Returns:
        Agreement analysis metrics
    """
    # Get agreement scores
    agreement = ensemble.get_agreement_score(emails)
    labels_device = labels.to(ensemble.device)

    # Get predictions
    probs = ensemble.predict_proba(emails, method=EnsembleMethod.WEIGHTED_AVG)
    preds = probs.argmax(dim=-1)

    # Bin by agreement level
    bins = [(0.0, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.01)]
    bin_results = {}

    for low, high in bins:
        mask = (agreement >= low) & (agreement < high)
        if mask.sum() == 0:
            continue

        bin_correct = ((preds == labels_device) & mask).sum().item()
        bin_total = mask.sum().item()

        bin_name = f"{low:.0%}-{high:.0%}" if high < 1.01 else "100%"
        bin_results[bin_name] = {
            'count': bin_total,
            'accuracy': bin_correct / bin_total if bin_total > 0 else 0,
        }

    return {
        'mean_agreement': agreement.mean().item(),
        'min_agreement': agreement.min().item(),
        'max_agreement': agreement.max().item(),
        'agreement_by_bin': bin_results,
    }


def print_results(
    individual_results: dict[str, dict],
    ensemble_results: dict[str, dict],
    agreement_analysis: dict,
) -> None:
    """Print formatted evaluation results."""
    print("\n" + "=" * 70)
    print("ENSEMBLE EVALUATION RESULTS")
    print("=" * 70)

    # Individual models
    print("\n" + "-" * 70)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("-" * 70)
    print(f"{'Model':<15} {'Accuracy':>10} {'reply_now':>10} {'reply_later':>12} {'forward':>10} {'archive':>10} {'delete':>10}")
    print("-" * 70)

    for name, metrics in sorted(individual_results.items(), key=lambda x: -x[1]['accuracy']):
        per_class = metrics['per_class_accuracy']
        print(f"{name:<15} {metrics['accuracy']:>10.1%} "
              f"{per_class.get('reply_now', 0):>10.1%} "
              f"{per_class.get('reply_later', 0):>12.1%} "
              f"{per_class.get('forward', 0):>10.1%} "
              f"{per_class.get('archive', 0):>10.1%} "
              f"{per_class.get('delete', 0):>10.1%}")

    # Ensemble methods
    print("\n" + "-" * 70)
    print("ENSEMBLE METHOD PERFORMANCE")
    print("-" * 70)
    print(f"{'Method':<15} {'Accuracy':>10} {'reply_now':>10} {'reply_later':>12} {'forward':>10} {'archive':>10} {'delete':>10}")
    print("-" * 70)

    for name, metrics in sorted(ensemble_results.items(), key=lambda x: -x[1]['accuracy']):
        per_class = metrics['per_class_accuracy']
        print(f"{name:<15} {metrics['accuracy']:>10.1%} "
              f"{per_class.get('reply_now', 0):>10.1%} "
              f"{per_class.get('reply_later', 0):>12.1%} "
              f"{per_class.get('forward', 0):>10.1%} "
              f"{per_class.get('archive', 0):>10.1%} "
              f"{per_class.get('delete', 0):>10.1%}")

    # Best comparison
    best_individual = max(individual_results.items(), key=lambda x: x[1]['accuracy'])
    best_ensemble = max(ensemble_results.items(), key=lambda x: x[1]['accuracy'])

    print("\n" + "-" * 70)
    print("BEST COMPARISON")
    print("-" * 70)
    print(f"Best individual: {best_individual[0]} ({best_individual[1]['accuracy']:.1%})")
    print(f"Best ensemble:   {best_ensemble[0]} ({best_ensemble[1]['accuracy']:.1%})")

    improvement = best_ensemble[1]['accuracy'] - best_individual[1]['accuracy']
    print(f"Improvement:     {improvement:+.1%}")

    # Agreement analysis
    print("\n" + "-" * 70)
    print("AGREEMENT ANALYSIS")
    print("-" * 70)
    print(f"Mean agreement: {agreement_analysis['mean_agreement']:.1%}")
    print(f"Min agreement:  {agreement_analysis['min_agreement']:.1%}")
    print(f"Max agreement:  {agreement_analysis['max_agreement']:.1%}")

    print("\nAccuracy by agreement level:")
    for bin_name, bin_metrics in agreement_analysis['agreement_by_bin'].items():
        print(f"  {bin_name}: {bin_metrics['accuracy']:.1%} ({bin_metrics['count']} samples)")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ensemble vs individual models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='Path to ensemble checkpoint directory',
    )

    # Data source options
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        '--test-data',
        type=Path,
        help='Path to test data JSON',
    )
    data_group.add_argument(
        '--database',
        type=str,
        help='SurrealDB database name (enron or gmail)',
    )
    data_group.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data for testing',
    )

    parser.add_argument(
        '--train-stacking',
        action='store_true',
        help='Train stacking meta-classifier before evaluation',
    )
    parser.add_argument(
        '--stacking-epochs',
        type=int,
        default=10,
        help='Epochs for stacking training (default: 10)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use (default: auto)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Optional path to save results JSON',
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of test samples',
    )

    args = parser.parse_args()

    # Load ensemble
    print(f"Loading ensemble from {args.checkpoint}...")
    ensemble = EnsemblePredictor.from_checkpoint_dir(args.checkpoint, device=args.device)
    print(f"Loaded {len(ensemble.members)} models")

    # Load test data
    if args.synthetic:
        from src.sft_training import create_synthetic_dataset, ACTION_TO_IDX
        print("Using synthetic test data...")
        test_emails = create_synthetic_dataset(num_samples=args.limit or 500, seed=456)
        labels = torch.tensor([
            ACTION_TO_IDX.get(e['action'].upper(), 3) for e in test_emails
        ])
    elif args.database:
        try:
            from db.dataset import SurrealEmailDataset, ACTION_TO_IDX
            print(f"Loading test data from SurrealDB (database={args.database})...")

            test_ds = SurrealEmailDataset(
                database=args.database,
                split='test',
                use_cached_embeddings=False,
            )

            test_emails = test_ds._emails
            if args.limit:
                test_emails = test_emails[:args.limit]

            # Extract labels
            labels = torch.tensor([
                ACTION_TO_IDX.get(e.get('action', 'KEPT'), 4) for e in test_emails
            ])

            print(f"Loaded {len(test_emails)} test emails")

        except Exception as e:
            print(f"Error loading from SurrealDB: {e}")
            sys.exit(1)
    else:
        from src.sft_training import load_emails, ACTION_TO_IDX
        test_emails = load_emails(args.test_data, args.limit)
        labels = torch.tensor([
            ACTION_TO_IDX.get(e.get('action', 'KEPT').upper(), 3) for e in test_emails
        ])

    # Convert emails to proper format if needed (SurrealDB uses different keys)
    formatted_emails = []
    for email in test_emails:
        formatted = {
            'from': email.get('from_email', email.get('from', '')),
            'to': email.get('to_emails', email.get('to', '')),
            'cc': email.get('cc_emails', email.get('cc', '')),
            'subject': email.get('subject', ''),
            'body': email.get('body', ''),
            'x_from': email.get('x_from', ''),
            'x_to': email.get('x_to', ''),
            'date': email.get('date', ''),
        }
        # Convert list fields to strings if needed
        if isinstance(formatted['to'], list):
            formatted['to'] = ','.join(formatted['to'])
        if isinstance(formatted['cc'], list):
            formatted['cc'] = ','.join(formatted['cc'])
        formatted_emails.append(formatted)

    # Train stacking if requested
    if args.train_stacking:
        print("\nTraining stacking meta-classifier...")

        # Need train/val data for stacking
        if args.database:
            from db.dataset import SurrealEmailDataset
            train_ds = SurrealEmailDataset(database=args.database, split='train', use_cached_embeddings=False)
            val_ds = SurrealEmailDataset(database=args.database, split='val', use_cached_embeddings=False)

            train_emails = train_ds._emails
            val_emails_raw = val_ds._emails

            train_labels = torch.tensor([ACTION_TO_IDX.get(e.get('action', 'KEPT'), 4) for e in train_emails])
            val_labels = torch.tensor([ACTION_TO_IDX.get(e.get('action', 'KEPT'), 4) for e in val_emails_raw])

            # Format emails
            train_formatted = []
            for email in train_emails:
                formatted = {
                    'from': email.get('from_email', email.get('from', '')),
                    'to': email.get('to_emails', email.get('to', '')),
                    'cc': email.get('cc_emails', email.get('cc', '')),
                    'subject': email.get('subject', ''),
                    'body': email.get('body', ''),
                    'x_from': email.get('x_from', ''),
                    'x_to': email.get('x_to', ''),
                }
                if isinstance(formatted['to'], list):
                    formatted['to'] = ','.join(formatted['to'])
                if isinstance(formatted['cc'], list):
                    formatted['cc'] = ','.join(formatted['cc'])
                train_formatted.append(formatted)

            val_formatted = []
            for email in val_emails_raw:
                formatted = {
                    'from': email.get('from_email', email.get('from', '')),
                    'to': email.get('to_emails', email.get('to', '')),
                    'cc': email.get('cc_emails', email.get('cc', '')),
                    'subject': email.get('subject', ''),
                    'body': email.get('body', ''),
                    'x_from': email.get('x_from', ''),
                    'x_to': email.get('x_to', ''),
                }
                if isinstance(formatted['to'], list):
                    formatted['to'] = ','.join(formatted['to'])
                if isinstance(formatted['cc'], list):
                    formatted['cc'] = ','.join(formatted['cc'])
                val_formatted.append(formatted)

            # Create simple datasets
            train_dataset = EmailListDataset(train_formatted, train_labels.tolist())
            val_dataset = EmailListDataset(val_formatted, val_labels.tolist())

            def email_collate(batch):
                emails = [item[0] for item in batch]
                labels = torch.tensor([item[1] for item in batch])
                return emails, labels

            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=email_collate)
            val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=email_collate)

            ensemble.train_stacking_classifier(train_loader, val_loader, epochs=args.stacking_epochs)

            # Save meta-classifier
            meta_path = args.checkpoint / 'meta_classifier.pt'
            ensemble.save_meta_classifier(meta_path)

    # Evaluate individual models
    print("\nEvaluating individual models...")
    individual_results = evaluate_individual_models(ensemble, formatted_emails, labels)

    # Evaluate ensemble methods
    print("\nEvaluating ensemble methods...")
    include_stacking = args.train_stacking or (args.checkpoint / 'meta_classifier.pt').exists()

    # Load meta-classifier if it exists
    meta_path = args.checkpoint / 'meta_classifier.pt'
    if meta_path.exists() and not args.train_stacking:
        ensemble.load_meta_classifier(meta_path)
        include_stacking = True

    ensemble_results = evaluate_ensemble_methods(
        ensemble, formatted_emails, labels,
        include_stacking=include_stacking
    )

    # Analyze agreement
    print("\nAnalyzing agreement...")
    agreement_analysis = analyze_agreement(ensemble, formatted_emails, labels)

    # Print results
    print_results(individual_results, ensemble_results, agreement_analysis)

    # Save results if requested
    if args.output:
        results = {
            'individual_models': individual_results,
            'ensemble_methods': ensemble_results,
            'agreement_analysis': agreement_analysis,
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
