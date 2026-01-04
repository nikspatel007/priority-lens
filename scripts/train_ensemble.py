#!/usr/bin/env python3
"""Train ensemble of SFT models with different configurations.

Creates multiple model variants to combine for better predictions:
1. SFT-balanced: Class-weighted loss (inverse frequency)
2. SFT-focal: Focal loss (down-weights easy examples)
3. SFT-oversampled: WeightedRandomSampler (balanced batches)
4. SFT-large-embed: 768-dim embeddings (all-mpnet-base-v2)

Usage:
    # Train all variants using SurrealDB
    python scripts/train_ensemble.py --database enron

    # Train specific variant
    python scripts/train_ensemble.py --database enron --variant balanced

    # Train with JSON data
    python scripts/train_ensemble.py --data data/train.json --val-data data/val.json
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.sft_training import (
    SFTConfig,
    SFTTrainer,
    SFTDataset,
    load_emails,
    create_policy_network,
    collate_fn,
    IDX_TO_ACTION,
    NUM_ACTION_TYPES,
)
from src.features.combined import FEATURE_DIMS, CombinedFeatureExtractor


# Large embedding model (768-dim)
LARGE_EMBED_MODEL = 'all-mpnet-base-v2'
LARGE_EMBED_DIM = 768


@dataclass
class EnsembleVariant:
    """Configuration for an ensemble member."""
    name: str
    use_class_weights: bool = True
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    use_balanced_sampling: bool = False
    include_content: bool = False
    content_model: str = 'all-MiniLM-L6-v2'
    description: str = ''


# Define ensemble variants
ENSEMBLE_VARIANTS = {
    'balanced': EnsembleVariant(
        name='balanced',
        use_class_weights=True,
        use_focal_loss=False,
        use_balanced_sampling=False,
        include_content=True,
        description='Class-weighted cross-entropy loss (baseline)',
    ),
    'focal': EnsembleVariant(
        name='focal',
        use_class_weights=True,
        use_focal_loss=True,
        focal_gamma=2.0,
        use_balanced_sampling=False,
        include_content=True,
        description='Focal loss for hard example mining',
    ),
    'oversampled': EnsembleVariant(
        name='oversampled',
        use_class_weights=False,
        use_focal_loss=False,
        use_balanced_sampling=True,
        include_content=True,
        description='Balanced sampling via WeightedRandomSampler',
    ),
    'large_embed': EnsembleVariant(
        name='large_embed',
        use_class_weights=True,
        use_focal_loss=False,
        use_balanced_sampling=False,
        include_content=True,
        content_model=LARGE_EMBED_MODEL,
        description='768-dim embeddings with all-mpnet-base-v2',
    ),
}


def get_feature_dim(variant: EnsembleVariant) -> int:
    """Get input dimension for a variant."""
    base_dim = FEATURE_DIMS['total_base']
    if variant.include_content:
        if variant.content_model == LARGE_EMBED_MODEL:
            return base_dim + LARGE_EMBED_DIM
        else:
            return FEATURE_DIMS['total_with_content']
    return base_dim


def create_extractor(variant: EnsembleVariant) -> CombinedFeatureExtractor:
    """Create feature extractor for a variant."""
    return CombinedFeatureExtractor(
        include_content=variant.include_content,
        content_model=variant.content_model,
    )


def train_variant(
    variant: EnsembleVariant,
    train_emails: list[dict],
    val_emails: Optional[list[dict]],
    config: SFTConfig,
    output_dir: Path,
) -> dict:
    """Train a single ensemble variant.

    Args:
        variant: Ensemble variant configuration
        train_emails: Training email data
        val_emails: Validation email data (optional)
        config: Base SFT training config
        output_dir: Directory to save checkpoints

    Returns:
        Dictionary with training results including final metrics
    """
    print(f"\n{'='*60}")
    print(f"TRAINING VARIANT: {variant.name.upper()}")
    print(f"{'='*60}")
    print(f"Description: {variant.description}")
    print()

    # Create feature extractor for this variant
    extractor = create_extractor(variant)
    input_dim = get_feature_dim(variant)

    print(f"Feature configuration:")
    print(f"  Content embeddings: {variant.include_content}")
    print(f"  Content model: {variant.content_model}")
    print(f"  Input dimension: {input_dim}")
    print()

    # Create datasets
    train_dataset = SFTDataset(train_emails, extractor)
    val_dataset = SFTDataset(val_emails, extractor) if val_emails else None

    # Create policy network with correct input dimension
    policy = create_policy_network(input_dim=input_dim)
    print(f"Policy network: {sum(p.numel() for p in policy.parameters()):,} parameters")

    # Create variant-specific config
    variant_config = SFTConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        device=config.device,
        checkpoint_dir=str(output_dir / variant.name),
        # Variant-specific options
        use_class_weights=variant.use_class_weights,
        use_focal_loss=variant.use_focal_loss,
        focal_gamma=variant.focal_gamma,
        use_balanced_sampling=variant.use_balanced_sampling,
    )

    # Create trainer and train
    trainer = SFTTrainer(policy, variant_config)
    history = trainer.train(train_dataset, val_dataset)

    # Get final metrics
    final_metrics = {
        'variant': variant.name,
        'input_dim': input_dim,
        'train_accuracy': history[-1]['train_accuracy'],
        'val_accuracy': history[-1].get('val_accuracy', 0),
    }

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            collate_fn=collate_fn,
        )
        eval_metrics = trainer.evaluate(val_loader)
        final_metrics['per_class_accuracy'] = {
            IDX_TO_ACTION[i]: eval_metrics.get(f'acc_{IDX_TO_ACTION[i]}', 0)
            for i in range(NUM_ACTION_TYPES)
        }

    # Save final checkpoint with metadata
    checkpoint_path = output_dir / variant.name / 'final.pt'
    trainer.save_checkpoint(checkpoint_path, input_dim=input_dim)

    # Save variant config
    config_path = output_dir / variant.name / 'config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'name': variant.name,
            'description': variant.description,
            'input_dim': input_dim,
            'use_class_weights': variant.use_class_weights,
            'use_focal_loss': variant.use_focal_loss,
            'focal_gamma': variant.focal_gamma,
            'use_balanced_sampling': variant.use_balanced_sampling,
            'include_content': variant.include_content,
            'content_model': variant.content_model,
            'final_metrics': final_metrics,
        }, f, indent=2)

    return final_metrics


def train_ensemble(
    train_emails: list[dict],
    val_emails: Optional[list[dict]],
    variants: list[str],
    config: SFTConfig,
    output_dir: Path,
) -> dict:
    """Train all ensemble variants.

    Args:
        train_emails: Training email data
        val_emails: Validation email data
        variants: List of variant names to train
        config: Base SFT training config
        output_dir: Directory to save checkpoints

    Returns:
        Dictionary with results for all variants
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for variant_name in variants:
        if variant_name not in ENSEMBLE_VARIANTS:
            print(f"Warning: Unknown variant '{variant_name}', skipping")
            continue

        variant = ENSEMBLE_VARIANTS[variant_name]
        results[variant_name] = train_variant(
            variant, train_emails, val_emails, config, output_dir
        )

    # Save ensemble summary
    summary_path = output_dir / 'ensemble_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'variants': list(results.keys()),
            'results': results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("ENSEMBLE TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nVariant accuracies:")
    for name, metrics in results.items():
        print(f"  {name}: {metrics['val_accuracy']:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train ensemble of SFT models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data source options (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        '--data',
        type=Path,
        help='Path to training data JSON',
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
        '--val-data',
        type=Path,
        help='Path to validation data JSON (for --data mode)',
    )
    parser.add_argument(
        '--variant',
        type=str,
        choices=list(ENSEMBLE_VARIANTS.keys()) + ['all'],
        default='all',
        help='Which variant(s) to train (default: all)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('checkpoints/ensemble'),
        help='Output directory for checkpoints (default: checkpoints/ensemble)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size (default: 64)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use (default: auto)',
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of training samples',
    )

    args = parser.parse_args()

    # Load data
    if args.synthetic:
        from src.sft_training import create_synthetic_dataset
        print("Using synthetic dataset for testing...")
        train_emails = create_synthetic_dataset(num_samples=args.limit or 1000)
        val_emails = create_synthetic_dataset(num_samples=200, seed=123)
    elif args.database:
        # Load from SurrealDB
        try:
            from db.dataset import SurrealEmailDataset
            print(f"Loading data from SurrealDB (database={args.database})...")

            # We need to get raw emails for feature extraction
            # Create a simple dataset to fetch emails
            train_ds = SurrealEmailDataset(
                database=args.database,
                split='train',
                use_cached_embeddings=False,
            )
            val_ds = SurrealEmailDataset(
                database=args.database,
                split='val',
                use_cached_embeddings=False,
            )

            # Convert to email dicts
            train_emails = train_ds._emails
            val_emails = val_ds._emails

            if args.limit:
                train_emails = train_emails[:args.limit]

            print(f"Loaded {len(train_emails)} train, {len(val_emails)} val emails")

        except Exception as e:
            print(f"Error loading from SurrealDB: {e}")
            print("Make sure SurrealDB is running:")
            print("  surreal start file:data/enron.db --user root --pass root")
            sys.exit(1)
    else:
        train_emails = load_emails(args.data, args.limit)
        val_emails = load_emails(args.val_data) if args.val_data else None

    # Determine variants to train
    if args.variant == 'all':
        variants = list(ENSEMBLE_VARIANTS.keys())
    else:
        variants = [args.variant]

    # Create config
    config = SFTConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )

    # Train ensemble
    train_ensemble(
        train_emails=train_emails,
        val_emails=val_emails,
        variants=variants,
        config=config,
        output_dir=args.output,
    )


if __name__ == '__main__':
    main()
