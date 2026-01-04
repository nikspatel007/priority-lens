#!/usr/bin/env python3
"""Supervised Fine-Tuning (SFT) for email policy network.

Stage 1 of the training pipeline: Learn from labeled actions in Gmail data.

Usage:
    python src/train_sft.py --train data/gmail_splits/train.json --val data/gmail_splits/val.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from features.combined import extract_combined_features, CombinedFeatureExtractor
from policy_network import EmailPolicyNetwork, PolicyConfig


# Action mapping (matches evaluate.py)
ACTION_NAMES = ['reply_now', 'reply_later', 'forward', 'archive', 'delete']
LABEL_TO_ACTION = {
    'REPLY_NOW': 0,
    'REPLY_LATER': 1,
    'FORWARD': 2,
    'ARCHIVE': 3,
    'DELETE': 4,
}


def build_user_index(emails: list[dict]) -> tuple[dict[str, int], int]:
    """Build a mapping from email addresses to user indices.

    Args:
        emails: List of email dictionaries with 'from' field

    Returns:
        Tuple of (email_to_idx dict, num_users)
        Index 0 is reserved for unknown users.
    """
    # Collect unique email addresses from the 'from' field
    unique_emails = set()
    for email in emails:
        sender = email.get('from', '').lower().strip()
        if sender:
            unique_emails.add(sender)

    # Create mapping (0 = unknown, 1..N = known users)
    email_to_idx = {email: idx + 1 for idx, email in enumerate(sorted(unique_emails))}
    return email_to_idx, len(unique_emails)


class EmailDataset(Dataset):
    """PyTorch dataset for labeled emails with optional user tracking."""

    def __init__(
        self,
        emails: list[dict],
        feature_extractor: CombinedFeatureExtractor,
        user_index: Optional[dict[str, int]] = None,
    ):
        """Initialize dataset.

        Args:
            emails: List of email dictionaries
            feature_extractor: Feature extractor instance
            user_index: Optional mapping from email addresses to user indices.
                        If provided, enables user-specific model training.
        """
        self.emails = emails
        self.feature_extractor = feature_extractor
        self.user_index = user_index or {}
        self.features = []
        self.labels = []
        self.user_ids = []  # User indices for user-specific models

        print(f"Extracting features from {len(emails)} emails...")
        for email in emails:
            try:
                # Extract features - pass email dict directly
                combined = extract_combined_features(email)
                feat_vec = combined.to_feature_vector()
                if isinstance(feat_vec, np.ndarray):
                    feat_vec = feat_vec.tolist()

                # Get label
                action_label = email.get('action', 'ARCHIVE')
                action_idx = LABEL_TO_ACTION.get(action_label, 3)

                # Get user ID (0 = unknown if not in index)
                sender = email.get('from', '').lower().strip()
                user_id = self.user_index.get(sender, 0)

                self.features.append(feat_vec)
                self.labels.append(action_idx)
                self.user_ids.append(user_id)
            except Exception as e:
                print(f"Warning: Failed to extract features: {e}")
                continue

        print(f"Extracted features for {len(self.features)} emails")
        if self.user_index:
            unique_users = len(set(self.user_ids) - {0})
            print(f"  Unique users in dataset: {unique_users}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.user_ids[idx], dtype=torch.long),
        )

    @property
    def has_user_ids(self) -> bool:
        """Check if this dataset has user ID information."""
        return bool(self.user_index)


def compute_class_weights(
    dataset: EmailDataset,
    weight_power: float = 0.3,
) -> torch.Tensor:
    """Compute class weights based on frequency distribution.

    Uses a softened inverse frequency approach:
        weight_i = (total / (n_classes * count_i)) ^ power

    Args:
        dataset: The email dataset
        weight_power: Controls balance between uniform and inverse frequency.
            - 0.0: Uniform weights (no class balancing, majority class dominates)
            - 0.3: Soft balancing (default, respects true distribution)
            - 0.5: Square root inverse frequency (moderate balancing)
            - 1.0: Full inverse frequency (heavy bias to rare classes)

    Returns:
        Tensor of class weights
    """
    counts = Counter(dataset.labels)
    total = len(dataset.labels)
    weights = []
    for i in range(len(ACTION_NAMES)):
        count = counts.get(i, 1)
        # Softened inverse frequency
        raw_weight = total / (len(ACTION_NAMES) * count)
        weights.append(raw_weight ** weight_power)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(
    model: EmailPolicyNetwork,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_user_ids: bool = False,
) -> dict:
    """Train for one epoch.

    Args:
        model: The policy network
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        use_user_ids: Whether to use user IDs for user-specific training

    Returns:
        Dict with 'loss' and 'accuracy' metrics
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        if use_user_ids:
            features, labels, user_ids = batch
            user_ids = user_ids.to(device)
        else:
            features, labels, _ = batch  # Ignore user_ids
            user_ids = None
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass (with optional user conditioning)
        output = model(features, user_ids)
        action_logits = output.action_logits

        # Compute loss (cross-entropy on action prediction)
        loss = criterion(action_logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * features.size(0)
        preds = action_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += features.size(0)

    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
    }


def evaluate(
    model: EmailPolicyNetwork,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_user_ids: bool = False,
) -> dict:
    """Evaluate model on a dataset.

    Args:
        model: The policy network
        dataloader: Evaluation data loader
        criterion: Loss function
        device: Device to use
        use_user_ids: Whether to use user IDs for user-specific evaluation

    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if use_user_ids:
                features, labels, user_ids = batch
                user_ids = user_ids.to(device)
            else:
                features, labels, _ = batch
                user_ids = None
            features = features.to(device)
            labels = labels.to(device)

            output = model(features, user_ids)
            action_logits = output.action_logits

            loss = criterion(action_logits, labels)
            total_loss += loss.item() * features.size(0)

            preds = action_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += features.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Per-class accuracy
    class_correct = Counter()
    class_total = Counter()
    for p, l in zip(all_preds, all_labels):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1

    per_class_acc = {}
    for i, name in enumerate(ACTION_NAMES):
        if class_total[i] > 0:
            per_class_acc[name] = class_correct[i] / class_total[i]
        else:
            per_class_acc[name] = 0.0

    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'per_class_accuracy': per_class_acc,
    }


def main():
    parser = argparse.ArgumentParser(description='SFT training for email policy network')
    parser.add_argument('--train', type=Path, required=True, help='Training data JSON')
    parser.add_argument('--val', type=Path, help='Validation data JSON')
    parser.add_argument('--output', type=Path, default=Path('checkpoints/sft_gmail.pt'),
                       help='Output checkpoint path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden-dims', type=str, default='256,128,64',
                       help='Hidden layer dimensions (comma-separated)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda, mps)')
    parser.add_argument('--weight-power', type=float, default=0.3,
                       help='Class weight power (0=uniform, 0.3=soft, 1.0=inverse freq)')
    # User-specific model arguments
    parser.add_argument('--user-specific', action='store_true',
                       help='Enable user-specific model with per-user embeddings')
    parser.add_argument('--user-embedding-dim', type=int, default=32,
                       help='Dimension of user embeddings (default: 32)')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load data
    print(f"Loading training data from {args.train}...")
    with open(args.train) as f:
        train_emails = json.load(f)
    print(f"Loaded {len(train_emails)} training emails")

    val_emails = None
    if args.val and args.val.exists():
        print(f"Loading validation data from {args.val}...")
        with open(args.val) as f:
            val_emails = json.load(f)
        print(f"Loaded {len(val_emails)} validation emails")

    # Build user index for user-specific training
    user_index = None
    num_users = 0
    if args.user_specific:
        print("\nBuilding user index for user-specific training...")
        user_index, num_users = build_user_index(train_emails)
        print(f"  Found {num_users} unique users")

    # Create datasets
    feature_extractor = CombinedFeatureExtractor()
    train_dataset = EmailDataset(train_emails, feature_extractor, user_index)

    val_dataset = None
    if val_emails:
        val_dataset = EmailDataset(val_emails, feature_extractor, user_index)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Print class distribution
    print("\nTraining class distribution:")
    train_counts = Counter(train_dataset.labels)
    for i, name in enumerate(ACTION_NAMES):
        count = train_counts.get(i, 0)
        pct = 100 * count / len(train_dataset) if train_dataset else 0
        print(f"  {name}: {count} ({pct:.1f}%)")

    # Create model
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(','))
    config = PolicyConfig(
        input_dim=69,  # CombinedFeatures dimension (with content features)
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        num_users=num_users if args.user_specific else 0,
        user_embedding_dim=args.user_embedding_dim,
    )
    model = EmailPolicyNetwork(config).to(device)
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    if args.user_specific:
        print(f"  User-specific: {num_users} users, {args.user_embedding_dim}d embeddings")

    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(train_dataset, args.weight_power).to(device)
    print(f"Class weights (power={args.weight_power}): {class_weights.tolist()}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    use_user_ids = args.user_specific
    best_val_acc = 0.0
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)

    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_user_ids=use_user_ids
        )

        log = f"Epoch {epoch+1:3d}: train_loss={train_metrics['loss']:.4f}, train_acc={train_metrics['accuracy']:.3f}"

        if val_loader:
            val_metrics = evaluate(
                model, val_loader, criterion, device,
                use_user_ids=use_user_ids
            )
            log += f", val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.3f}"
            scheduler.step(val_metrics['loss'])

            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                args.output.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'val_accuracy': best_val_acc,
                    'user_index': user_index,  # Save for inference
                }, args.output)
                log += " *"
        else:
            scheduler.step(train_metrics['loss'])
            # Save periodically
            if (epoch + 1) % 10 == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'user_index': user_index,
                }, args.output)

        print(log)

    print("-" * 60)
    print(f"Training complete. Best validation accuracy: {best_val_acc:.3f}")
    print(f"Checkpoint saved to: {args.output}")

    # Final evaluation on validation set
    if val_loader:
        print("\nFinal validation metrics:")
        final_metrics = evaluate(
            model, val_loader, criterion, device,
            use_user_ids=use_user_ids
        )
        for action, acc in final_metrics['per_class_accuracy'].items():
            print(f"  {action}: {acc:.3f}")


if __name__ == '__main__':
    main()
