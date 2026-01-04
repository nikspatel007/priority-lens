#!/usr/bin/env python3
"""GRPO (Group Relative Policy Optimization) Training for Email Policy.

Implements the DeepSeek GRPO algorithm for training email prioritization policies.
This is Stage 3 of the RL training pipeline, targeting 80-85% accuracy.

GRPO Key Insight:
- Sample K actions per state (email)
- Compute rewards for each action using reward model
- Use group mean as baseline (no value network needed!)
- Advantage = reward - group_mean_reward
- Update policy using PPO-style clipped objective

Usage:
    python src/grpo_training.py --data data/train.json --epochs 10
    python src/grpo_training.py --data data/train.json --reward-model checkpoints/reward.pt
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from .features.combined import CombinedFeatureExtractor
    from .policy_network import (
        EmailPolicyNetwork,
        PolicyConfig,
        create_policy_network,
        NUM_ACTION_TYPES,
        NUM_RESPONSE_TIMES,
    )
    from .reward_model import (
        EmailRewardModel,
        RewardConfig,
        create_reward_model,
        ACTION_TO_PRIORITY,
    )
except ImportError:
    from features.combined import CombinedFeatureExtractor
    from policy_network import (
        EmailPolicyNetwork,
        PolicyConfig,
        create_policy_network,
        NUM_ACTION_TYPES,
        NUM_RESPONSE_TIMES,
    )
    from reward_model import (
        EmailRewardModel,
        RewardConfig,
        create_reward_model,
        ACTION_TO_PRIORITY,
    )


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # Sampling
    num_samples: int = 8  # K samples per state (email)
    temperature: float = 1.0  # Sampling temperature

    # PPO-style parameters
    clip_epsilon: float = 0.2  # PPO clipping parameter
    value_clip: float = 0.2  # Value function clipping (unused in pure GRPO)
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    kl_coef: float = 0.1  # KL penalty coefficient
    max_kl: float = 0.02  # Maximum KL divergence before early stopping

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 0.5
    batch_size: int = 64
    epochs: int = 10
    warmup_steps: int = 100

    # Training dynamics
    use_advantage_normalization: bool = True
    gamma: float = 0.99  # Not used in bandit setting, but kept for compatibility
    gae_lambda: float = 0.95  # Not used in bandit setting

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1  # Save every N epochs
    log_every: int = 10  # Log every N batches

    # Device
    device: str = "auto"


class GRPOBatch(NamedTuple):
    """A batch of samples for GRPO training."""
    features: torch.Tensor  # (batch, feature_dim)
    actions: torch.Tensor  # (batch, num_samples)
    timings: torch.Tensor  # (batch, num_samples)
    rewards: torch.Tensor  # (batch, num_samples)
    old_log_probs: torch.Tensor  # (batch, num_samples)
    advantages: torch.Tensor  # (batch, num_samples)


class EmailDataset(Dataset):
    """Dataset of emails with ground truth actions."""

    def __init__(
        self,
        emails: list[dict],
        extractor: Optional[CombinedFeatureExtractor] = None,
    ):
        self.emails = emails
        self.extractor = extractor or CombinedFeatureExtractor()

        # Pre-extract features for efficiency
        print(f"Extracting features from {len(emails)} emails...")
        self.features = []
        self.actions = []

        for i, email in enumerate(emails):
            feature_vec = self.extractor.to_vector(email)
            self.features.append(torch.tensor(feature_vec, dtype=torch.float32))

            # Get ground truth action
            action = email.get('action', 'KEPT')
            self.actions.append(action)

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{len(emails)} emails")

        print(f"Feature extraction complete")

    def __len__(self) -> int:
        return len(self.emails)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        return self.features[idx], self.actions[idx]


def get_device(device_str: str) -> torch.device:
    """Get the appropriate torch device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def compute_action_reward(
    reward_model: EmailRewardModel,
    features: torch.Tensor,
    action_idx: torch.Tensor,
    timing_idx: torch.Tensor,
    ground_truth_action: Optional[list[str]] = None,
) -> torch.Tensor:
    """Compute reward for actions using reward model and ground truth.

    The reward combines:
    1. Reward model's learned preference score
    2. Bonus for matching ground truth action (if available)

    Args:
        reward_model: Trained reward model
        features: Email features (batch, feature_dim)
        action_idx: Predicted action indices (batch,) or (batch, num_samples)
        timing_idx: Predicted timing indices
        ground_truth_action: Optional list of ground truth action labels

    Returns:
        Reward tensor of same shape as action_idx
    """
    batch_size = features.shape[0]
    device = features.device

    # Get base reward from reward model
    with torch.no_grad():
        base_reward = reward_model.get_reward(features)  # (batch,)

    # Handle multi-sample case
    is_multi_sample = action_idx.dim() == 2
    if is_multi_sample:
        num_samples = action_idx.shape[1]
        reward = base_reward.unsqueeze(1).expand(-1, num_samples).clone()
    else:
        num_samples = 1
        reward = base_reward.clone()

    # Add bonus for correct action prediction
    if ground_truth_action is not None:
        # Map ground truth to action indices
        gt_action_map = {
            'REPLIED': 0,
            'FORWARDED': 2,
            'DELETED': 4,
            'ARCHIVED': 3,
            'AUTO_FILED': 3,
            'KEPT': 3,
            'COMPOSED': 0,
            'JUNK': 4,
        }

        for i, gt in enumerate(ground_truth_action):
            gt_idx = gt_action_map.get(gt, 3)  # Default to archive
            # Bonus for matching ground truth
            if is_multi_sample:
                match_bonus = (action_idx[i] == gt_idx).float() * 1.0
                reward[i] = reward[i] + match_bonus
            else:
                match_bonus = float(action_idx[i].item() == gt_idx) * 1.0
                reward[i] = reward[i] + match_bonus

    return reward


class GRPOTrainer:
    """GRPO Trainer for email policy optimization.

    Implements the DeepSeek GRPO algorithm:
    1. Sample K actions per email from current policy
    2. Compute rewards using reward model
    3. Compute group-normalized advantages (reward - group_mean)
    4. Update policy using clipped importance sampling
    """

    def __init__(
        self,
        policy: EmailPolicyNetwork,
        reward_model: EmailRewardModel,
        config: Optional[GRPOConfig] = None,
    ):
        self.policy = policy
        self.reward_model = reward_model
        self.config = config or GRPOConfig()

        self.device = get_device(self.config.device)
        self.policy.to(self.device)
        self.reward_model.to(self.device)
        self.reward_model.eval()  # Freeze reward model

        # Keep reference policy for KL computation
        self.ref_policy = create_policy_network()
        self.ref_policy.load_state_dict(policy.state_dict())
        self.ref_policy.to(self.device)
        self.ref_policy.eval()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler with warmup
        self.scheduler = None

        # Metrics tracking
        self.metrics_history = []

    def sample_actions(
        self,
        features: torch.Tensor,
        num_samples: int,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample multiple actions per state from the policy.

        Args:
            features: Email features (batch, feature_dim)
            num_samples: Number of actions to sample per state
            temperature: Sampling temperature

        Returns:
            Tuple of:
            - action_indices: (batch, num_samples)
            - timing_indices: (batch, num_samples)
            - log_probs: (batch, num_samples) - sum of action and timing log probs
        """
        batch_size = features.shape[0]

        # Get policy output
        output = self.policy(features)

        # Apply temperature
        action_logits = output.action_logits / temperature
        timing_logits = output.timing_logits / temperature

        # Sample actions
        action_dist = torch.distributions.Categorical(logits=action_logits)
        timing_dist = torch.distributions.Categorical(logits=timing_logits)

        all_actions = []
        all_timings = []
        all_log_probs = []

        for _ in range(num_samples):
            actions = action_dist.sample()
            timings = timing_dist.sample()

            action_log_prob = action_dist.log_prob(actions)
            timing_log_prob = timing_dist.log_prob(timings)

            all_actions.append(actions)
            all_timings.append(timings)
            all_log_probs.append(action_log_prob + timing_log_prob)

        return (
            torch.stack(all_actions, dim=1),
            torch.stack(all_timings, dim=1),
            torch.stack(all_log_probs, dim=1),
        )

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Compute group-relative advantages.

        This is the core GRPO insight: use the group mean as baseline.
        Advantage_i = reward_i - mean(rewards in group)

        Args:
            rewards: Reward tensor (batch, num_samples)
            normalize: Whether to normalize advantages

        Returns:
            Advantage tensor (batch, num_samples)
        """
        # Compute group mean for each state
        group_mean = rewards.mean(dim=1, keepdim=True)

        # Advantage = reward - group_mean
        advantages = rewards - group_mean

        if normalize:
            # Normalize across all samples in batch
            adv_mean = advantages.mean()
            adv_std = advantages.std().clamp(min=1e-8)
            advantages = (advantages - adv_mean) / adv_std

        return advantages

    def compute_policy_loss(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        timings: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute GRPO policy loss with clipping.

        Args:
            features: Email features (batch, feature_dim)
            actions: Action indices (batch, num_samples)
            timings: Timing indices (batch, num_samples)
            old_log_probs: Log probs from sampling (batch, num_samples)
            advantages: Computed advantages (batch, num_samples)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch_size, num_samples = actions.shape

        # Get current policy log probs
        output = self.policy(features)
        action_dist = torch.distributions.Categorical(logits=output.action_logits)
        timing_dist = torch.distributions.Categorical(logits=output.timing_logits)

        # Compute log probs for all sampled actions
        total_loss = torch.tensor(0.0, device=self.device)
        total_entropy = torch.tensor(0.0, device=self.device)
        total_kl = torch.tensor(0.0, device=self.device)

        for s in range(num_samples):
            action_log_prob = action_dist.log_prob(actions[:, s])
            timing_log_prob = timing_dist.log_prob(timings[:, s])
            new_log_prob = action_log_prob + timing_log_prob

            # Importance ratio
            ratio = torch.exp(new_log_prob - old_log_probs[:, s])

            # Clipped surrogate loss
            surr1 = ratio * advantages[:, s]
            surr2 = torch.clamp(
                ratio,
                1.0 - self.config.clip_epsilon,
                1.0 + self.config.clip_epsilon,
            ) * advantages[:, s]

            policy_loss = -torch.min(surr1, surr2).mean()
            total_loss = total_loss + policy_loss

        total_loss = total_loss / num_samples

        # Entropy bonus (computed once, not per sample)
        entropy = action_dist.entropy() + timing_dist.entropy()
        entropy_loss = -self.config.entropy_coef * entropy.mean()
        total_entropy = entropy.mean()

        # KL divergence against reference policy
        with torch.no_grad():
            ref_output = self.ref_policy(features)
            ref_action_dist = torch.distributions.Categorical(logits=ref_output.action_logits)
            ref_timing_dist = torch.distributions.Categorical(logits=ref_output.timing_logits)

        kl_action = torch.distributions.kl_divergence(action_dist, ref_action_dist)
        kl_timing = torch.distributions.kl_divergence(timing_dist, ref_timing_dist)
        kl = (kl_action + kl_timing).mean()
        kl_loss = self.config.kl_coef * kl
        total_kl = kl

        # Combined loss
        loss = total_loss + entropy_loss + kl_loss

        metrics = {
            'policy_loss': total_loss.item(),
            'entropy': total_entropy.item(),
            'kl': total_kl.item(),
            'entropy_loss': entropy_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': loss.item(),
        }

        return loss, metrics

    def train_step(
        self,
        features: torch.Tensor,
        ground_truth_actions: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """Perform one GRPO training step.

        Args:
            features: Email features (batch, feature_dim)
            ground_truth_actions: Optional ground truth action labels

        Returns:
            Dictionary of metrics
        """
        self.policy.train()
        features = features.to(self.device)

        # Step 1: Sample K actions from current policy
        with torch.no_grad():
            actions, timings, old_log_probs = self.sample_actions(
                features,
                self.config.num_samples,
                self.config.temperature,
            )

            # Step 2: Compute rewards for each sampled action
            rewards = compute_action_reward(
                self.reward_model,
                features,
                actions,
                timings,
                ground_truth_actions,
            )

            # Step 3: Compute group-relative advantages
            advantages = self.compute_advantages(
                rewards,
                normalize=self.config.use_advantage_normalization,
            )

        # Step 4: Compute policy loss and update
        self.optimizer.zero_grad()
        loss, metrics = self.compute_policy_loss(
            features,
            actions,
            timings,
            old_log_probs,
            advantages,
        )

        loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.max_grad_norm,
            )

        self.optimizer.step()

        # Add reward stats to metrics
        metrics['reward_mean'] = rewards.mean().item()
        metrics['reward_std'] = rewards.std().item()
        metrics['advantage_mean'] = advantages.mean().item()
        metrics['advantage_std'] = advantages.std().item()

        return metrics

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: DataLoader yielding (features, actions)

        Returns:
            Dictionary of average metrics
        """
        epoch_metrics = defaultdict(float)
        num_batches = 0

        for batch_idx, (features, actions) in enumerate(dataloader):
            metrics = self.train_step(features, list(actions))

            for k, v in metrics.items():
                epoch_metrics[k] += v
            num_batches += 1

            if (batch_idx + 1) % self.config.log_every == 0:
                avg_loss = epoch_metrics['total_loss'] / num_batches
                avg_reward = epoch_metrics['reward_mean'] / num_batches
                print(f"  Batch {batch_idx + 1}: loss={avg_loss:.4f}, reward={avg_reward:.4f}")

        # Average metrics
        return {k: v / num_batches for k, v in epoch_metrics.items()}

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """Evaluate policy on validation/test set.

        Args:
            dataloader: DataLoader yielding (features, actions)

        Returns:
            Dictionary of evaluation metrics
        """
        self.policy.eval()

        total_correct = 0
        total_samples = 0
        total_reward = 0.0

        # Action mapping
        gt_action_map = {
            'REPLIED': 0,
            'FORWARDED': 2,
            'DELETED': 4,
            'ARCHIVED': 3,
            'AUTO_FILED': 3,
            'KEPT': 3,
            'COMPOSED': 0,
            'JUNK': 4,
        }

        with torch.no_grad():
            for features, actions in dataloader:
                features = features.to(self.device)
                batch_size = features.shape[0]

                # Get greedy predictions
                output = self.policy(features)
                pred_actions = output.action_logits.argmax(dim=-1)

                # Get rewards
                rewards = self.reward_model.get_reward(features)
                total_reward += rewards.sum().item()

                # Compute accuracy
                for i, gt in enumerate(actions):
                    gt_idx = gt_action_map.get(gt, 3)
                    if pred_actions[i].item() == gt_idx:
                        total_correct += 1
                    total_samples += 1

        return {
            'accuracy': total_correct / total_samples if total_samples > 0 else 0.0,
            'avg_reward': total_reward / total_samples if total_samples > 0 else 0.0,
            'total_samples': total_samples,
        }

    def train(
        self,
        train_dataset: EmailDataset,
        val_dataset: Optional[EmailDataset] = None,
        num_epochs: Optional[int] = None,
    ) -> list[dict]:
        """Full training loop.

        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            num_epochs: Override config epochs

        Returns:
            List of per-epoch metrics
        """
        epochs = num_epochs or self.config.epochs
        history = []

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
            )

        print(f"\nStarting GRPO training for {epochs} epochs")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Samples per state: {self.config.num_samples}")
        print(f"  Device: {self.device}")
        print()

        best_val_acc = 0.0
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader)
            epoch_time = time.time() - start_time

            metrics = {
                'epoch': epoch + 1,
                'time': epoch_time,
                **{f'train_{k}': v for k, v in train_metrics.items()},
            }

            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                metrics.update({f'val_{k}': v for k, v in val_metrics.items()})

                # Save best model
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    self.save_checkpoint(checkpoint_dir / 'best_grpo.pt')

            history.append(metrics)

            # Print epoch summary
            msg = f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s)"
            msg += f" | Loss: {train_metrics['total_loss']:.4f}"
            msg += f" | Reward: {train_metrics['reward_mean']:.4f}"
            if val_loader is not None:
                msg += f" | Val Acc: {val_metrics['accuracy']:.4f}"
            print(msg)

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(checkpoint_dir / f'grpo_epoch_{epoch + 1}.pt')

            # Early stopping on high KL
            if train_metrics.get('kl', 0) > self.config.max_kl:
                print(f"  Early stopping: KL divergence {train_metrics['kl']:.4f} > {self.config.max_kl}")
                break

        print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
        return history

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, path)
        print(f"  Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {path}")


def create_synthetic_dataset(
    num_samples: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """Create synthetic email dataset for testing.

    Args:
        num_samples: Number of emails to generate
        seed: Random seed

    Returns:
        List of email dictionaries
    """
    random.seed(seed)

    actions = ['REPLIED', 'FORWARDED', 'ARCHIVED', 'DELETED', 'KEPT']
    action_weights = [0.25, 0.03, 0.45, 0.09, 0.18]  # Roughly matching Enron

    subjects = [
        "URGENT: Need your approval",
        "FYI: Weekly report",
        "RE: Project status update",
        "Meeting tomorrow at 3pm",
        "Question about the budget",
        "Action required: Review document",
        "Quick question",
        "Following up on our discussion",
        "Please review and respond",
        "FW: Customer inquiry",
    ]

    emails = []
    for i in range(num_samples):
        action = random.choices(actions, weights=action_weights)[0]
        subject = random.choice(subjects)

        email = {
            'from': f'user{random.randint(1, 100)}@example.com',
            'to': 'you@example.com',
            'subject': subject,
            'body': f"Email body content {i}. " * random.randint(1, 5),
            'action': action,
        }
        emails.append(email)

    return emails


def load_emails(data_path: Path, limit: Optional[int] = None) -> list[dict]:
    """Load emails from JSON file."""
    print(f"Loading emails from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        emails = json.load(f)

    if limit:
        emails = emails[:limit]

    print(f"Loaded {len(emails)} emails")
    return emails


def main():
    parser = argparse.ArgumentParser(description='GRPO Training for Email Policy')
    parser.add_argument(
        '--data',
        type=Path,
        help='Path to training data JSON',
    )
    parser.add_argument(
        '--val-data',
        type=Path,
        help='Path to validation data JSON',
    )
    parser.add_argument(
        '--reward-model',
        type=Path,
        help='Path to trained reward model checkpoint',
    )
    parser.add_argument(
        '--policy-checkpoint',
        type=Path,
        help='Path to initial policy checkpoint (from SFT)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size',
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=8,
        help='Number of action samples per state (K)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate',
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data for testing',
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of training samples',
    )

    args = parser.parse_args()

    # Load or create data
    if args.synthetic or args.data is None:
        print("Using synthetic dataset for testing...")
        train_emails = create_synthetic_dataset(num_samples=args.limit or 1000)
        val_emails = create_synthetic_dataset(num_samples=200, seed=123)
    else:
        train_emails = load_emails(args.data, args.limit)
        val_emails = load_emails(args.val_data) if args.val_data else None

    # Create datasets
    train_dataset = EmailDataset(train_emails)
    val_dataset = EmailDataset(val_emails) if val_emails else None

    # Create policy network
    policy = create_policy_network()
    if args.policy_checkpoint and args.policy_checkpoint.exists():
        print(f"Loading policy from {args.policy_checkpoint}")
        checkpoint = torch.load(args.policy_checkpoint, map_location='cpu', weights_only=False)
        policy.load_state_dict(checkpoint['model_state_dict'])

    # Create reward model
    reward_model = create_reward_model()
    if args.reward_model and args.reward_model.exists():
        print(f"Loading reward model from {args.reward_model}")
        checkpoint = torch.load(args.reward_model, map_location='cpu', weights_only=False)
        reward_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Using untrained reward model (will use ground truth signals)")

    # Configure GRPO
    config = GRPOConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        learning_rate=args.lr,
    )

    # Create trainer and train
    trainer = GRPOTrainer(policy, reward_model, config)

    print("\n" + "=" * 60)
    print("GRPO TRAINING")
    print("=" * 60)

    history = trainer.train(train_dataset, val_dataset)

    # Final evaluation
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        final_metrics = trainer.evaluate(val_loader)
        print(f"\nFinal validation accuracy: {final_metrics['accuracy']:.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
