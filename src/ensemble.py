#!/usr/bin/env python3
"""Ensemble inference for SFT models.

Combines predictions from multiple model variants using various strategies:
1. Hard Voting: Majority vote across models
2. Soft Voting (Averaging): Average softmax probabilities
3. Weighted Averaging: Weight by validation accuracy
4. Stacking: Train meta-classifier on model outputs

Usage:
    from src.ensemble import EnsemblePredictor, EnsembleMethod

    # Load ensemble from checkpoint directory
    ensemble = EnsemblePredictor.from_checkpoint_dir('checkpoints/ensemble')

    # Predict with different methods
    probs = ensemble.predict_proba(features, method=EnsembleMethod.WEIGHTED_AVG)
    action = ensemble.predict(features)
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .policy_network import (
        EmailPolicyNetwork,
        PolicyConfig,
        create_policy_network,
        NUM_ACTION_TYPES,
    )
    from .features.combined import CombinedFeatureExtractor, FEATURE_DIMS
except ImportError:
    from policy_network import (
        EmailPolicyNetwork,
        PolicyConfig,
        create_policy_network,
        NUM_ACTION_TYPES,
    )
    from features.combined import CombinedFeatureExtractor, FEATURE_DIMS


class EnsembleMethod(Enum):
    """Ensemble combination methods."""
    HARD_VOTING = 'hard_voting'     # Majority vote on predictions
    SOFT_VOTING = 'soft_voting'     # Average probabilities
    WEIGHTED_AVG = 'weighted_avg'   # Weight by validation accuracy
    STACKING = 'stacking'           # Meta-classifier on outputs


@dataclass
class ModelMember:
    """A single model in the ensemble."""
    name: str
    model: EmailPolicyNetwork
    extractor: CombinedFeatureExtractor
    input_dim: int
    weight: float = 1.0
    validation_accuracy: float = 0.0
    config: dict = field(default_factory=dict)


class EnsemblePredictor:
    """Ensemble predictor combining multiple SFT models.

    Supports multiple combination strategies and handles models with
    different input dimensions (e.g., different embedding models).
    """

    def __init__(
        self,
        members: list[ModelMember],
        device: str = 'auto',
    ):
        """Initialize ensemble predictor.

        Args:
            members: List of ModelMember objects
            device: Device for inference
        """
        self.members = members
        self.device = self._get_device(device)

        # Move all models to device
        for member in self.members:
            member.model.to(self.device)
            member.model.eval()

        # Compute normalized weights based on validation accuracy
        self._compute_weights()

        # Meta-classifier for stacking (trained lazily)
        self._meta_classifier = None

    def _get_device(self, device_str: str) -> torch.device:
        """Get the appropriate torch device."""
        if device_str == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            return torch.device('cpu')
        return torch.device(device_str)

    def _compute_weights(self) -> None:
        """Compute normalized weights from validation accuracies."""
        total_acc = sum(m.validation_accuracy for m in self.members)
        if total_acc > 0:
            for member in self.members:
                member.weight = member.validation_accuracy / total_acc
        else:
            # Equal weights if no accuracy data
            for member in self.members:
                member.weight = 1.0 / len(self.members)

    def _extract_features(
        self,
        email: dict,
        member: ModelMember,
    ) -> torch.Tensor:
        """Extract features for a single model member."""
        feature_vec = member.extractor.to_vector(email)
        return torch.tensor(feature_vec, dtype=torch.float32, device=self.device)

    def _extract_features_batch(
        self,
        emails: list[dict],
        member: ModelMember,
    ) -> torch.Tensor:
        """Extract features for a batch of emails."""
        features_list = [
            member.extractor.to_vector(email) for email in emails
        ]
        return torch.tensor(features_list, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _get_member_probs(
        self,
        features: torch.Tensor,
        member: ModelMember,
    ) -> torch.Tensor:
        """Get softmax probabilities from a single member."""
        output = member.model(features)
        return F.softmax(output.action_logits, dim=-1)

    @torch.no_grad()
    def _get_member_prediction(
        self,
        features: torch.Tensor,
        member: ModelMember,
    ) -> torch.Tensor:
        """Get hard prediction from a single member."""
        output = member.model(features)
        return output.action_logits.argmax(dim=-1)

    def predict_proba_from_features(
        self,
        features_dict: dict[str, torch.Tensor],
        method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVG,
    ) -> torch.Tensor:
        """Get ensemble probability predictions from pre-extracted features.

        Args:
            features_dict: Dictionary mapping member names to feature tensors
            method: Ensemble combination method

        Returns:
            Tensor of shape (batch, num_classes) with probabilities
        """
        if method == EnsembleMethod.STACKING and self._meta_classifier is not None:
            return self._stacking_predict(features_dict)

        batch_size = next(iter(features_dict.values())).size(0)

        if method == EnsembleMethod.HARD_VOTING:
            # Collect votes from each model
            votes = torch.zeros(batch_size, NUM_ACTION_TYPES, device=self.device)
            for member in self.members:
                features = features_dict[member.name]
                preds = self._get_member_prediction(features, member)
                votes.scatter_add_(1, preds.unsqueeze(1), torch.ones(batch_size, 1, device=self.device))
            # Normalize to probabilities
            return votes / len(self.members)

        elif method == EnsembleMethod.SOFT_VOTING:
            # Average probabilities
            probs_sum = torch.zeros(batch_size, NUM_ACTION_TYPES, device=self.device)
            for member in self.members:
                features = features_dict[member.name]
                probs = self._get_member_probs(features, member)
                probs_sum += probs
            return probs_sum / len(self.members)

        elif method == EnsembleMethod.WEIGHTED_AVG:
            # Weighted average of probabilities
            probs_sum = torch.zeros(batch_size, NUM_ACTION_TYPES, device=self.device)
            for member in self.members:
                features = features_dict[member.name]
                probs = self._get_member_probs(features, member)
                probs_sum += member.weight * probs
            return probs_sum

        else:
            raise ValueError(f"Unknown ensemble method: {method}")

    def predict_proba(
        self,
        emails: Union[dict, list[dict]],
        method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVG,
    ) -> torch.Tensor:
        """Get ensemble probability predictions.

        Args:
            emails: Single email dict or list of email dicts
            method: Ensemble combination method

        Returns:
            Tensor of shape (batch, num_classes) with probabilities
        """
        if isinstance(emails, dict):
            emails = [emails]

        # Extract features for each member
        features_dict = {}
        for member in self.members:
            features_dict[member.name] = self._extract_features_batch(emails, member)

        return self.predict_proba_from_features(features_dict, method)

    def predict(
        self,
        emails: Union[dict, list[dict]],
        method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVG,
    ) -> torch.Tensor:
        """Get ensemble action predictions.

        Args:
            emails: Single email dict or list of email dicts
            method: Ensemble combination method

        Returns:
            Tensor of shape (batch,) with action indices
        """
        probs = self.predict_proba(emails, method)
        return probs.argmax(dim=-1)

    def train_stacking_classifier(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        lr: float = 1e-3,
    ) -> float:
        """Train a meta-classifier for stacking ensemble.

        The meta-classifier takes the concatenated probability outputs
        from all base models and learns optimal weights.

        Args:
            train_loader: DataLoader yielding (email_dict, label) pairs
            val_loader: Optional validation DataLoader
            epochs: Training epochs
            lr: Learning rate

        Returns:
            Final validation accuracy
        """
        # Meta-classifier: simple MLP on concatenated probabilities
        input_dim = len(self.members) * NUM_ACTION_TYPES
        self._meta_classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, NUM_ACTION_TYPES),
        ).to(self.device)

        optimizer = torch.optim.AdamW(self._meta_classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        print(f"Training stacking meta-classifier...")
        print(f"  Input dim: {input_dim}")
        print(f"  Epochs: {epochs}")

        for epoch in range(epochs):
            self._meta_classifier.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            for batch in train_loader:
                if isinstance(batch, tuple) and len(batch) == 2:
                    emails, labels = batch
                else:
                    raise ValueError("DataLoader must yield (emails, labels) tuples")

                # Get base model probabilities
                all_probs = []
                for member in self.members:
                    features = self._extract_features_batch(emails, member)
                    probs = self._get_member_probs(features, member)
                    all_probs.append(probs)

                # Concatenate probabilities
                meta_input = torch.cat(all_probs, dim=-1)
                labels = labels.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                logits = self._meta_classifier(meta_input)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(labels)
                total_correct += (logits.argmax(dim=-1) == labels).sum().item()
                total_samples += len(labels)

            train_acc = total_correct / total_samples
            print(f"  Epoch {epoch + 1}/{epochs}: loss={total_loss/total_samples:.4f}, acc={train_acc:.1%}")

        # Evaluate on validation set
        if val_loader is not None:
            val_acc = self._evaluate_stacking(val_loader)
            print(f"  Validation accuracy: {val_acc:.1%}")
            return val_acc

        return train_acc

    @torch.no_grad()
    def _evaluate_stacking(self, dataloader: DataLoader) -> float:
        """Evaluate stacking meta-classifier."""
        self._meta_classifier.eval()
        total_correct = 0
        total_samples = 0

        for emails, labels in dataloader:
            all_probs = []
            for member in self.members:
                features = self._extract_features_batch(emails, member)
                probs = self._get_member_probs(features, member)
                all_probs.append(probs)

            meta_input = torch.cat(all_probs, dim=-1)
            labels = labels.to(self.device)

            logits = self._meta_classifier(meta_input)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_samples += len(labels)

        return total_correct / total_samples

    def _stacking_predict(self, features_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get predictions using stacking meta-classifier."""
        all_probs = []
        for member in self.members:
            features = features_dict[member.name]
            probs = self._get_member_probs(features, member)
            all_probs.append(probs)

        meta_input = torch.cat(all_probs, dim=-1)
        logits = self._meta_classifier(meta_input)
        return F.softmax(logits, dim=-1)

    def evaluate(
        self,
        dataloader: DataLoader,
        method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVG,
    ) -> dict[str, float]:
        """Evaluate ensemble on a dataset.

        Args:
            dataloader: DataLoader yielding (features, labels) or (emails, labels)
            method: Ensemble combination method

        Returns:
            Dictionary with accuracy metrics
        """
        total_correct = 0
        total_samples = 0
        class_correct = {i: 0 for i in range(NUM_ACTION_TYPES)}
        class_total = {i: 0 for i in range(NUM_ACTION_TYPES)}

        for batch in dataloader:
            if len(batch) == 2:
                data, labels = batch
                labels = labels.to(self.device)

                # Check if data is emails (list of dicts) or features (tensor)
                if isinstance(data, (list, tuple)) and isinstance(data[0], dict):
                    preds = self.predict(data, method)
                else:
                    # Assume pre-extracted features - use first member
                    # This is a fallback for simple evaluation
                    member = self.members[0]
                    data = data.to(self.device)
                    output = member.model(data)
                    preds = output.action_logits.argmax(dim=-1)

                correct = (preds == labels).sum().item()
                total_correct += correct
                total_samples += len(labels)

                # Per-class accuracy
                for i in range(len(labels)):
                    label = labels[i].item()
                    pred = preds[i].item()
                    class_total[label] += 1
                    if pred == label:
                        class_correct[label] += 1

        # Compute metrics
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        per_class_acc = {}
        for cls in range(NUM_ACTION_TYPES):
            if class_total[cls] > 0:
                per_class_acc[cls] = class_correct[cls] / class_total[cls]
            else:
                per_class_acc[cls] = 0.0

        return {
            'accuracy': accuracy,
            'total_samples': total_samples,
            'per_class_accuracy': per_class_acc,
        }

    def get_member_predictions(
        self,
        emails: Union[dict, list[dict]],
    ) -> dict[str, torch.Tensor]:
        """Get individual predictions from each member.

        Useful for analyzing ensemble disagreement.

        Args:
            emails: Single email dict or list of email dicts

        Returns:
            Dictionary mapping member names to prediction tensors
        """
        if isinstance(emails, dict):
            emails = [emails]

        predictions = {}
        for member in self.members:
            features = self._extract_features_batch(emails, member)
            preds = self._get_member_prediction(features, member)
            predictions[member.name] = preds

        return predictions

    def get_agreement_score(
        self,
        emails: Union[dict, list[dict]],
    ) -> torch.Tensor:
        """Compute agreement score across ensemble members.

        Returns the fraction of members agreeing on the majority prediction.

        Args:
            emails: Single email dict or list of email dicts

        Returns:
            Tensor of shape (batch,) with agreement scores [0, 1]
        """
        predictions = self.get_member_predictions(emails)

        # Stack predictions
        pred_stack = torch.stack(list(predictions.values()), dim=0)  # (n_members, batch)

        # Get majority vote
        votes = torch.zeros(pred_stack.size(1), NUM_ACTION_TYPES, device=self.device)
        for i, name in enumerate(predictions.keys()):
            votes.scatter_add_(1, predictions[name].unsqueeze(1), torch.ones_like(votes[:, :1]))

        max_votes = votes.max(dim=-1).values
        agreement = max_votes / len(self.members)

        return agreement

    @classmethod
    def from_checkpoint_dir(
        cls,
        checkpoint_dir: Union[str, Path],
        device: str = 'auto',
    ) -> 'EnsemblePredictor':
        """Load ensemble from a checkpoint directory.

        Expects directory structure:
            checkpoint_dir/
                variant_name/
                    final.pt        # Model checkpoint
                    config.json     # Variant configuration
                ensemble_summary.json  # Optional summary with accuracies

        Args:
            checkpoint_dir: Path to ensemble checkpoint directory
            device: Device for inference

        Returns:
            EnsemblePredictor instance
        """
        checkpoint_dir = Path(checkpoint_dir)

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        # Load summary if available
        summary = {}
        summary_path = checkpoint_dir / 'ensemble_summary.json'
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)

        # Find all variant subdirectories
        members = []
        for variant_dir in checkpoint_dir.iterdir():
            if not variant_dir.is_dir():
                continue

            config_path = variant_dir / 'config.json'
            checkpoint_path = variant_dir / 'final.pt'

            if not config_path.exists() or not checkpoint_path.exists():
                continue

            # Load config
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Create feature extractor
            extractor = CombinedFeatureExtractor(
                include_content=config.get('include_content', False),
                content_model=config.get('content_model', 'all-MiniLM-L6-v2'),
            )

            # Create and load model
            input_dim = config.get('input_dim', FEATURE_DIMS['total_base'])
            model = create_policy_network(input_dim=input_dim)

            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Get validation accuracy
            val_acc = 0.0
            if 'final_metrics' in config:
                val_acc = config['final_metrics'].get('val_accuracy', 0.0)
            elif summary.get('results', {}).get(config['name'], {}):
                val_acc = summary['results'][config['name']].get('val_accuracy', 0.0)

            member = ModelMember(
                name=config['name'],
                model=model,
                extractor=extractor,
                input_dim=input_dim,
                validation_accuracy=val_acc,
                config=config,
            )
            members.append(member)
            print(f"Loaded model: {config['name']} (input_dim={input_dim}, val_acc={val_acc:.1%})")

        if not members:
            raise ValueError(f"No valid model checkpoints found in {checkpoint_dir}")

        return cls(members=members, device=device)

    def save_meta_classifier(self, path: Union[str, Path]) -> None:
        """Save trained meta-classifier."""
        if self._meta_classifier is None:
            raise ValueError("No meta-classifier trained yet")

        torch.save({
            'meta_classifier': self._meta_classifier.state_dict(),
            'input_dim': len(self.members) * NUM_ACTION_TYPES,
        }, path)
        print(f"Saved meta-classifier to {path}")

    def load_meta_classifier(self, path: Union[str, Path]) -> None:
        """Load trained meta-classifier."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        input_dim = checkpoint['input_dim']
        self._meta_classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, NUM_ACTION_TYPES),
        ).to(self.device)

        self._meta_classifier.load_state_dict(checkpoint['meta_classifier'])
        self._meta_classifier.eval()
        print(f"Loaded meta-classifier from {path}")


# Action index to name mapping for display
IDX_TO_ACTION = {
    0: 'reply_now',
    1: 'reply_later',
    2: 'forward',
    3: 'archive',
    4: 'delete',
}


def compare_methods(
    ensemble: EnsemblePredictor,
    dataloader: DataLoader,
) -> dict[str, dict]:
    """Compare all ensemble methods on a dataset.

    Args:
        ensemble: Trained EnsemblePredictor
        dataloader: Evaluation DataLoader

    Returns:
        Dictionary mapping method names to evaluation metrics
    """
    results = {}

    for method in EnsembleMethod:
        if method == EnsembleMethod.STACKING and ensemble._meta_classifier is None:
            continue

        metrics = ensemble.evaluate(dataloader, method=method)
        results[method.value] = metrics
        print(f"{method.value}: {metrics['accuracy']:.1%}")

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("ENSEMBLE PREDICTOR TEST")
    print("=" * 60)

    # Create mock ensemble for testing
    from policy_network import create_policy_network

    members = []
    for name, input_dim in [('balanced', 69), ('focal', 69), ('oversampled', 69)]:
        model = create_policy_network(input_dim=input_dim)
        extractor = CombinedFeatureExtractor(include_content=False)
        member = ModelMember(
            name=name,
            model=model,
            extractor=extractor,
            input_dim=input_dim,
            validation_accuracy=0.6 + 0.1 * len(members),  # Mock accuracies
        )
        members.append(member)

    ensemble = EnsemblePredictor(members=members, device='cpu')

    # Test prediction
    sample_email = {
        'from': 'john@example.com',
        'to': 'jane@example.com',
        'subject': 'URGENT: Need your input',
        'body': 'Please review and respond by EOD.',
    }

    print(f"\nTest email: {sample_email['subject']}")

    for method in [EnsembleMethod.HARD_VOTING, EnsembleMethod.SOFT_VOTING, EnsembleMethod.WEIGHTED_AVG]:
        probs = ensemble.predict_proba(sample_email, method=method)
        pred = probs.argmax().item()
        print(f"\n{method.value}:")
        print(f"  Prediction: {IDX_TO_ACTION[pred]}")
        print(f"  Probabilities: {probs[0].tolist()[:3]}...")

    # Test agreement score
    agreement = ensemble.get_agreement_score(sample_email)
    print(f"\nAgreement score: {agreement.item():.1%}")

    # Test individual predictions
    member_preds = ensemble.get_member_predictions(sample_email)
    print("\nIndividual predictions:")
    for name, pred in member_preds.items():
        print(f"  {name}: {IDX_TO_ACTION[pred.item()]}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
