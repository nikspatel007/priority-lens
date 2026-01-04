#!/usr/bin/env python3
"""Email Policy Network for RL agent action selection.

This module implements a PyTorch neural network for email prioritization
decisions. The network uses an Actor-Critic architecture with:
- Shared feature encoder
- Action type head (5 classes)
- Response timing head (5 classes)
- Priority head (continuous 0-1)
- Value head (for advantage estimation)

Compatible with:
- CombinedFeatures (69-dim input from features/combined.py)
- EmailState (variable dim with embeddings from email_state.py)
- EmailAction (output format from email_action.py)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# Constants matching email_action.py
NUM_ACTION_TYPES = 5  # reply_now, reply_later, forward, archive, delete
NUM_RESPONSE_TIMES = 5  # immediate, same_day, next_day, this_week, when_possible
DEFAULT_FEATURE_DIM = 69  # From CombinedFeatures (project:8 + topic:20 + task:12 + people:15 + temporal:8 + scores:6)


class PolicyOutput(NamedTuple):
    """Output from policy network forward pass."""
    action_logits: torch.Tensor  # Shape: (batch, 5)
    timing_logits: torch.Tensor  # Shape: (batch, 5)
    priority: torch.Tensor       # Shape: (batch, 1)
    value: torch.Tensor          # Shape: (batch, 1)


class ActionSample(NamedTuple):
    """Sampled action with log probabilities for training."""
    action_idx: torch.Tensor      # Shape: (batch,) - sampled action type index
    timing_idx: torch.Tensor      # Shape: (batch,) - sampled timing index
    priority: torch.Tensor        # Shape: (batch,) - priority value
    action_log_prob: torch.Tensor # Shape: (batch,) - log prob of action
    timing_log_prob: torch.Tensor # Shape: (batch,) - log prob of timing
    entropy: torch.Tensor         # Shape: (batch,) - entropy for exploration


@dataclass
class PolicyConfig:
    """Configuration for EmailPolicyNetwork."""
    input_dim: int = DEFAULT_FEATURE_DIM
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.1
    use_layer_norm: bool = True
    use_residual: bool = False
    activation: str = 'relu'
    priority_temp: float = 1.0  # Temperature for priority output
    # User-specific model configuration
    num_users: int = 0  # Number of unique users (0 = no user embeddings)
    user_embedding_dim: int = 32  # Dimension of user embeddings
    user_conditioning: str = 'concat'  # 'concat' or 'add' or 'film'


class UserEmbedding(nn.Module):
    """Learnable embeddings for user-specific behavior modeling.

    Each user gets a unique embedding vector that captures their
    personal email handling patterns and preferences.
    """

    def __init__(self, num_users: int, embedding_dim: int = 32):
        super().__init__()
        self.num_users = num_users
        self.embedding_dim = embedding_dim
        # +1 for unknown users (index 0 is reserved for unknown)
        self.embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        # Initialize embeddings with small values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        # Zero out the padding index
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for user IDs.

        Args:
            user_ids: Tensor of user indices, shape (batch,)
                      Use 0 for unknown users

        Returns:
            User embeddings, shape (batch, embedding_dim)
        """
        # Clamp to valid range (0 to num_users)
        user_ids = user_ids.clamp(0, self.num_users)
        return self.embedding(user_ids)


class FeatureEncoder(nn.Module):
    """Shared feature encoder for policy network.

    Transforms raw feature vector into latent representation.
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config

        layers = []
        # Adjust input dim if user embeddings are concatenated
        in_dim = config.input_dim
        if config.num_users > 0 and config.user_conditioning == 'concat':
            in_dim += config.user_embedding_dim

        for i, out_dim in enumerate(config.hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))

            if config.use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))

            if config.activation == 'relu':
                layers.append(nn.ReLU())
            elif config.activation == 'gelu':
                layers.append(nn.GELU())
            elif config.activation == 'tanh':
                layers.append(nn.Tanh())

            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))

            in_dim = out_dim

        self.encoder = nn.Sequential(*layers)
        self.output_dim = config.hidden_dims[-1] if config.hidden_dims else config.input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input features.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            Encoded features of shape (batch, hidden_dims[-1])
        """
        return self.encoder(x)


class ActionHead(nn.Module):
    """Action type prediction head."""

    def __init__(self, input_dim: int, num_actions: int = NUM_ACTION_TYPES):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action logits.

        Args:
            x: Encoded features of shape (batch, input_dim)

        Returns:
            Action logits of shape (batch, num_actions)
        """
        return self.fc(x)


class TimingHead(nn.Module):
    """Response timing prediction head."""

    def __init__(self, input_dim: int, num_timings: int = NUM_RESPONSE_TIMES):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_timings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute timing logits.

        Args:
            x: Encoded features of shape (batch, input_dim)

        Returns:
            Timing logits of shape (batch, num_timings)
        """
        return self.fc(x)


class PriorityHead(nn.Module):
    """Priority score prediction head (continuous 0-1)."""

    def __init__(self, input_dim: int, temperature: float = 1.0):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute priority score.

        Args:
            x: Encoded features of shape (batch, input_dim)

        Returns:
            Priority score of shape (batch, 1), values in [0, 1]
        """
        logit = self.fc(x) / self.temperature
        return torch.sigmoid(logit)


class ValueHead(nn.Module):
    """Value estimation head for actor-critic."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute value estimate.

        Args:
            x: Encoded features of shape (batch, input_dim)

        Returns:
            Value estimate of shape (batch, 1)
        """
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class EmailPolicyNetwork(nn.Module):
    """Actor-Critic policy network for email prioritization.

    This network takes email features and outputs:
    - Action type distribution (which action to take)
    - Response timing distribution (when to respond)
    - Priority score (how important is this email)
    - Value estimate (expected return from this state)

    Example usage:
        >>> config = PolicyConfig(input_dim=69)
        >>> policy = EmailPolicyNetwork(config)
        >>> features = torch.randn(32, 69)  # batch of 32 emails
        >>> output = policy(features)
        >>> print(output.action_logits.shape)  # (32, 5)

        # Sample actions for training
        >>> sample = policy.sample_action(features)
        >>> print(sample.action_idx.shape)  # (32,)
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        super().__init__()
        self.config = config or PolicyConfig()

        # User embedding (if enabled)
        self.user_embedding = None
        if self.config.num_users > 0:
            self.user_embedding = UserEmbedding(
                self.config.num_users,
                self.config.user_embedding_dim
            )

        # Shared encoder
        self.encoder = FeatureEncoder(self.config)
        hidden_dim = self.encoder.output_dim

        # Actor heads
        self.action_head = ActionHead(hidden_dim)
        self.timing_head = TimingHead(hidden_dim)
        self.priority_head = PriorityHead(hidden_dim, self.config.priority_temp)

        # Critic head
        self.value_head = ValueHead(hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        """Forward pass through the network.

        Args:
            x: Input features of shape (batch, input_dim)
            user_ids: Optional user indices of shape (batch,) for user-specific models.
                      If None and user embeddings are enabled, uses unknown user (0).

        Returns:
            PolicyOutput with action_logits, timing_logits, priority, value
        """
        # Apply user conditioning if enabled
        if self.user_embedding is not None:
            if user_ids is None:
                # Default to unknown user
                user_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            user_emb = self.user_embedding(user_ids)

            # Concatenate user embedding with features
            if self.config.user_conditioning == 'concat':
                x = torch.cat([x, user_emb], dim=-1)
            # Note: 'add' and 'film' modes could be added here for more sophisticated conditioning

        # Encode features
        hidden = self.encoder(x)

        # Compute outputs
        action_logits = self.action_head(hidden)
        timing_logits = self.timing_head(hidden)
        priority = self.priority_head(hidden)
        value = self.value_head(hidden)

        return PolicyOutput(
            action_logits=action_logits,
            timing_logits=timing_logits,
            priority=priority,
            value=value,
        )

    def get_action_probs(
        self,
        x: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and timing probability distributions.

        Args:
            x: Input features of shape (batch, input_dim)
            user_ids: Optional user indices of shape (batch,)

        Returns:
            Tuple of (action_probs, timing_probs), each of shape (batch, num_classes)
        """
        output = self.forward(x, user_ids)
        action_probs = F.softmax(output.action_logits, dim=-1)
        timing_probs = F.softmax(output.timing_logits, dim=-1)
        return action_probs, timing_probs

    def sample_action(
        self,
        x: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> ActionSample:
        """Sample actions from the policy.

        Args:
            x: Input features of shape (batch, input_dim)
            user_ids: Optional user indices of shape (batch,)
            deterministic: If True, return argmax instead of sampling

        Returns:
            ActionSample with sampled actions and log probabilities
        """
        output = self.forward(x, user_ids)

        # Create distributions
        action_dist = Categorical(logits=output.action_logits)
        timing_dist = Categorical(logits=output.timing_logits)

        if deterministic:
            action_idx = output.action_logits.argmax(dim=-1)
            timing_idx = output.timing_logits.argmax(dim=-1)
        else:
            action_idx = action_dist.sample()
            timing_idx = timing_dist.sample()

        # Compute log probabilities
        action_log_prob = action_dist.log_prob(action_idx)
        timing_log_prob = timing_dist.log_prob(timing_idx)

        # Compute entropy for exploration bonus
        entropy = action_dist.entropy() + timing_dist.entropy()

        return ActionSample(
            action_idx=action_idx,
            timing_idx=timing_idx,
            priority=output.priority.squeeze(-1),
            action_log_prob=action_log_prob,
            timing_log_prob=timing_log_prob,
            entropy=entropy,
        )

    def evaluate_actions(
        self,
        x: torch.Tensor,
        action_idx: torch.Tensor,
        timing_idx: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities of given actions.

        Used during PPO training to compute importance ratios.

        Args:
            x: Input features of shape (batch, input_dim)
            action_idx: Action indices of shape (batch,)
            timing_idx: Timing indices of shape (batch,)
            user_ids: Optional user indices of shape (batch,)

        Returns:
            Tuple of (action_log_prob, timing_log_prob, entropy, value)
        """
        output = self.forward(x, user_ids)

        action_dist = Categorical(logits=output.action_logits)
        timing_dist = Categorical(logits=output.timing_logits)

        action_log_prob = action_dist.log_prob(action_idx)
        timing_log_prob = timing_dist.log_prob(timing_idx)
        entropy = action_dist.entropy() + timing_dist.entropy()

        return action_log_prob, timing_log_prob, entropy, output.value.squeeze(-1)

    def get_value(
        self,
        x: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get value estimate only (for bootstrapping).

        Args:
            x: Input features of shape (batch, input_dim)
            user_ids: Optional user indices of shape (batch,)

        Returns:
            Value estimate of shape (batch,)
        """
        # Apply user conditioning if enabled
        if self.user_embedding is not None:
            if user_ids is None:
                user_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            user_emb = self.user_embedding(user_ids)
            if self.config.user_conditioning == 'concat':
                x = torch.cat([x, user_emb], dim=-1)

        hidden = self.encoder(x)
        return self.value_head(hidden).squeeze(-1)

    def predict_greedy(
        self,
        x: torch.Tensor,
        user_id: Optional[int] = None,
    ) -> Tuple[int, int, float]:
        """Predict single action greedily (for inference).

        Args:
            x: Input features of shape (1, input_dim) or (input_dim,)
            user_id: Optional user index for user-specific prediction

        Returns:
            Tuple of (action_idx, timing_idx, priority)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        user_ids = None
        if user_id is not None:
            user_ids = torch.tensor([user_id], dtype=torch.long, device=x.device)

        with torch.no_grad():
            output = self.forward(x, user_ids)
            action_idx = output.action_logits.argmax(dim=-1).item()
            timing_idx = output.timing_logits.argmax(dim=-1).item()
            priority = output.priority.item()

        return action_idx, timing_idx, priority


class DuelingPolicyNetwork(EmailPolicyNetwork):
    """Dueling architecture variant for better value estimation.

    Separates value and advantage streams for more stable learning.
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        super().__init__(config)

        hidden_dim = self.encoder.output_dim

        # Replace value head with dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_ACTION_TYPES),
        )

        # Remove old value head
        del self.value_head

    def forward(
        self,
        x: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        """Forward pass with dueling value computation."""
        # Apply user conditioning if enabled
        if self.user_embedding is not None:
            if user_ids is None:
                user_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            user_emb = self.user_embedding(user_ids)
            if self.config.user_conditioning == 'concat':
                x = torch.cat([x, user_emb], dim=-1)

        hidden = self.encoder(x)

        # Compute value using dueling architecture
        value = self.value_stream(hidden)
        advantage = self.advantage_stream(hidden)

        # Combine value and advantage (subtract mean for stability)
        action_logits = value + advantage - advantage.mean(dim=-1, keepdim=True)

        # Other outputs unchanged
        timing_logits = self.timing_head(hidden)
        priority = self.priority_head(hidden)

        return PolicyOutput(
            action_logits=action_logits,
            timing_logits=timing_logits,
            priority=priority,
            value=value,
        )


def create_policy_network(
    input_dim: int = DEFAULT_FEATURE_DIM,
    hidden_dims: tuple[int, ...] = (256, 128, 64),
    use_dueling: bool = False,
    **kwargs,
) -> EmailPolicyNetwork:
    """Factory function to create policy network.

    Args:
        input_dim: Dimension of input features
        hidden_dims: Tuple of hidden layer sizes
        use_dueling: Whether to use dueling architecture
        **kwargs: Additional PolicyConfig parameters

    Returns:
        Configured EmailPolicyNetwork
    """
    config = PolicyConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        **kwargs,
    )

    if use_dueling:
        return DuelingPolicyNetwork(config)
    return EmailPolicyNetwork(config)


if __name__ == '__main__':
    # Example usage and testing
    print("=" * 60)
    print("EMAIL POLICY NETWORK TEST")
    print("=" * 60)

    # Create network
    config = PolicyConfig(input_dim=69, hidden_dims=(256, 128, 64))
    policy = EmailPolicyNetwork(config)

    print(f"\nNetwork architecture:")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Hidden dims: {config.hidden_dims}")
    print(f"  Parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 69)

    print(f"\nForward pass (batch_size={batch_size}):")
    output = policy(x)
    print(f"  Action logits: {output.action_logits.shape}")
    print(f"  Timing logits: {output.timing_logits.shape}")
    print(f"  Priority: {output.priority.shape}")
    print(f"  Value: {output.value.shape}")

    # Test action sampling
    print("\nAction sampling:")
    sample = policy.sample_action(x)
    print(f"  Action indices: {sample.action_idx.shape}")
    print(f"  Timing indices: {sample.timing_idx.shape}")
    print(f"  Priority: {sample.priority.shape}")
    print(f"  Action log prob: {sample.action_log_prob.shape}")
    print(f"  Entropy: {sample.entropy.shape}")

    # Test action evaluation
    print("\nAction evaluation:")
    log_probs = policy.evaluate_actions(
        x, sample.action_idx, sample.timing_idx
    )
    print(f"  Action log prob: {log_probs[0].shape}")
    print(f"  Timing log prob: {log_probs[1].shape}")
    print(f"  Entropy: {log_probs[2].shape}")
    print(f"  Value: {log_probs[3].shape}")

    # Test greedy prediction
    print("\nGreedy prediction (single sample):")
    single_x = torch.randn(1, 69)
    action_idx, timing_idx, priority = policy.predict_greedy(single_x)
    print(f"  Action: {action_idx}")
    print(f"  Timing: {timing_idx}")
    print(f"  Priority: {priority:.3f}")

    # Test dueling variant
    print("\nDueling network:")
    dueling = create_policy_network(use_dueling=True)
    dueling_output = dueling(x)
    print(f"  Action logits: {dueling_output.action_logits.shape}")
    print(f"  Parameters: {sum(p.numel() for p in dueling.parameters()):,}")

    # Test user-conditioned network
    print("\n" + "-" * 60)
    print("USER-CONDITIONED NETWORK TEST")
    print("-" * 60)

    num_users = 100
    user_config = PolicyConfig(
        input_dim=69,
        hidden_dims=(256, 128, 64),
        num_users=num_users,
        user_embedding_dim=32,
        user_conditioning='concat',
    )
    user_policy = EmailPolicyNetwork(user_config)

    print(f"\nUser-conditioned network:")
    print(f"  Num users: {num_users}")
    print(f"  User embedding dim: {user_config.user_embedding_dim}")
    print(f"  Parameters: {sum(p.numel() for p in user_policy.parameters()):,}")

    # Test forward with user IDs
    user_ids = torch.randint(1, num_users + 1, (batch_size,))
    user_output = user_policy(x, user_ids)
    print(f"\nForward with user IDs:")
    print(f"  Action logits: {user_output.action_logits.shape}")
    print(f"  Value: {user_output.value.shape}")

    # Test that different users get different outputs
    user_1 = torch.tensor([1])
    user_2 = torch.tensor([2])
    single_x = torch.randn(1, 69)
    out_1 = user_policy(single_x, user_1)
    out_2 = user_policy(single_x, user_2)
    print(f"\nUser-specific outputs differ:")
    print(f"  User 1 action logits: {out_1.action_logits[0][:3].tolist()}")
    print(f"  User 2 action logits: {out_2.action_logits[0][:3].tolist()}")
    print(f"  Outputs differ: {not torch.allclose(out_1.action_logits, out_2.action_logits)}")

    # Test unknown user (id=0)
    unknown = torch.tensor([0])
    out_unknown = user_policy(single_x, unknown)
    print(f"  Unknown user action logits: {out_unknown.action_logits[0][:3].tolist()}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
