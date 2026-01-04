# RL System Architecture

## Overview

The system uses reinforcement learning to learn email handling behavior from the Enron dataset. The agent observes incoming emails and learns to predict the appropriate action based on what users actually did.

```
┌─────────────────────────────────────────────────────────────┐
│                    Email RL System                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │  Email   │───▶│   Feature    │───▶│   RL Agent      │   │
│  │  Input   │    │  Extractor   │    │   (Policy)      │   │
│  └──────────┘    └──────────────┘    └────────┬────────┘   │
│                                               │             │
│                                               ▼             │
│                                      ┌─────────────────┐   │
│                                      │  Action Output  │   │
│                                      │  - Priority     │   │
│                                      │  - Response     │   │
│                                      │  - Task?        │   │
│                                      └─────────────────┘   │
│                                               │             │
│  ┌──────────────────────────────────────────┐│             │
│  │           Reward Signal                   │◀            │
│  │  (Compare to actual user behavior)        │             │
│  └──────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. State Space (Email Representation)

The state is a vector representation of an incoming email:

```python
@dataclass
class EmailState:
    # Sender features
    sender_frequency: float      # How often this sender emails user
    sender_importance: float     # Derived from org hierarchy
    sender_reply_rate: float     # % of emails from sender that get replies

    # Content features
    subject_embedding: np.array  # Semantic embedding of subject
    body_embedding: np.array     # Semantic embedding of body
    has_question: bool           # Contains question marks, question words
    has_deadline: bool           # Mentions dates, "by EOD", "ASAP"
    has_attachment: bool
    email_length: int

    # Thread features
    is_reply: bool
    thread_length: int
    thread_participants: int
    user_already_replied: bool

    # Temporal features
    hour_of_day: int
    day_of_week: int
    time_since_last_email: float

    # Contextual features
    topic_vector: np.array       # Topic model output
    project_indicators: np.array # Known project mentions
    urgency_signals: float       # "urgent", "important", "ASAP"
```

### 2. Action Space

The agent outputs a multi-dimensional action:

```python
@dataclass
class EmailAction:
    # Primary action (discrete, 5-class)
    action_type: Literal[
        'reply_now',      # Respond within 1 hour
        'reply_later',    # Respond after 1 hour
        'forward',        # Forward to someone else
        'archive',        # Move to folder (no response)
        'delete',         # Delete/spam
    ]

    # Priority score (continuous, 0-1)
    priority: float

    # Response timing (if reply)
    suggested_response_time: Literal[
        'immediate',      # Within 1 hour
        'same_day',       # Within 8 hours
        'next_day',       # Within 24 hours
        'this_week',      # Within 7 days
        'when_possible',  # No urgency
    ]

    # Forward details (if forward)
    forward_to: Optional[str]  # Suggested recipient
```

### 3. Reward Function

The reward compares agent predictions to actual user behavior:

```python
def compute_reward(predicted_action, actual_action, email, user_context):
    """
    Reward function based on matching user behavior.

    Key signals from Enron data:
    - Did user reply? (found in sent folder)
    - How quickly did they reply?
    - Did they forward it?
    - Did they delete it?
    """
    reward = 0.0

    # Action type matching
    if predicted_action.action_type == actual_action.action_type:
        reward += 1.0
    elif is_similar_action(predicted_action.action_type, actual_action.action_type):
        reward += 0.5  # Partial credit for similar actions

    # Priority accuracy (for replied emails)
    if actual_action.action_type in ['reply_now', 'reply_later']:
        actual_response_time = actual_action.response_time_hours
        predicted_urgency = predicted_action.priority

        # Higher priority should correlate with faster response
        time_score = 1.0 - abs(predicted_urgency - urgency_from_time(actual_response_time))
        reward += time_score * 0.5

    # Negative reward for missing important emails
    if actual_action.action_type == 'reply_now' and predicted_action.action_type == 'archive':
        reward -= 2.0  # Significant penalty

    # Negative reward for false urgency
    if predicted_action.action_type == 'reply_now' and actual_action.action_type == 'archive':
        reward -= 0.5  # Minor penalty (better safe than sorry)

    return reward
```

### 4. Network Architecture

```python
import torch
import torch.nn as nn

class EmailPolicyNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim=768,
        hidden_dim=256,
        num_action_types=6,
    ):
        super().__init__()

        # Process email embeddings
        self.email_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 32, hidden_dim),  # subject + body + metadata
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Process sender/context features
        self.context_encoder = nn.Sequential(
            nn.Linear(64, hidden_dim // 2),
            nn.ReLU(),
        )

        # Combined processing
        self.combined = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Output heads
        self.action_head = nn.Linear(hidden_dim, num_action_types)
        self.priority_head = nn.Linear(hidden_dim, 1)
        self.timing_head = nn.Linear(hidden_dim, 5)  # 5 timing categories

    def forward(self, email_features, context_features):
        email_encoded = self.email_encoder(email_features)
        context_encoded = self.context_encoder(context_features)

        combined = torch.cat([email_encoded, context_encoded], dim=-1)
        hidden = self.combined(combined)

        action_logits = self.action_head(hidden)
        priority = torch.sigmoid(self.priority_head(hidden))
        timing_logits = self.timing_head(hidden)

        return action_logits, priority, timing_logits
```

### 5. RL Algorithm

We use **Proximal Policy Optimization (PPO)** for training:

```python
class EmailRLTrainer:
    def __init__(self, policy_net, lr=3e-4):
        self.policy = policy_net
        self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
        self.clip_epsilon = 0.2

    def train_step(self, batch):
        """PPO training step."""
        states, actions, rewards, old_log_probs = batch

        # Get current policy outputs
        action_logits, priorities, timing_logits = self.policy(
            states['email_features'],
            states['context_features']
        )

        # Compute new log probabilities
        action_dist = torch.distributions.Categorical(logits=action_logits)
        new_log_probs = action_dist.log_prob(actions['action_type'])

        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * rewards, clipped_ratio * rewards).mean()

        # Priority regression loss
        priority_loss = nn.MSELoss()(priorities.squeeze(), actions['priority_target'])

        # Total loss
        loss = policy_loss + 0.5 * priority_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Data Preparation                                             │
│     ┌──────────┐    ┌──────────────┐    ┌─────────────────┐     │
│     │  Enron   │───▶│   Parse &    │───▶│   Label with    │     │
│     │  Emails  │    │   Clean      │    │   User Actions  │     │
│     └──────────┘    └──────────────┘    └─────────────────┘     │
│                                                                  │
│  2. Feature Extraction                                           │
│     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│     │   Embed      │    │   Extract    │    │   Build      │    │
│     │   Text       │───▶│   Metadata   │───▶│   State      │    │
│     │   (BERT)     │    │   Features   │    │   Vectors    │    │
│     └──────────────┘    └──────────────┘    └──────────────┘    │
│                                                                  │
│  3. Training Loop                                                │
│     ┌──────────────────────────────────────────────────────┐    │
│     │  For each user:                                       │    │
│     │    - Sample email from inbox                          │    │
│     │    - Agent predicts action                            │    │
│     │    - Compare to actual user behavior                  │    │
│     │    - Compute reward                                   │    │
│     │    - Update policy (PPO)                              │    │
│     └──────────────────────────────────────────────────────┘    │
│                                                                  │
│  4. Evaluation                                                   │
│     - Hold out subset of users for testing                       │
│     - Measure action prediction accuracy                         │
│     - Measure priority ranking correlation                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Inference Flow

When deployed, the system processes new emails:

```python
class EmailAgent:
    def __init__(self, policy_path):
        self.policy = EmailPolicyNetwork()
        self.policy.load_state_dict(torch.load(policy_path))
        self.policy.eval()

        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def process_email(self, email, user_context):
        """Process a new email and recommend action."""

        # Extract features
        state = self.extract_features(email, user_context)

        # Get policy prediction
        with torch.no_grad():
            action_logits, priority, timing_logits = self.policy(
                state['email_features'],
                state['context_features']
            )

        # Decode outputs
        action_type = ACTION_TYPES[action_logits.argmax()]
        priority_score = priority.item()
        suggested_timing = TIMING_OPTIONS[timing_logits.argmax()]

        return {
            'action': action_type,
            'priority': priority_score,
            'timing': suggested_timing,
            'requires_response': action_type in ('reply_now', 'reply_later'),
            'context': self.extract_context(email),
        }
```

## Next Steps

- See [Feature Extraction](./03-features.md) for detailed feature definitions
- See [Training Guide](./04-training.md) for training instructions
