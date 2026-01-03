# Training Guide

## Overview

This guide covers how to train the email RL agent using the Enron dataset, including data preparation, training configuration, and evaluation metrics.

## Prerequisites

```bash
# Python dependencies
pip install torch transformers sentence-transformers
pip install scikit-learn pandas numpy
pip install tqdm wandb  # For logging
```

## Data Preparation

### Step 1: Download and Parse Enron Data

```bash
# Download dataset
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
tar -xzf enron_mail_20150507.tar.gz

# Parse into structured format
python scripts/parse_enron.py --input maildir/ --output data/emails.json
```

### Step 2: Label User Actions

The key insight is that user behavior in Enron data provides our training signal:

```python
# scripts/label_actions.py

def label_user_actions(emails_json_path, output_path):
    """
    For each inbox email, determine what action the user took.

    Labels derived from:
    - sent/ folder: User replied
    - sent_items/ folder: User replied or forwarded
    - deleted_items/ folder: User deleted
    - No corresponding action: Archived/ignored
    """
    with open(emails_json_path) as f:
        all_emails = json.load(f)

    # Group by user
    by_user = defaultdict(list)
    for email in all_emails:
        by_user[email['user']].append(email)

    labeled_data = []

    for user, emails in tqdm(by_user.items()):
        # Separate by folder type
        inbox = [e for e in emails if 'inbox' in e['folder'].lower()]
        sent = [e for e in emails if 'sent' in e['folder'].lower()]
        deleted = [e for e in emails if 'deleted' in e['folder'].lower()]

        # Build reply index
        sent_by_reply_to = {}
        for e in sent:
            if e.get('in_reply_to'):
                sent_by_reply_to[e['in_reply_to']] = e

        # Label each inbox email
        for email in inbox:
            msg_id = email['message_id']
            label = {
                'email': email,
                'user': user,
            }

            if msg_id in sent_by_reply_to:
                reply = sent_by_reply_to[msg_id]
                label['action'] = 'replied'
                label['reply_email'] = reply
                label['response_time_hours'] = compute_response_time(
                    email['date'], reply['date']
                )
            elif email in deleted:
                label['action'] = 'deleted'
            else:
                # Check if forwarded
                if was_forwarded(email, sent):
                    label['action'] = 'forwarded'
                else:
                    label['action'] = 'archived'

            labeled_data.append(label)

    with open(output_path, 'w') as f:
        json.dump(labeled_data, f, indent=2)

    return labeled_data
```

### Step 3: Generate Features

```python
# scripts/generate_features.py

def generate_training_features(labeled_data, output_path):
    """
    Convert labeled emails into feature vectors.
    """
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    features_list = []

    for item in tqdm(labeled_data):
        email = item['email']
        user = item['user']

        # Text embeddings
        subject_emb = embedder.encode(email.get('subject', ''))
        body_emb = embedder.encode(email.get('body', '')[:1000])

        # Build user context from historical data
        user_context = build_user_context(user, labeled_data)

        # Extract all feature scores
        people_features = extract_people_features(email, user_context)
        topic_features = classify_topic(email)
        task_features = extract_tasks(email)

        # Combine into single record
        features_list.append({
            'email_id': email['message_id'],
            'user': user,
            'subject_embedding': subject_emb.tolist(),
            'body_embedding': body_emb.tolist(),
            'people_score': compute_people_score(people_features),
            'topic_score': compute_topic_score(topic_features),
            'task_score': compute_task_score(task_features),
            'is_question': topic_features.is_question,
            'is_action_request': topic_features.is_action_request,
            'urgency': topic_features.urgency_language,
            'sender_org_level': people_features.sender_org_level,
            # Target labels
            'action': item['action'],
            'response_time_hours': item.get('response_time_hours'),
        })

    # Save as parquet for efficient loading
    df = pd.DataFrame(features_list)
    df.to_parquet(output_path)

    return df
```

### Step 4: Create Train/Val/Test Splits

```python
# Split by user to test generalization
def create_splits(features_df):
    """
    Split data by user, not by email.

    This tests if the model generalizes to new users.
    """
    users = features_df['user'].unique()
    np.random.shuffle(users)

    n_users = len(users)
    train_users = users[:int(0.7 * n_users)]
    val_users = users[int(0.7 * n_users):int(0.85 * n_users)]
    test_users = users[int(0.85 * n_users):]

    train_df = features_df[features_df['user'].isin(train_users)]
    val_df = features_df[features_df['user'].isin(val_users)]
    test_df = features_df[features_df['user'].isin(test_users)]

    return train_df, val_df, test_df
```

## Training Configuration

### Model Hyperparameters

```python
# config/training_config.yaml

model:
  embedding_dim: 384  # Sentence transformer output
  hidden_dim: 256
  num_action_types: 6
  dropout: 0.2

training:
  batch_size: 64
  learning_rate: 3e-4
  epochs: 50
  early_stopping_patience: 5

  # PPO specific
  clip_epsilon: 0.2
  value_loss_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5

  # Reward weights
  action_match_reward: 1.0
  priority_correlation_reward: 0.5
  missed_important_penalty: 2.0
  false_urgent_penalty: 0.5

data:
  max_body_length: 1000
  min_emails_per_user: 50  # Filter users with too few emails
```

## Training Loop

```python
# src/train.py

import torch
from torch.utils.data import DataLoader
import wandb

def train(config):
    # Initialize
    wandb.init(project="email-rl", config=config)

    # Load data
    train_df = pd.read_parquet('data/train_features.parquet')
    val_df = pd.read_parquet('data/val_features.parquet')

    train_dataset = EmailDataset(train_df)
    val_dataset = EmailDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # Model
    model = EmailPolicyNetwork(
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training loop
    best_val_accuracy = 0
    patience_counter = 0

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # Forward pass
            action_logits, priority, timing_logits = model(
                batch['email_features'],
                batch['context_features']
            )

            # Compute loss
            action_loss = F.cross_entropy(action_logits, batch['action_label'])
            priority_loss = F.mse_loss(priority.squeeze(), batch['priority_target'])

            loss = action_loss + 0.5 * priority_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()

            train_loss += loss.item()
            train_correct += (action_logits.argmax(dim=1) == batch['action_label']).sum().item()
            train_total += len(batch['action_label'])

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                action_logits, _, _ = model(
                    batch['email_features'],
                    batch['context_features']
                )
                val_correct += (action_logits.argmax(dim=1) == batch['action_label']).sum().item()
                val_total += len(batch['action_label'])

        val_accuracy = val_correct / val_total
        train_accuracy = train_correct / train_total

        # Logging
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss / len(train_loader),
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
        })

        print(f"Epoch {epoch+1}: Train Acc={train_accuracy:.3f}, Val Acc={val_accuracy:.3f}")

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print("Early stopping triggered")
                break

    return model
```

## Reinforcement Learning Training

For more advanced training with PPO:

```python
# src/train_rl.py

class EmailEnvironment:
    """
    RL environment that simulates email processing.
    """
    def __init__(self, emails_df):
        self.emails = emails_df.to_dict('records')
        self.current_idx = 0

    def reset(self):
        self.current_idx = 0
        return self._get_state()

    def _get_state(self):
        if self.current_idx >= len(self.emails):
            return None
        email = self.emails[self.current_idx]
        return {
            'email_features': torch.tensor(
                email['subject_embedding'] + email['body_embedding']
            ),
            'context_features': torch.tensor([
                email['people_score'],
                email['topic_score'],
                email['task_score'],
                email['is_question'],
                email['is_action_request'],
                email['urgency'],
            ]),
        }

    def step(self, action):
        """
        Take action, return reward based on actual user behavior.
        """
        email = self.emails[self.current_idx]
        actual_action = email['action']

        # Compute reward
        reward = self._compute_reward(action, actual_action, email)

        # Move to next email
        self.current_idx += 1
        done = self.current_idx >= len(self.emails)
        next_state = None if done else self._get_state()

        return next_state, reward, done

    def _compute_reward(self, predicted, actual, email):
        """
        Reward function.
        """
        action_map = {'replied': 0, 'forwarded': 1, 'archived': 2, 'deleted': 3}
        actual_idx = action_map.get(actual, 2)

        reward = 0.0

        # Exact match
        if predicted == actual_idx:
            reward += 1.0
        # Similar action partial credit
        elif (predicted in [0, 1] and actual_idx in [0, 1]):  # Both are responses
            reward += 0.5

        # Penalty for missing important emails
        if actual == 'replied' and email.get('response_time_hours', 24) < 4:
            # User replied quickly = was important
            if predicted in [2, 3]:  # Archive or delete
                reward -= 2.0

        return reward


def train_ppo(env, policy, config):
    """
    PPO training loop.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=config['learning_rate'])

    for episode in range(config['num_episodes']):
        state = env.reset()
        episode_rewards = []
        states, actions, rewards, log_probs = [], [], [], []

        while state is not None:
            # Get action from policy
            with torch.no_grad():
                action_logits, _, _ = policy(
                    state['email_features'].unsqueeze(0),
                    state['context_features'].unsqueeze(0)
                )
                action_dist = torch.distributions.Categorical(logits=action_logits)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

            # Take action
            next_state, reward, done = env.step(action.item())

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            episode_rewards.append(reward)

            state = next_state

        # PPO update
        returns = compute_returns(rewards, config['gamma'])
        advantages = returns - torch.tensor(returns).mean()

        # Multiple PPO epochs on collected data
        for _ in range(config['ppo_epochs']):
            for i in range(len(states)):
                action_logits, _, _ = policy(
                    states[i]['email_features'].unsqueeze(0),
                    states[i]['context_features'].unsqueeze(0)
                )
                action_dist = torch.distributions.Categorical(logits=action_logits)
                new_log_prob = action_dist.log_prob(actions[i])

                ratio = torch.exp(new_log_prob - log_probs[i])
                clipped = torch.clamp(ratio, 1 - config['clip_epsilon'], 1 + config['clip_epsilon'])

                loss = -torch.min(ratio * advantages[i], clipped * advantages[i])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Episode {episode}: Mean Reward = {np.mean(episode_rewards):.3f}")
```

## Evaluation Metrics

### Action Prediction Accuracy

```python
def evaluate_action_accuracy(model, test_loader):
    """
    Measure how often model predicts correct action.
    """
    model.eval()
    correct = 0
    total = 0
    confusion = np.zeros((6, 6))

    with torch.no_grad():
        for batch in test_loader:
            logits, _, _ = model(batch['email_features'], batch['context_features'])
            preds = logits.argmax(dim=1)
            labels = batch['action_label']

            correct += (preds == labels).sum().item()
            total += len(labels)

            for p, l in zip(preds, labels):
                confusion[l, p] += 1

    accuracy = correct / total
    return accuracy, confusion
```

### Priority Ranking Correlation

```python
def evaluate_priority_ranking(model, test_df):
    """
    Measure if model's priority scores correlate with response time.

    Emails that got quick replies should have higher priority scores.
    """
    model.eval()

    predictions = []
    actuals = []

    for _, row in test_df.iterrows():
        features = build_features_from_row(row)

        with torch.no_grad():
            _, priority, _ = model(
                features['email_features'].unsqueeze(0),
                features['context_features'].unsqueeze(0)
            )

        predictions.append(priority.item())

        # Actual urgency = inverse of response time
        if row['response_time_hours'] is not None:
            actual_urgency = 1.0 / (1 + row['response_time_hours'])
        else:
            actual_urgency = 0.1  # Low urgency for non-replied

        actuals.append(actual_urgency)

    correlation = np.corrcoef(predictions, actuals)[0, 1]
    return correlation
```

### Task Creation Precision/Recall

```python
def evaluate_task_creation(model, test_df):
    """
    Evaluate task creation recommendations.

    Ground truth: emails with deadlines/action items that got replies
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for _, row in test_df.iterrows():
        predicted_task = model_predicts_task(model, row)
        actual_task = should_have_been_task(row)

        if predicted_task and actual_task:
            true_positives += 1
        elif predicted_task and not actual_task:
            false_positives += 1
        elif not predicted_task and actual_task:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return {'precision': precision, 'recall': recall, 'f1': f1}
```

## Running Training

```bash
# Full training pipeline
python scripts/parse_enron.py --input maildir/ --output data/emails.json
python scripts/label_actions.py --input data/emails.json --output data/labeled.json
python scripts/generate_features.py --input data/labeled.json --output data/features.parquet

# Train model
python src/train.py --config config/training_config.yaml

# Or RL training
python src/train_rl.py --config config/rl_config.yaml

# Evaluate
python src/evaluate.py --model models/best_model.pt --test data/test_features.parquet
```

## Expected Results

| Metric | Baseline | Target |
|--------|----------|--------|
| Action Accuracy | 40% (random) | 70%+ |
| Priority Correlation | 0.0 | 0.5+ |
| Task F1 | 0.2 | 0.6+ |

## Iterating on the Model

1. **Feature Engineering**: Add more signals (calendar context, thread history)
2. **User Personalization**: Fine-tune per-user or add user embeddings
3. **Multi-Task Learning**: Joint training on action + priority + timing
4. **Curriculum Learning**: Start with clear-cut emails, progress to ambiguous
5. **Human-in-the-Loop**: Use model predictions with human feedback for fine-tuning
