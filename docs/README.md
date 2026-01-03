# Email RL System Documentation

A reinforcement learning system that learns to prioritize, classify, and recommend actions for emails based on human behavior patterns.

## Overview

This project uses the Enron email dataset to train an RL agent that learns to:
- Identify important emails requiring action
- Score emails based on people, projects, topics, and urgency
- Recommend appropriate responses and actions
- Determine if tasks need to be created
- Extract relevant contextual information

## Documentation Structure

1. **[Dataset Setup](./01-enron-dataset.md)** - How to download and prepare the Enron email dataset
2. **[System Architecture](./02-architecture.md)** - RL system design and components
3. **[Feature Extraction](./03-features.md)** - Email scoring dimensions and extraction methods
4. **[Training Guide](./04-training.md)** - How to train and evaluate the model

## Quick Start

```bash
# 1. Download dataset
./scripts/download_enron.sh

# 2. Preprocess emails
python src/preprocess.py

# 3. Train the model
python src/train.py

# 4. Evaluate
python src/evaluate.py
```

## Goal

When a user receives an email, the system should predict:
- **Response Priority**: How urgently does this need attention?
- **Action Type**: Reply, forward, archive, create task, delegate?
- **Task Creation**: Should this become a tracked task?
- **Context Needed**: What information is relevant for handling this?

The RL agent learns by comparing its predictions against actual user behavior in the Enron dataset.
