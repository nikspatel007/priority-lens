# Email RL System Documentation

A reinforcement learning system that learns to prioritize, classify, and recommend actions for emails based on human behavior patterns. **Target: 95% accuracy** using multi-stage RL training with state-of-the-art algorithms.

## Overview

This project uses the Enron email dataset to train an RL agent that learns to:
- Identify important emails requiring action
- Score emails based on people, projects, topics, and urgency
- Recommend appropriate responses and actions
- Determine if tasks need to be created
- Extract relevant contextual information

## Hardware Requirements

- **Recommended**: Apple Silicon M4 Max with 128GB RAM
- Supports running 70B+ parameter models locally
- Uses MPS (Metal Performance Shaders) and MLX for acceleration

## Documentation Structure

1. **[Dataset Setup](./01-enron-dataset.md)** - How to download and prepare the Enron email dataset
2. **[System Architecture](./02-architecture.md)** - RL system design and components
3. **[Feature Extraction](./03-features.md)** - Email scoring dimensions and extraction methods
4. **[Training Guide](./04-training.md)** - Basic training methodology
5. **[Advanced Training](./05-advanced-training.md)** - Multi-stage RL for 95% accuracy (GRPO, DPO, KTO, RLHF)

## Quick Start

```bash
# 1. Setup environment (Apple Silicon optimized)
conda create -n email-rl python=3.11
conda activate email-rl
pip install -r requirements.txt

# 2. Download dataset
./scripts/download_enron.sh

# 3. Preprocess emails
python src/preprocess.py

# 4. Train the model (basic)
python src/train.py

# 5. For 95% accuracy, run full pipeline
python src/train_full_pipeline.py --target_accuracy 0.95

# 6. Evaluate
python src/evaluate.py
```

## Training Pipeline (95% Accuracy Target)

| Stage | Method | Expected Accuracy |
|-------|--------|------------------|
| 1 | Supervised Fine-Tuning (SFT) | 65-70% |
| 2 | Reward Model Training | - |
| 3 | GRPO (DeepSeek algorithm) | 80-85% |
| 4 | DPO (Direct Preference Optimization) | 88-90% |
| 5 | Temporal RLHF (future emails as feedback) | 92-94% |
| 6 | Rejection Sampling Refinement | **95%+** |

See [Advanced Training Guide](./05-advanced-training.md) for full details.

## Goal

When a user receives an email, the system should predict:
- **Response Priority**: How urgently does this need attention?
- **Action Type**: Reply, forward, archive, create task, delegate?
- **Task Creation**: Should this become a tracked task?
- **Context Needed**: What information is relevant for handling this?

The RL agent learns by comparing its predictions against actual user behavior in the Enron dataset.
