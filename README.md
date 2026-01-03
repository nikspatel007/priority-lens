# Email RL System

A reinforcement learning system that learns to prioritize, classify, and recommend actions for emails based on human behavior patterns. Trained on the Enron email dataset with a **target of 95% accuracy**.

## What This Does

When a user receives an email, the system predicts:
- **Action**: Reply now, reply later, forward, archive, delete, or create task
- **Priority**: How urgently does this need attention (0-1 score)
- **Task Creation**: Should this become a tracked task item?
- **Context**: What information is relevant for handling this?

The RL agent learns by comparing its predictions against actual user behavior in the Enron dataset.

## Scoring Dimensions

Emails are scored across multiple dimensions:
- **People**: Sender importance, org level, relationship strength
- **Project**: Active project references, deadlines, user's role
- **Topic**: Meeting request, task assignment, decision needed, FYI
- **Task**: Deadlines, action items, deliverables mentioned
- **Action**: Expected response type based on all signals

## Training Pipeline (95% Accuracy Target)

| Stage | Method | Expected Accuracy |
|-------|--------|------------------|
| 1 | Supervised Fine-Tuning (SFT) | 65-70% |
| 2 | Reward Model Training | - |
| 3 | GRPO (DeepSeek algorithm) | 80-85% |
| 4 | DPO (Direct Preference Optimization) | 88-90% |
| 5 | Temporal RLHF (future emails as feedback) | 92-94% |
| 6 | Rejection Sampling Refinement | **95%+** |

## Hardware Requirements

- **Recommended**: Apple Silicon M4 Max with 128GB RAM
- Supports running 70B+ parameter models locally
- Uses MPS (Metal Performance Shaders) and MLX for acceleration

## Quick Start

```bash
# 1. Setup environment
conda create -n email-rl python=3.11
conda activate email-rl
pip install -r requirements.txt

# 2. Download Enron dataset
./scripts/download_enron.sh

# 3. Preprocess emails
python src/preprocess.py

# 4. Run full training pipeline
python src/train_full_pipeline.py --target_accuracy 0.95

# 5. Evaluate
python src/evaluate.py
```

## Documentation

See the [docs/](./docs/) folder for detailed documentation:

1. **[Dataset Setup](./docs/01-enron-dataset.md)** - Download and prepare Enron emails
2. **[System Architecture](./docs/02-architecture.md)** - RL system design
3. **[Feature Extraction](./docs/03-features.md)** - Scoring dimensions
4. **[Training Guide](./docs/04-training.md)** - Basic training
5. **[Advanced Training](./docs/05-advanced-training.md)** - Multi-stage RL for 95% accuracy

## Key Algorithms

- **GRPO** (Group Relative Policy Optimization) - DeepSeek's efficient alternative to PPO
- **DPO** (Direct Preference Optimization) - Anthropic-style direct alignment
- **KTO** (Kahneman-Tversky Optimization) - Works with unpaired preferences
- **Temporal RLHF** - Uses future emails as human feedback signal

## Project Structure

```
rl-emails/
├── README.md
├── requirements.txt
├── docs/                  # Documentation
├── scripts/               # Setup scripts
│   └── download_enron.sh
├── src/                   # Source code
│   ├── preprocess.py
│   ├── train.py
│   ├── train_full_pipeline.py
│   └── evaluate.py
├── data/                  # Dataset (after download)
├── models/                # Downloaded LLMs
└── checkpoints/           # Training checkpoints
```

## License

MIT
