# Advanced Training: Multi-Stage RL for 95% Accuracy

## Overview

This guide covers advanced training techniques to achieve 95%+ accuracy on email action prediction using:

1. **Multi-Stage RL Training** - Progressive skill building
2. **State-of-the-Art Algorithms** - GRPO (DeepSeek), DPO, KTO, PPO variants
3. **RLHF with Temporal Validation** - Using future emails as human feedback
4. **Local Model Deployment** - Optimized for Apple Silicon M4 Max (128GB RAM)

## Hardware Setup: Apple Silicon M4 Max

### Environment Configuration

```bash
# Create conda environment with Apple Silicon optimizations
conda create -n email-rl python=3.11
conda activate email-rl

# PyTorch with MPS (Metal Performance Shaders) support
pip install torch torchvision torchaudio

# MLX - Apple's ML framework (extremely fast on Apple Silicon)
pip install mlx mlx-lm

# Transformers with Apple Silicon support
pip install transformers accelerate bitsandbytes
pip install sentence-transformers

# RL libraries
pip install trl  # Transformer Reinforcement Learning (HuggingFace)
pip install peft  # Parameter-Efficient Fine-Tuning

# Additional dependencies
pip install datasets wandb scipy scikit-learn pandas numpy
pip install flash-attn --no-build-isolation  # If supported
```

### Verify MPS Backend

```python
import torch

# Check MPS availability
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Memory check (128GB unified memory)
import subprocess
result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
mem_bytes = int(result.stdout.split(':')[1].strip())
print(f"Total memory: {mem_bytes / (1024**3):.0f} GB")
```

### MLX Configuration for Large Models

```python
# mlx_config.py
import mlx.core as mx
import mlx.nn as nn

# MLX automatically uses unified memory efficiently
# Can load models up to ~100GB on 128GB machine

def load_model_mlx(model_path):
    """Load model using MLX for Apple Silicon."""
    from mlx_lm import load, generate

    model, tokenizer = load(model_path)
    return model, tokenizer
```

## Local Models for Email Understanding

### Recommended Models (Fit in 128GB)

| Model | Size | VRAM Needed | Use Case |
|-------|------|-------------|----------|
| Qwen2.5-72B-Instruct | 72B | ~80GB | Primary reasoning model |
| Llama-3.1-70B | 70B | ~75GB | Alternative backbone |
| Mistral-Large-2 | 123B | ~100GB (4-bit) | Maximum capability |
| DeepSeek-V2-Lite | 16B | ~20GB | Fast iteration |
| Phi-3-medium | 14B | ~16GB | Efficient baseline |

### Download and Setup Models

```bash
# Using Hugging Face CLI
pip install huggingface_hub

# Login (for gated models)
huggingface-cli login

# Download models
huggingface-cli download Qwen/Qwen2.5-72B-Instruct --local-dir ./models/qwen-72b
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct --local-dir ./models/llama-70b
huggingface-cli download deepseek-ai/DeepSeek-V2-Lite --local-dir ./models/deepseek-v2-lite

# For MLX-optimized versions (faster on Apple Silicon)
pip install mlx-lm
mlx_lm.convert --hf-path Qwen/Qwen2.5-72B-Instruct --mlx-path ./models/qwen-72b-mlx -q
```

### Model Loading with Memory Optimization

```python
# src/models/local_llm.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LocalLLM:
    """Load and manage local LLMs on Apple Silicon."""

    def __init__(self, model_name="Qwen/Qwen2.5-72B-Instruct", quantize=True):
        self.device = torch.device("mps")

        if quantize:
            # 4-bit quantization for larger models
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, prompt, max_tokens=512):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class MLXModel:
    """MLX-based model for maximum Apple Silicon performance."""

    def __init__(self, model_path="./models/qwen-72b-mlx"):
        from mlx_lm import load
        self.model, self.tokenizer = load(model_path)

    def generate(self, prompt, max_tokens=512):
        from mlx_lm import generate
        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=0.7,
        )
```

## Multi-Stage RL Training Pipeline

### Stage Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Multi-Stage Training Pipeline                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Stage 1: Supervised Pre-training (SFT)                                      │
│  ────────────────────────────────────                                        │
│  • Train on labeled email-action pairs                                       │
│  • Target: 60-70% accuracy baseline                                          │
│  • Loss: Cross-entropy on action prediction                                  │
│                                                                              │
│                           ▼                                                  │
│                                                                              │
│  Stage 2: Reward Model Training                                              │
│  ─────────────────────────────────                                           │
│  • Train reward model on preference pairs                                    │
│  • Use response time as implicit preference signal                           │
│  • Bradley-Terry preference modeling                                         │
│                                                                              │
│                           ▼                                                  │
│                                                                              │
│  Stage 3: PPO / GRPO Reinforcement Learning                                  │
│  ───────────────────────────────────────────                                 │
│  • Optimize policy against reward model                                      │
│  • Target: 80-85% accuracy                                                   │
│  • Use KL divergence constraint to prevent drift                             │
│                                                                              │
│                           ▼                                                  │
│                                                                              │
│  Stage 4: DPO / KTO Direct Alignment                                         │
│  ────────────────────────────────────                                        │
│  • Direct preference optimization without reward model                       │
│  • Use future emails as ground truth                                         │
│  • Target: 88-92% accuracy                                                   │
│                                                                              │
│                           ▼                                                  │
│                                                                              │
│  Stage 5: RLHF with Temporal Validation                                      │
│  ──────────────────────────────────────                                      │
│  • Use emails from future timeframes as "human feedback"                     │
│  • Iterative refinement with rejection sampling                              │
│  • Target: 95%+ accuracy                                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Supervised Fine-Tuning (SFT)

```python
# src/training/stage1_sft.py

from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import torch

class EmailSFTTrainer:
    """Stage 1: Supervised fine-tuning on email-action pairs."""

    def __init__(self, base_model_path, output_dir):
        self.device = torch.device("mps")

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Apply LoRA for efficient fine-tuning
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,  # Rank
            lora_alpha=128,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.model = get_peft_model(self.model, lora_config)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.output_dir = output_dir

    def prepare_sft_dataset(self, emails_df):
        """Convert emails to instruction format."""
        examples = []

        for _, row in emails_df.iterrows():
            prompt = self._format_email_prompt(row)
            response = self._format_action_response(row)

            examples.append({
                "prompt": prompt,
                "completion": response,
            })

        return Dataset.from_list(examples)

    def _format_email_prompt(self, row):
        return f"""Analyze this email and determine the appropriate action.

From: {row['from']}
To: {row['to']}
Subject: {row['subject']}
Body: {row['body'][:1500]}

Context:
- Sender importance: {row['people_score']:.2f}
- Topic urgency: {row['urgency']:.2f}
- Contains question: {row['is_question']}
- Action requested: {row['is_action_request']}

What action should be taken? Respond with:
1. Action type (reply_now, reply_later, forward, archive, delete)
2. Priority score (0-1)
3. Reasoning"""

    def _format_action_response(self, row):
        action_map = {
            'replied': 'reply_now' if row.get('response_time_hours', 24) < 4 else 'reply_later',
            'forwarded': 'forward',
            'deleted': 'delete',
            'archived': 'archive',
        }
        action = action_map.get(row['action'], 'archive')

        return f"""Action: {action}
Priority: {row.get('priority_target', 0.5):.2f}
Reasoning: Based on sender importance ({row['people_score']:.2f}) and content urgency."""

    def train(self, train_dataset, eval_dataset):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            fp16=True,  # Use fp16 on MPS
            optim="adamw_torch",
            report_to="wandb",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        trainer.save_model(f"{self.output_dir}/sft_final")

        return self.model
```

### Stage 2: Reward Model Training

```python
# src/training/stage2_reward_model.py

import torch
import torch.nn as nn
from transformers import AutoModel

class EmailRewardModel(nn.Module):
    """
    Reward model trained on preference pairs.

    Preferences derived from:
    - Response time (faster = higher reward)
    - Action taken vs ignored
    - Thread continuation patterns
    """

    def __init__(self, base_model_path):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
        )

        # Freeze most layers, fine-tune top layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.layers[-4:].parameters():
            param.requires_grad = True

        # Reward head
        hidden_size = self.backbone.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, -1, :]  # Last token
        reward = self.reward_head(hidden)
        return reward


def create_preference_pairs(emails_df):
    """
    Create preference pairs from Enron data.

    Preference signals:
    1. Replied quickly > Replied slowly > Not replied
    2. Action taken > Ignored
    3. Important sender > Less important sender
    """
    pairs = []

    # Group by user
    for user, user_emails in emails_df.groupby('user'):
        user_emails = user_emails.sort_values('date')

        for i, email_a in user_emails.iterrows():
            for j, email_b in user_emails.iterrows():
                if i >= j:
                    continue

                # Determine preference
                preference = compare_emails(email_a, email_b)

                if preference != 0:
                    if preference > 0:
                        chosen, rejected = email_a, email_b
                    else:
                        chosen, rejected = email_b, email_a

                    pairs.append({
                        'chosen': format_email_for_rm(chosen),
                        'rejected': format_email_for_rm(rejected),
                    })

    return pairs


def compare_emails(a, b):
    """
    Compare two emails, return preference.

    Returns:
        1 if a is preferred
        -1 if b is preferred
        0 if no clear preference
    """
    # Quick reply > slow reply > no reply
    a_replied = a['action'] == 'replied'
    b_replied = b['action'] == 'replied'

    if a_replied and not b_replied:
        return 1
    if b_replied and not a_replied:
        return -1

    if a_replied and b_replied:
        a_time = a.get('response_time_hours', 100)
        b_time = b.get('response_time_hours', 100)
        if a_time < b_time - 2:  # Significantly faster
            return 1
        if b_time < a_time - 2:
            return -1

    return 0


def train_reward_model(model, preference_pairs, output_dir):
    """Train reward model using Bradley-Terry loss."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    device = torch.device("mps")
    model = model.to(device)

    for epoch in range(3):
        total_loss = 0
        correct = 0

        for pair in tqdm(preference_pairs):
            chosen_inputs = tokenizer(pair['chosen'], return_tensors="pt").to(device)
            rejected_inputs = tokenizer(pair['rejected'], return_tensors="pt").to(device)

            chosen_reward = model(**chosen_inputs)
            rejected_reward = model(**rejected_inputs)

            # Bradley-Terry loss: maximize log sigmoid(chosen - rejected)
            loss = -torch.log(torch.sigmoid(chosen_reward - rejected_reward)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (chosen_reward > rejected_reward).sum().item()

        accuracy = correct / len(preference_pairs)
        print(f"Epoch {epoch}: Loss={total_loss/len(preference_pairs):.4f}, Acc={accuracy:.3f}")

    torch.save(model.state_dict(), f"{output_dir}/reward_model.pt")
    return model
```

### Stage 3: GRPO (Group Relative Policy Optimization)

DeepSeek's GRPO algorithm - more efficient than PPO:

```python
# src/training/stage3_grpo.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class GRPOTrainer:
    """
    GRPO: Group Relative Policy Optimization (DeepSeek).

    Key innovations:
    - No separate critic network needed
    - Groups responses and uses relative rewards
    - More stable than PPO
    """

    def __init__(self, policy_model, reward_model, ref_model, config):
        self.policy = policy_model
        self.reward_model = reward_model
        self.ref_model = ref_model  # Frozen reference for KL

        self.device = torch.device("mps")
        self.config = config

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config['learning_rate']
        )

    def train_step(self, prompts):
        """
        GRPO training step.

        1. Generate G responses per prompt
        2. Score with reward model
        3. Compute group-relative advantages
        4. Update policy with clipped objective
        """
        G = self.config['group_size']  # Typically 4-8

        all_responses = []
        all_rewards = []
        all_log_probs = []
        all_ref_log_probs = []

        # Generate G responses per prompt
        for prompt in prompts:
            responses = []
            log_probs = []

            for _ in range(G):
                response, log_prob = self.generate_with_log_prob(prompt)
                responses.append(response)
                log_probs.append(log_prob)

            # Score responses
            rewards = [self.reward_model.score(prompt, r) for r in responses]

            # Get reference log probs
            with torch.no_grad():
                ref_log_probs = [self.get_log_prob(self.ref_model, prompt, r) for r in responses]

            all_responses.extend(responses)
            all_rewards.extend(rewards)
            all_log_probs.extend(log_probs)
            all_ref_log_probs.extend(ref_log_probs)

        # Compute group-relative advantages
        rewards_tensor = torch.tensor(all_rewards, device=self.device)

        # Group rewards and normalize within groups
        rewards_grouped = rewards_tensor.view(-1, G)
        advantages = (rewards_grouped - rewards_grouped.mean(dim=1, keepdim=True)) / (rewards_grouped.std(dim=1, keepdim=True) + 1e-8)
        advantages = advantages.view(-1)

        # Policy update
        log_probs_tensor = torch.stack(all_log_probs)
        ref_log_probs_tensor = torch.stack(all_ref_log_probs)

        # Compute KL divergence
        kl_div = log_probs_tensor - ref_log_probs_tensor

        # GRPO objective
        ratio = torch.exp(log_probs_tensor - log_probs_tensor.detach())
        clipped_ratio = torch.clamp(ratio, 1 - self.config['clip_epsilon'], 1 + self.config['clip_epsilon'])

        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        kl_loss = self.config['kl_coef'] * kl_div.mean()

        total_loss = policy_loss + kl_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['max_grad_norm'])
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
            'mean_reward': rewards_tensor.mean().item(),
        }

    def generate_with_log_prob(self, prompt):
        """Generate response and compute log probability."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.policy.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.8,
                return_dict_in_generate=True,
                output_scores=True,
            )

        response_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Compute log prob
        log_prob = self.compute_sequence_log_prob(outputs.scores, response_ids)

        return response, log_prob

    def compute_sequence_log_prob(self, scores, token_ids):
        """Compute log probability of generated sequence."""
        log_probs = []
        for i, (score, token_id) in enumerate(zip(scores, token_ids)):
            probs = F.softmax(score, dim=-1)
            log_probs.append(torch.log(probs[0, token_id] + 1e-10))
        return torch.stack(log_probs).sum()


def run_grpo_training(policy, reward_model, train_data, config):
    """Full GRPO training loop."""

    # Create frozen reference model
    ref_model = copy.deepcopy(policy)
    for param in ref_model.parameters():
        param.requires_grad = False

    trainer = GRPOTrainer(policy, reward_model, ref_model, config)

    for epoch in range(config['num_epochs']):
        epoch_metrics = []

        for batch in tqdm(DataLoader(train_data, batch_size=config['batch_size'])):
            metrics = trainer.train_step(batch['prompts'])
            epoch_metrics.append(metrics)

        # Log epoch stats
        avg_reward = np.mean([m['mean_reward'] for m in epoch_metrics])
        print(f"Epoch {epoch}: Mean Reward = {avg_reward:.4f}")

        # Evaluate
        accuracy = evaluate_policy(policy, val_data)
        print(f"Validation Accuracy: {accuracy:.3f}")

        if accuracy > 0.85:
            print("Target accuracy reached, moving to Stage 4")
            break

    return policy
```

### Stage 4: DPO (Direct Preference Optimization)

```python
# src/training/stage4_dpo.py

from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

class EmailDPOTrainer:
    """
    DPO: Direct Preference Optimization (Anthropic-style).

    Advantages:
    - No reward model needed
    - More stable training
    - Direct optimization of preferences
    """

    def __init__(self, model_path, output_dir):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.output_dir = output_dir

    def prepare_dpo_dataset(self, emails_df):
        """
        Create DPO dataset from temporal email data.

        Use future emails as "chosen" (what user actually did)
        and model predictions as "rejected" (or vice versa).
        """
        dpo_data = []

        for _, row in emails_df.iterrows():
            prompt = format_email_prompt(row)

            # Chosen: What user actually did
            chosen = format_action_response(row, row['action'])

            # Rejected: Alternative action (wrong choice)
            wrong_actions = ['archive', 'delete', 'reply_later', 'reply_now']
            actual_action = row['action']
            wrong_actions = [a for a in wrong_actions if a != actual_action]

            for wrong_action in wrong_actions[:1]:  # One negative per positive
                rejected = format_action_response(row, wrong_action)

                dpo_data.append({
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected,
                })

        return Dataset.from_list(dpo_data)

    def train(self, train_dataset, eval_dataset):
        """Run DPO training."""

        # LoRA config for efficient training
        peft_config = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )

        dpo_config = DPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            learning_rate=5e-6,
            beta=0.1,  # KL penalty coefficient
            max_length=2048,
            max_prompt_length=1024,
            logging_steps=10,
            eval_steps=100,
            save_steps=200,
            warmup_ratio=0.1,
            bf16=False,  # Use fp16 on MPS
            fp16=True,
            report_to="wandb",
        )

        trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # Will create automatically
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            peft_config=peft_config,
        )

        trainer.train()
        trainer.save_model(f"{self.output_dir}/dpo_final")

        return self.model
```

### Stage 5: KTO (Kahneman-Tversky Optimization)

```python
# src/training/stage5_kto.py

from trl import KTOTrainer, KTOConfig

class EmailKTOTrainer:
    """
    KTO: Kahneman-Tversky Optimization.

    Advantages over DPO:
    - Works with unpaired preferences (just good/bad labels)
    - Based on prospect theory (losses hurt more than gains help)
    - Better for imbalanced data
    """

    def __init__(self, model_path, output_dir):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.output_dir = output_dir

    def prepare_kto_dataset(self, emails_df):
        """
        Create KTO dataset with binary labels.

        Each example is labeled as desirable or undesirable.
        """
        kto_data = []

        for _, row in emails_df.iterrows():
            prompt = format_email_prompt(row)

            # Desirable: Quick responses to important emails
            if row['action'] == 'replied' and row.get('response_time_hours', 24) < 4:
                completion = format_action_response(row, 'reply_now')
                kto_data.append({
                    'prompt': prompt,
                    'completion': completion,
                    'label': True,  # Desirable
                })

            # Desirable: Archived FYI emails
            elif row['action'] == 'archived' and not row['is_action_request']:
                completion = format_action_response(row, 'archive')
                kto_data.append({
                    'prompt': prompt,
                    'completion': completion,
                    'label': True,
                })

            # Undesirable: Ignored important emails
            if row['action'] == 'archived' and row['urgency'] > 0.7:
                completion = format_action_response(row, 'archive')
                kto_data.append({
                    'prompt': prompt,
                    'completion': completion,
                    'label': False,  # Undesirable
                })

        return Dataset.from_list(kto_data)

    def train(self, train_dataset, eval_dataset):
        """Run KTO training."""

        kto_config = KTOConfig(
            output_dir=self.output_dir,
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            learning_rate=5e-6,
            beta=0.1,
            desirable_weight=1.0,
            undesirable_weight=1.0,
            max_length=2048,
            logging_steps=10,
            fp16=True,
            report_to="wandb",
        )

        trainer = KTOTrainer(
            model=self.model,
            ref_model=None,
            args=kto_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        return self.model
```

## RLHF with Temporal Validation

### Using Future Emails as Human Feedback

```python
# src/training/temporal_rlhf.py

class TemporalRLHF:
    """
    Use temporal structure of Enron data for RLHF.

    Key insight: Future emails reveal user preferences
    - If user replied quickly -> action was important
    - If thread continued -> engagement was valuable
    - If forwarded -> delegation was appropriate
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("mps")

    def create_temporal_splits(self, emails_df):
        """
        Split emails temporally for RLHF.

        Train on emails from time T
        Validate using what happened at time T+1
        """
        emails_df['date'] = pd.to_datetime(emails_df['date'])
        emails_df = emails_df.sort_values('date')

        # Use rolling window
        windows = []
        window_size = pd.Timedelta(days=30)
        lookahead = pd.Timedelta(days=7)

        min_date = emails_df['date'].min()
        max_date = emails_df['date'].max()

        current = min_date

        while current + window_size + lookahead < max_date:
            train_mask = (emails_df['date'] >= current) & (emails_df['date'] < current + window_size)
            future_mask = (emails_df['date'] >= current + window_size) & (emails_df['date'] < current + window_size + lookahead)

            train_emails = emails_df[train_mask]
            future_emails = emails_df[future_mask]

            windows.append({
                'train': train_emails,
                'future': future_emails,
                'start': current,
                'end': current + window_size,
            })

            current += pd.Timedelta(days=7)  # Sliding window

        return windows

    def extract_feedback_from_future(self, email, future_emails, user):
        """
        Use future emails to determine if action was correct.

        Signals:
        - Reply in future -> original was important
        - Follow-up email -> task wasn't completed
        - No further communication -> was handled appropriately
        """
        msg_id = email['message_id']
        subject = email['subject']
        sender = email['from']

        # Check for replies to this email
        replies = future_emails[
            (future_emails['in_reply_to'] == msg_id) |
            (future_emails['references'].str.contains(msg_id, na=False))
        ]

        # Check for follow-ups on same subject
        subject_clean = clean_subject(subject)
        follow_ups = future_emails[
            (future_emails['subject'].apply(clean_subject) == subject_clean) &
            (future_emails['from'] == sender)
        ]

        feedback = {
            'had_reply': len(replies) > 0,
            'num_follow_ups': len(follow_ups),
            'thread_continued': len(replies) > 0 or len(follow_ups) > 0,
        }

        # Compute implicit preference score
        if email['action'] == 'replied':
            if feedback['had_reply']:
                feedback['score'] = 1.0  # Good: conversation continued
            else:
                feedback['score'] = 0.7  # OK: replied but no continuation
        elif email['action'] == 'archived':
            if feedback['num_follow_ups'] > 0:
                feedback['score'] = 0.2  # Bad: ignored but got follow-ups
            else:
                feedback['score'] = 0.8  # Good: correctly identified as low priority

        return feedback

    def run_rlhf_iteration(self, train_emails, future_emails, config):
        """
        Single RLHF iteration using future emails as feedback.
        """
        # Generate predictions for train emails
        predictions = []
        for _, email in train_emails.iterrows():
            prompt = format_email_prompt(email)
            prediction = self.generate_prediction(prompt)
            predictions.append(prediction)

        # Get feedback from future
        feedback_scores = []
        for email, pred in zip(train_emails.itertuples(), predictions):
            feedback = self.extract_feedback_from_future(
                email._asdict(),
                future_emails,
                email.user
            )

            # Compare prediction to actual outcome
            pred_action = parse_action(pred)
            actual_action = email.action

            if pred_action == actual_action:
                score = feedback.get('score', 0.5)
            else:
                score = 1.0 - feedback.get('score', 0.5)

            feedback_scores.append(score)

        # Create preference pairs from feedback
        pairs = self.create_pairs_from_feedback(
            train_emails, predictions, feedback_scores
        )

        # DPO update
        self.dpo_update(pairs)

        return np.mean(feedback_scores)
```

### Rejection Sampling Fine-Tuning (RSF)

```python
# src/training/rejection_sampling.py

class RejectionSamplingTrainer:
    """
    Rejection Sampling Fine-tuning (RSF).

    Used by Anthropic and others for final refinement.
    Generate many samples, keep the best, fine-tune on those.
    """

    def __init__(self, model, reward_model, tokenizer):
        self.model = model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = torch.device("mps")

    def generate_and_filter(self, prompts, n_samples=16, top_k=4):
        """
        Generate multiple responses, keep best by reward.
        """
        best_samples = []

        for prompt in tqdm(prompts):
            samples = []

            for _ in range(n_samples):
                response = self.generate(prompt)
                reward = self.reward_model.score(prompt, response)
                samples.append((response, reward))

            # Keep top-k by reward
            samples.sort(key=lambda x: x[1], reverse=True)
            best = samples[:top_k]

            best_samples.extend([
                {'prompt': prompt, 'response': r, 'reward': s}
                for r, s in best
            ])

        return best_samples

    def fine_tune_on_best(self, best_samples, config):
        """
        SFT on rejection-sampled best responses.
        """
        dataset = Dataset.from_list([
            {'prompt': s['prompt'], 'completion': s['response']}
            for s in best_samples
        ])

        training_args = TrainingArguments(
            output_dir=config['output_dir'],
            num_train_epochs=1,
            per_device_train_batch_size=4,
            learning_rate=1e-6,  # Very small LR for refinement
            warmup_ratio=0.1,
            fp16=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        return self.model

    def iterative_rsf(self, prompts, n_iterations=5):
        """
        Iterative RSF for continuous improvement.
        """
        for i in range(n_iterations):
            print(f"RSF Iteration {i+1}")

            # Generate and filter
            best_samples = self.generate_and_filter(prompts)

            # Fine-tune
            self.fine_tune_on_best(best_samples, {'output_dir': f'./rsf_iter_{i}'})

            # Evaluate
            accuracy = evaluate_model(self.model, val_data)
            print(f"Accuracy after iteration {i+1}: {accuracy:.3f}")

            if accuracy > 0.95:
                print("95% accuracy achieved!")
                break

        return self.model
```

## Complete Training Pipeline

```python
# src/train_full_pipeline.py

def run_full_training_pipeline(config):
    """
    Complete multi-stage training to 95% accuracy.
    """
    import wandb
    wandb.init(project="email-rl-95", config=config)

    # Load data
    print("Loading data...")
    train_df, val_df, test_df = load_and_split_data(config['data_path'])

    # Stage 1: SFT
    print("\n" + "="*50)
    print("STAGE 1: Supervised Fine-Tuning")
    print("="*50)

    sft_trainer = EmailSFTTrainer(
        base_model_path=config['base_model'],
        output_dir="./checkpoints/stage1_sft"
    )
    sft_dataset = sft_trainer.prepare_sft_dataset(train_df)
    model = sft_trainer.train(sft_dataset, val_df)

    accuracy = evaluate_model(model, test_df)
    print(f"Stage 1 Accuracy: {accuracy:.3f}")
    wandb.log({"stage1_accuracy": accuracy})

    # Stage 2: Reward Model
    print("\n" + "="*50)
    print("STAGE 2: Reward Model Training")
    print("="*50)

    preference_pairs = create_preference_pairs(train_df)
    reward_model = EmailRewardModel(config['base_model'])
    reward_model = train_reward_model(
        reward_model,
        preference_pairs,
        "./checkpoints/stage2_reward"
    )

    # Stage 3: GRPO
    print("\n" + "="*50)
    print("STAGE 3: GRPO Training")
    print("="*50)

    grpo_config = {
        'learning_rate': 1e-5,
        'group_size': 4,
        'clip_epsilon': 0.2,
        'kl_coef': 0.1,
        'max_grad_norm': 1.0,
        'num_epochs': 5,
        'batch_size': 8,
    }

    model = run_grpo_training(model, reward_model, train_df, grpo_config)

    accuracy = evaluate_model(model, test_df)
    print(f"Stage 3 Accuracy: {accuracy:.3f}")
    wandb.log({"stage3_accuracy": accuracy})

    # Stage 4: DPO
    print("\n" + "="*50)
    print("STAGE 4: DPO Training")
    print("="*50)

    dpo_trainer = EmailDPOTrainer(
        model_path="./checkpoints/stage3_grpo",
        output_dir="./checkpoints/stage4_dpo"
    )
    dpo_dataset = dpo_trainer.prepare_dpo_dataset(train_df)
    model = dpo_trainer.train(dpo_dataset, val_df)

    accuracy = evaluate_model(model, test_df)
    print(f"Stage 4 Accuracy: {accuracy:.3f}")
    wandb.log({"stage4_accuracy": accuracy})

    # Stage 5: Temporal RLHF
    print("\n" + "="*50)
    print("STAGE 5: Temporal RLHF")
    print("="*50)

    temporal_trainer = TemporalRLHF(model, tokenizer)
    windows = temporal_trainer.create_temporal_splits(train_df)

    for i, window in enumerate(windows):
        score = temporal_trainer.run_rlhf_iteration(
            window['train'],
            window['future'],
            config
        )
        print(f"Window {i}: Mean feedback score = {score:.3f}")

    accuracy = evaluate_model(model, test_df)
    print(f"Stage 5 Accuracy: {accuracy:.3f}")
    wandb.log({"stage5_accuracy": accuracy})

    # Stage 6: Final RSF Refinement
    print("\n" + "="*50)
    print("STAGE 6: Rejection Sampling Refinement")
    print("="*50)

    rsf_trainer = RejectionSamplingTrainer(model, reward_model, tokenizer)
    prompts = [format_email_prompt(row) for _, row in train_df.iterrows()]
    model = rsf_trainer.iterative_rsf(prompts, n_iterations=5)

    # Final evaluation
    final_accuracy = evaluate_model(model, test_df)
    print(f"\n" + "="*50)
    print(f"FINAL ACCURACY: {final_accuracy:.3f}")
    print("="*50)
    wandb.log({"final_accuracy": final_accuracy})

    # Save final model
    model.save_pretrained("./checkpoints/final_model")

    return model, final_accuracy


if __name__ == "__main__":
    config = {
        'base_model': 'Qwen/Qwen2.5-72B-Instruct',
        'data_path': './data/features.parquet',
        # ... other config
    }

    model, accuracy = run_full_training_pipeline(config)
```

## Techniques for 95% Accuracy

### 1. Data Augmentation

```python
def augment_email_data(emails_df):
    """
    Augment training data for better generalization.
    """
    augmented = []

    for _, row in emails_df.iterrows():
        # Original
        augmented.append(row)

        # Paraphrase subject
        paraphrased = row.copy()
        paraphrased['subject'] = paraphrase(row['subject'])
        augmented.append(paraphrased)

        # Swap similar senders
        if row['people_score'] > 0.5:
            swapped = row.copy()
            swapped['from'] = get_similar_sender(row['from'])
            augmented.append(swapped)

    return pd.DataFrame(augmented)
```

### 2. Ensemble Methods

```python
class EnsembleEmailAgent:
    """Ensemble of models for higher accuracy."""

    def __init__(self, model_paths):
        self.models = [load_model(p) for p in model_paths]

    def predict(self, email):
        predictions = [m.predict(email) for m in self.models]

        # Majority vote for action
        actions = [p['action'] for p in predictions]
        final_action = max(set(actions), key=actions.count)

        # Average priority
        final_priority = np.mean([p['priority'] for p in predictions])

        return {'action': final_action, 'priority': final_priority}
```

### 3. Confidence Calibration

```python
def calibrate_predictions(model, val_data):
    """
    Calibrate model confidence using temperature scaling.
    """
    from sklearn.calibration import CalibratedClassifierCV

    # Get predictions and true labels
    preds = []
    labels = []

    for email in val_data:
        logits = model.get_logits(email)
        preds.append(logits)
        labels.append(email['action'])

    # Find optimal temperature
    best_temp = 1.0
    best_ece = float('inf')

    for temp in np.arange(0.5, 3.0, 0.1):
        calibrated = softmax(np.array(preds) / temp, axis=1)
        ece = expected_calibration_error(calibrated, labels)
        if ece < best_ece:
            best_ece = ece
            best_temp = temp

    return best_temp
```

## Running the Full Pipeline

```bash
# 1. Setup environment
conda activate email-rl
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Allow full memory usage

# 2. Download models
python scripts/download_models.py

# 3. Prepare data
python scripts/prepare_data.py

# 4. Run full training pipeline
python src/train_full_pipeline.py \
    --base_model Qwen/Qwen2.5-72B-Instruct \
    --data_path ./data/features.parquet \
    --output_dir ./checkpoints \
    --target_accuracy 0.95

# 5. Evaluate final model
python src/evaluate.py \
    --model_path ./checkpoints/final_model \
    --test_data ./data/test.parquet
```

## Expected Progression

| Stage | Method | Expected Accuracy |
|-------|--------|------------------|
| 1 | SFT | 65-70% |
| 2 | Reward Model | - |
| 3 | GRPO | 80-85% |
| 4 | DPO | 88-90% |
| 5 | Temporal RLHF | 92-94% |
| 6 | RSF Refinement | 95%+ |

## Monitoring & Debugging

```python
# Use wandb for experiment tracking
import wandb

wandb.init(project="email-rl-95")

# Log training metrics
wandb.log({
    "accuracy": accuracy,
    "loss": loss,
    "reward": mean_reward,
    "kl_divergence": kl_div,
})

# Log model predictions for debugging
wandb.log({
    "predictions": wandb.Table(
        columns=["email", "predicted", "actual", "correct"],
        data=prediction_samples
    )
})
```
