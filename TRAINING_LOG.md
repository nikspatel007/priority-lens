# Training Log

## Run: 2026-01-04 (Enron Full Pipeline)

### Data Pipeline
- **Source**: Enron email dataset (CMU mirror)
- **Raw emails**: 517,401
- **Labeled (actionable)**: 199,336
- **Splits**:
  - Train: 140,000 (112 users)
  - Val: 32,000 (19 users)
  - Test: 27,169 (19 users)

### Action Distribution (5-class)
| Action | Train | Val | Test |
|--------|-------|-----|------|
| archive | 63% | 35% | 37% |
| reply_later | 25% | 33% | 33% |
| delete | 9% | 23% | 23% |
| forward | 3% | 9% | 7% |
| reply_now | 0% | 0% | 0% |

**Note**: `reply_now` is 0% because response-time matching failed for most replies
(Enron has sparse In-Reply-To/References headers).

### Results
| Stage | Val Accuracy | Test Accuracy | Notes |
|-------|--------------|---------------|-------|
| Baseline | - | 1.59% | Random init, predicts reply_now |
| SFT | 96.8% | 37.48% | Mode collapse → archive |

### Artifacts (gitignored, local only)
```
data/
├── maildir/           # Raw Enron emails (517K files)
├── emails.json        # Parsed emails (517K)
├── emails_labeled.json # Labeled (199K)
├── train.json         # Training split (140K)
├── val.json           # Validation split (32K)
├── test.json          # Test split (27K)
└── split_metadata.json

checkpoints/
├── stage_1.pt         # SFT final checkpoint
├── best_sft.pt        # Best validation checkpoint
└── sft_epoch_*.pt     # Per-epoch checkpoints
```

### Issues Identified
1. **Class imbalance**: archive dominates training (63%)
2. **Distribution shift**: train vs test distributions differ significantly
3. **Mode collapse**: model learns to predict majority class
4. **No reply_now**: response-time matching failed

### Next Steps
- [ ] Add class weighting to SFT loss
- [ ] Try focal loss for hard examples
- [ ] Run GRPO with reward shaping
- [ ] Try Gmail data (better threading headers)
