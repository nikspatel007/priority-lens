#!/usr/bin/env python3
"""Create train/val/test splits for email dataset.

Splits data by user to prevent leakage - all emails from a single user
are assigned to the same split. Within each split, emails are sorted
chronologically.
"""

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_date(date_str: str) -> datetime:
    """Parse email date string to datetime for sorting."""
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S",
        "%d %b %Y %H:%M:%S %z",
        "%d %b %Y %H:%M:%S",
    ]
    for fmt in formats:
        try:
            # Handle -0700 (PDT) style timezone
            cleaned = date_str.replace(" (PDT)", "").replace(" (PST)", "")
            cleaned = cleaned.replace(" (EDT)", "").replace(" (EST)", "")
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    # Fallback: return epoch if parsing fails
    return datetime(1970, 1, 1)


def create_splits(
    input_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """Create train/val/test splits by user.

    Args:
        input_path: Path to labeled emails JSON file
        output_dir: Directory for output files
        train_ratio: Fraction of users for training
        val_ratio: Fraction of users for validation
        test_ratio: Fraction of users for testing
        seed: Random seed for reproducibility

    Returns:
        Statistics dictionary
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # Load data
    print(f"Loading data from {input_path}...")
    with open(input_path) as f:
        emails = json.load(f)
    print(f"Loaded {len(emails)} emails")

    # Group emails by user
    by_user = defaultdict(list)
    for email in emails:
        by_user[email["user"]].append(email)

    users = sorted(by_user.keys())
    print(f"Found {len(users)} unique users")

    # Shuffle users deterministically
    random.seed(seed)
    random.shuffle(users)

    # Split users
    n_users = len(users)
    n_train = int(n_users * train_ratio)
    n_val = int(n_users * val_ratio)

    train_users = set(users[:n_train])
    val_users = set(users[n_train : n_train + n_val])
    test_users = set(users[n_train + n_val :])

    print(f"Split: {len(train_users)} train / {len(val_users)} val / {len(test_users)} test users")

    # Collect emails for each split
    def get_split_emails(user_set):
        emails_list = []
        for user in user_set:
            emails_list.extend(by_user[user])
        # Sort by date within split
        emails_list.sort(key=lambda e: parse_date(e.get("date", "")))
        return emails_list

    train_emails = get_split_emails(train_users)
    val_emails = get_split_emails(val_users)
    test_emails = get_split_emails(test_users)

    print(f"Emails: {len(train_emails)} train / {len(val_emails)} val / {len(test_emails)} test")

    # Write output files
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train_emails), ("val", val_emails), ("test", test_emails)]:
        output_path = output_dir / f"{name}.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Wrote {output_path}")

    # Write split metadata
    meta = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "n_users": {"train": len(train_users), "val": len(val_users), "test": len(test_users)},
        "n_emails": {"train": len(train_emails), "val": len(val_emails), "test": len(test_emails)},
        "train_users": sorted(train_users),
        "val_users": sorted(val_users),
        "test_users": sorted(test_users),
    }
    meta_path = output_dir / "split_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")

    return meta


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("input", type=Path, help="Path to labeled emails JSON")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data"),
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    create_splits(
        args.input,
        args.output,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    return 0


if __name__ == "__main__":
    exit(main())
