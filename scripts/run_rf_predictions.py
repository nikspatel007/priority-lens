#!/usr/bin/env python3
"""Run RF model predictions on full email dataset.

Loads the trained RandomForest model and runs predictions on all emails
with available features, storing results in email_predictions table.

Usage:
    uv run python scripts/run_rf_predictions.py
    uv run python scripts/run_rf_predictions.py --batch-size 1000 --dry-run
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

# Database connection
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/rl_emails"
)


def load_model(model_path: str = "models/action_classifier_v1.pkl") -> dict:
    """Load the trained RF model and encoders."""
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data


def load_binary_model(model_path: str = "models/delete_vs_keep_v1.pkl") -> dict:
    """Load the binary delete/keep classifier."""
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data


def get_emails_with_features(conn, limit: int = None) -> pd.DataFrame:
    """Load all emails with features for prediction."""
    query = """
    SELECT
        e.id as email_id,
        e.subject,
        e.word_count,
        e.has_attachments,
        ef.is_service_email,
        ef.service_type,
        llm.requires_response,
        llm.overall_urgency,
        llm.topic_category
    FROM emails e
    JOIN email_features ef ON e.id = ef.email_id
    LEFT JOIN email_llm_features llm ON e.message_id = llm.email_id
    """
    if limit:
        query += f" LIMIT {limit}"

    return pd.read_sql(query, conn)


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features from email data."""
    # Subject-based features
    df['subject_len'] = df['subject'].fillna('').str.len()
    df['subject_words'] = df['subject'].fillna('').str.split().str.len().fillna(0).astype(int)
    df['has_re'] = df['subject'].fillna('').str.lower().str.contains(r'^re:', regex=True).astype(int)
    df['has_fwd'] = df['subject'].fillna('').str.lower().str.contains(r'^fw[d]?:', regex=True).astype(int)
    df['is_noreply'] = 0  # Would need from email, set to 0 for now

    # Ensure boolean fields are int
    df['has_attachments'] = df['has_attachments'].fillna(False).astype(int)
    df['requires_response'] = df['requires_response'].fillna(False).astype(int)
    df['is_service_email'] = df['is_service_email'].fillna(False).astype(int)

    # Fill missing word_count
    df['word_count'] = df['word_count'].fillna(0).astype(int)

    return df


def encode_features(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Encode categorical features using trained label encoders."""
    # Handle overall_urgency encoding
    df['overall_urgency_str'] = df['overall_urgency'].fillna('unknown').astype(str)
    # Round to match trained encoder classes
    def round_urgency(val):
        try:
            v = float(val)
            # Round to nearest 0.05 that's in our encoder
            rounded = round(v * 20) / 20
            return f"{rounded:.1f}" if rounded >= 0.1 else "0.1"
        except:
            return 'unknown'
    df['overall_urgency_str'] = df['overall_urgency_str'].apply(round_urgency)

    # Ensure values are in encoder's classes
    urgency_classes = set(encoders['overall_urgency'].classes_)
    df['overall_urgency_str'] = df['overall_urgency_str'].apply(
        lambda x: x if x in urgency_classes else 'unknown'
    )
    df['overall_urgency_enc'] = encoders['overall_urgency'].transform(df['overall_urgency_str'])

    # Handle topic_category encoding
    df['topic_category'] = df['topic_category'].fillna('unknown').str.lower()
    topic_classes = set(encoders['topic_category'].classes_)
    df['topic_category'] = df['topic_category'].apply(
        lambda x: x if x in topic_classes else 'unknown'
    )
    df['topic_category_enc'] = encoders['topic_category'].transform(df['topic_category'])

    # Handle service_type encoding
    df['service_type'] = df['service_type'].fillna('unknown').str.lower()
    service_classes = set(encoders['service_type'].classes_)
    df['service_type'] = df['service_type'].apply(
        lambda x: x if x in service_classes else 'unknown'
    )
    df['service_type_enc'] = encoders['service_type'].transform(df['service_type'])

    return df


def run_predictions(df: pd.DataFrame, model_data: dict) -> pd.DataFrame:
    """Run RF model predictions."""
    model = model_data['model']
    feature_cols = model_data['feature_cols']

    # Prepare feature matrix
    X = df[feature_cols].values

    # Get predictions and probabilities
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Get confidence (max probability)
    confidence = probabilities.max(axis=1)

    df['predicted_action'] = predictions
    df['confidence'] = confidence

    return df


def run_binary_predictions(df: pd.DataFrame, model_data: dict) -> pd.DataFrame:
    """Run binary delete/keep classifier."""
    model = model_data['model']
    feature_cols = model_data['feature_cols']

    # Check if we have all required features
    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) < len(feature_cols):
        print(f"Warning: Missing columns for binary model: {set(feature_cols) - set(available_cols)}")
        df['binary_action'] = None
        df['binary_confidence'] = None
        return df

    X = df[feature_cols].values
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    confidence = probabilities.max(axis=1)

    df['binary_action'] = predictions
    df['binary_confidence'] = confidence

    return df


def save_predictions(conn, df: pd.DataFrame, model_version: str = "rf_v1", dry_run: bool = False):
    """Save predictions to email_predictions table."""
    if dry_run:
        print(f"[DRY RUN] Would save {len(df)} predictions with model_version={model_version}")
        return

    cursor = conn.cursor()

    # Delete existing predictions for this model version
    cursor.execute(
        "DELETE FROM email_predictions WHERE model_version = %s",
        (model_version,)
    )
    deleted = cursor.rowcount
    print(f"Deleted {deleted} existing predictions for model_version={model_version}")

    # Prepare data for insertion
    values = [
        (
            row['email_id'],
            row['predicted_action'],
            row['confidence'],
            model_version,
            row.get('binary_action'),
            row.get('binary_confidence')
        )
        for _, row in df.iterrows()
    ]

    # Insert in batches
    insert_query = """
    INSERT INTO email_predictions
    (email_id, predicted_action, confidence, model_version, binary_action, binary_confidence)
    VALUES %s
    ON CONFLICT (email_id, model_version) DO UPDATE SET
        predicted_action = EXCLUDED.predicted_action,
        confidence = EXCLUDED.confidence,
        binary_action = EXCLUDED.binary_action,
        binary_confidence = EXCLUDED.binary_confidence,
        created_at = CURRENT_TIMESTAMP
    """

    execute_values(cursor, insert_query, values, page_size=1000)
    conn.commit()
    print(f"Saved {len(values)} predictions")


def main():
    parser = argparse.ArgumentParser(description="Run RF predictions on full dataset")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size for processing")
    parser.add_argument("--dry-run", action="store_true", help="Don't save to database")
    parser.add_argument("--limit", type=int, help="Limit number of emails to process")
    parser.add_argument("--model", default="models/action_classifier_v1.pkl", help="Path to model file")
    args = parser.parse_args()

    print("Loading RF model...")
    model_data = load_model(args.model)
    print(f"Model loaded: {type(model_data['model']).__name__}")
    print(f"Feature columns: {model_data['feature_cols']}")

    # Try loading binary model
    binary_model_data = None
    try:
        binary_model_data = load_binary_model()
        print("Binary model loaded")
    except Exception as e:
        print(f"Binary model not available: {e}")

    print("\nConnecting to database...")
    conn = psycopg2.connect(DB_URL)

    print("Loading emails with features...")
    df = get_emails_with_features(conn, limit=args.limit)
    print(f"Loaded {len(df)} emails")

    if len(df) == 0:
        print("No emails found!")
        return

    print("\nComputing derived features...")
    df = compute_derived_features(df)

    print("Encoding categorical features...")
    df = encode_features(df, model_data['encoders'])

    print("\nRunning RF predictions...")
    df = run_predictions(df, model_data)

    # Print action distribution
    print("\nPrediction distribution:")
    print(df['predicted_action'].value_counts())

    # Run binary predictions if model available
    if binary_model_data:
        print("\nRunning binary predictions...")
        df = run_binary_predictions(df, binary_model_data)

    print(f"\nSaving predictions (dry_run={args.dry_run})...")
    save_predictions(conn, df, model_version="rf_v1", dry_run=args.dry_run)

    conn.close()
    print("\nDone!")

    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total emails processed: {len(df)}")
    print(f"Mean confidence: {df['confidence'].mean():.3f}")
    print(f"Median confidence: {df['confidence'].median():.3f}")


if __name__ == "__main__":
    main()
