#!/usr/bin/env python3
"""Gate 6: Feature output validation.

Runs feature extraction on 100 sample emails and validates:
- No NaN/null values
- Correct dimensions
- Reasonable value ranges
"""

import random
import sys
from datetime import datetime, timedelta
from typing import Any

import numpy as np

sys.path.insert(0, str(__file__).replace('/scripts/validate_features.py', ''))

from src.features.combined import (
    extract_combined_features,
    extract_batch,
    build_feature_matrix,
    FEATURE_DIMS,
)


# Sample data for generating diverse emails
SENDERS = [
    ('john.smith@enron.com', 'John Smith', 'VP Operations'),
    ('jane.doe@enron.com', 'Jane Doe', 'Director'),
    ('bob.jones@enron.com', 'Bob Jones', 'Manager'),
    ('alice.williams@enron.com', 'Alice Williams', ''),
    ('noreply@alerts.enron.com', 'System Alerts', 'Automated'),
    ('client@external.com', 'External Client', 'Partner'),
    ('ceo@enron.com', 'CEO Office', 'CEO'),
    ('newsletter@marketing.com', 'Marketing Newsletter', ''),
]

SUBJECTS = [
    'URGENT: Project deadline tomorrow',
    'Meeting request for Friday',
    'FYI: Updated documents attached',
    'Action Required: Review and approve',
    'Weekly status update',
    'Question about the budget',
    'RE: Follow up on our conversation',
    'Important: Policy change notification',
    'Invitation: Team lunch',
    'Critical issue needs resolution',
    'Please review by EOD',
    'Quick question',
    'Today\'s agenda',
    'Monthly report',
    'Feedback needed',
]

BODIES = [
    'Please review the attached document and provide feedback by EOD.',
    'Hi team, just a quick update on the project status.',
    'Can you send me the latest numbers? I need them for the presentation.',
    'FYI - sharing this for your reference. No action needed.',
    'We have a critical issue that needs immediate attention.',
    'Thanks for your help on this. Great work!',
    'I need you to complete this task before the meeting tomorrow.',
    'Let me know your availability for a quick call.',
    'The deadline has been moved to Friday. Please plan accordingly.',
    'Here\'s the weekly progress report as requested.',
    'Could you take a look at this and let me know your thoughts?',
    'This is blocking our ability to proceed. Urgent response needed.',
    'Just checking in on the status. Any updates?',
    'Please confirm receipt and your availability.',
    'Attached is the proposal for your review.',
]


def generate_sample_emails(n: int = 100) -> list[dict]:
    """Generate n diverse sample emails."""
    emails = []
    base_date = datetime.now() - timedelta(days=30)

    for i in range(n):
        sender = random.choice(SENDERS)
        subject = random.choice(SUBJECTS)
        body = random.choice(BODIES)

        # Sometimes combine multiple body fragments
        if random.random() > 0.5:
            body += '\n\n' + random.choice(BODIES)

        # Add urgency markers sometimes
        if random.random() > 0.7:
            body += '\n\nThis is urgent. Please respond ASAP.'

        # Add deadline sometimes
        if random.random() > 0.6:
            deadlines = ['by tomorrow', 'by EOD', 'by Friday', 'by end of week']
            body += f'\n\nPlease complete {random.choice(deadlines)}.'

        # Generate date
        days_ago = random.randint(0, 30)
        hours = random.randint(0, 23)
        email_date = base_date + timedelta(days=days_ago, hours=hours)

        email = {
            'from': sender[0],
            'to': 'user@enron.com',
            'cc': 'team@enron.com' if random.random() > 0.7 else '',
            'x_from': f'{sender[1]}, {sender[2]}' if sender[2] else sender[1],
            'x_to': 'User',
            'subject': subject,
            'body': body,
            'date': email_date.isoformat(),
        }
        emails.append(email)

    return emails


def validate_no_nan_null(matrix: np.ndarray) -> tuple[bool, list[str]]:
    """Check for NaN or null values in feature matrix."""
    errors = []

    if np.isnan(matrix).any():
        nan_count = np.isnan(matrix).sum()
        nan_locations = np.argwhere(np.isnan(matrix))
        errors.append(f'Found {nan_count} NaN values at: {nan_locations[:5].tolist()}...')

    if np.isinf(matrix).any():
        inf_count = np.isinf(matrix).sum()
        errors.append(f'Found {inf_count} Inf values')

    # Check for None/null (should be numeric)
    if matrix.dtype == object:
        errors.append('Matrix has object dtype, expected numeric')

    return len(errors) == 0, errors


def validate_dimensions(matrix: np.ndarray, n_emails: int) -> tuple[bool, list[str]]:
    """Check feature matrix dimensions."""
    errors = []

    expected_dims = FEATURE_DIMS['total_base']
    if matrix.shape != (n_emails, expected_dims):
        errors.append(f'Expected shape ({n_emails}, {expected_dims}), got {matrix.shape}')

    return len(errors) == 0, errors


def validate_value_ranges(matrix: np.ndarray) -> tuple[bool, list[str]]:
    """Check that values are in reasonable ranges."""
    errors = []

    # Most features should be in [0, 1] range (normalized)
    # Allow some values slightly outside due to normalization quirks
    min_val = matrix.min()
    max_val = matrix.max()

    # Binary and normalized features should be in [-1, 1] or [0, 1]
    # Give buffer for scores that might combine
    if min_val < -2:
        errors.append(f'Unexpectedly low minimum value: {min_val}')
    if max_val > 2:
        errors.append(f'Unexpectedly high maximum value: {max_val}')

    # Check for reasonable variance (features should have some variation)
    per_feature_std = matrix.std(axis=0)
    zero_variance_count = (per_feature_std == 0).sum()
    if zero_variance_count > FEATURE_DIMS['total_base'] * 0.5:
        errors.append(f'{zero_variance_count} features have zero variance (too many)')

    # Check that normalized features are actually normalized
    # (most values should be between 0 and 1)
    in_range = ((matrix >= 0) & (matrix <= 1)).mean()
    if in_range < 0.8:
        errors.append(f'Only {in_range:.1%} of values are in [0,1] range')

    return len(errors) == 0, errors


def run_validation():
    """Run full validation pipeline."""
    print('=' * 60)
    print('GATE 6: Feature Output Validation')
    print('=' * 60)
    print()

    # Generate sample emails
    print('Generating 100 sample emails...')
    emails = generate_sample_emails(100)
    print(f'  Generated {len(emails)} emails')
    print()

    # Extract features
    print('Extracting features...')
    matrix = build_feature_matrix(emails)
    print(f'  Matrix shape: {matrix.shape}')
    print(f'  Matrix dtype: {matrix.dtype}')
    print()

    all_passed = True

    # Validate 1: No NaN/null
    print('Validation 1: No NaN/null values')
    passed, errors = validate_no_nan_null(matrix)
    if passed:
        print('  PASS: No NaN/null values found')
    else:
        print('  FAIL:')
        for e in errors:
            print(f'    - {e}')
        all_passed = False
    print()

    # Validate 2: Correct dimensions
    print('Validation 2: Correct dimensions')
    passed, errors = validate_dimensions(matrix, len(emails))
    if passed:
        print(f'  PASS: Shape is ({len(emails)}, {FEATURE_DIMS["total_base"]})')
    else:
        print('  FAIL:')
        for e in errors:
            print(f'    - {e}')
        all_passed = False
    print()

    # Validate 3: Reasonable value ranges
    print('Validation 3: Reasonable value ranges')
    passed, errors = validate_value_ranges(matrix)
    if passed:
        print(f'  PASS: Values in expected ranges')
        print(f'    - Min: {matrix.min():.4f}')
        print(f'    - Max: {matrix.max():.4f}')
        print(f'    - Mean: {matrix.mean():.4f}')
        print(f'    - Std: {matrix.std():.4f}')
    else:
        print('  FAIL:')
        for e in errors:
            print(f'    - {e}')
        all_passed = False
    print()

    # Feature distribution summary
    print('Feature dimension breakdown:')
    start = 0
    for name in ['project', 'topic', 'task', 'people', 'temporal', 'scores']:
        dim = FEATURE_DIMS[name]
        end = start + dim
        segment = matrix[:, start:end]
        print(f'  {name}: dims {start}-{end-1} ({dim}), '
              f'mean={segment.mean():.3f}, std={segment.std():.3f}')
        start = end
    print()

    # Final result
    print('=' * 60)
    if all_passed:
        print('RESULT: ALL VALIDATIONS PASSED')
        print('Gate 6 complete: Feature output validated successfully.')
        return 0
    else:
        print('RESULT: VALIDATION FAILED')
        return 1


if __name__ == '__main__':
    sys.exit(run_validation())
