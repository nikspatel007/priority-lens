# Enron Email Dataset Setup

## About the Dataset

The Enron email dataset contains approximately 500,000 emails from 150 users, primarily senior management of Enron Corporation. It was made public during the Federal Energy Regulatory Commission investigation.

**Why Enron?**
- Real-world corporate email data
- Contains full email threads with replies
- Shows actual human decision-making patterns
- Includes diverse email types: tasks, meetings, reports, personal, urgent matters

## Download Options

### Option 1: CMU Mirror (Recommended)

```bash
# Download the May 7, 2015 version (cleaned, ~1.7GB compressed)
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz

# Extract
tar -xzf enron_mail_20150507.tar.gz
```

### Option 2: Kaggle Dataset

```bash
# Install kaggle CLI
pip install kaggle

# Download (requires Kaggle API credentials)
kaggle datasets download -d wcukierski/enron-email-dataset

# Extract
unzip enron-email-dataset.zip
```

### Option 3: Stanford SNAP

```bash
# Alternative cleaned version
wget https://snap.stanford.edu/data/email-Enron.txt.gz
gunzip email-Enron.txt.gz
```

## Dataset Structure

After extraction, the dataset is organized by user:

```
maildir/
├── allen-p/           # Philip Allen's mailbox
│   ├── inbox/
│   ├── sent/
│   ├── sent_items/
│   ├── deleted_items/
│   └── ...
├── arnold-j/          # John Arnold's mailbox
├── bass-e/            # Eric Bass's mailbox
└── ...                # ~150 users total
```

Each email is a text file with headers and body:

```
Message-ID: <...>
Date: Mon, 14 May 2001 16:39:00 -0700
From: phillip.allen@enron.com
To: tim.belden@enron.com
Subject: Re: Weekly Report
...

<email body>
```

## Data Preprocessing

### Email Parsing Script

```python
import os
import email
from email import policy
from pathlib import Path
import json

def parse_email(file_path):
    """Parse a single email file into structured data."""
    with open(file_path, 'r', encoding='latin-1') as f:
        msg = email.message_from_file(f, policy=policy.default)

    return {
        'message_id': msg.get('Message-ID', ''),
        'date': msg.get('Date', ''),
        'from': msg.get('From', ''),
        'to': msg.get('To', ''),
        'cc': msg.get('Cc', ''),
        'subject': msg.get('Subject', ''),
        'body': msg.get_body(preferencelist=('plain',)).get_content() if msg.get_body() else '',
        'in_reply_to': msg.get('In-Reply-To', ''),
        'references': msg.get('References', ''),
    }

def process_maildir(maildir_path, output_path):
    """Process entire maildir into JSON format."""
    emails = []

    for user_dir in Path(maildir_path).iterdir():
        if not user_dir.is_dir():
            continue

        user = user_dir.name

        for email_file in user_dir.rglob('*'):
            if email_file.is_file() and not email_file.name.startswith('.'):
                try:
                    email_data = parse_email(email_file)
                    email_data['user'] = user
                    email_data['folder'] = email_file.parent.name
                    email_data['file_path'] = str(email_file)
                    emails.append(email_data)
                except Exception as e:
                    print(f"Error parsing {email_file}: {e}")

    with open(output_path, 'w') as f:
        json.dump(emails, f, indent=2)

    return emails
```

### Key Preprocessing Steps

1. **Thread Reconstruction**
   - Match emails by `In-Reply-To` and `References` headers
   - Build conversation trees

2. **User Action Labeling**
   - Identify which emails received replies (found in `sent/` folder)
   - Track time-to-response
   - Note which emails were forwarded, deleted, or archived

3. **Metadata Extraction**
   - Parse dates into timestamps
   - Normalize email addresses
   - Extract organizational hierarchy from addresses

```python
def label_user_actions(emails_by_user):
    """Label what action the user took on each email."""

    for user, emails in emails_by_user.items():
        sent_emails = [e for e in emails if 'sent' in e['folder'].lower()]
        inbox_emails = [e for e in emails if 'inbox' in e['folder'].lower()]

        # Build reply mapping
        sent_reply_to = {e['in_reply_to']: e for e in sent_emails if e['in_reply_to']}

        for email in inbox_emails:
            msg_id = email['message_id']

            if msg_id in sent_reply_to:
                email['action'] = 'replied'
                email['reply'] = sent_reply_to[msg_id]
            elif email['folder'] == 'deleted_items':
                email['action'] = 'deleted'
            else:
                email['action'] = 'no_action'  # Read but no reply
```

## Data Statistics

| Metric | Value |
|--------|-------|
| Total Emails | ~500,000 |
| Unique Users | 150 |
| Date Range | 1998-2002 |
| Avg Emails per User | ~3,500 |

## Storage Requirements

- Raw dataset: ~1.7GB compressed, ~6GB extracted
- Processed JSON: ~2-3GB
- Training-ready format: ~500MB-1GB

## Next Steps

After downloading and preprocessing:
1. Review [System Architecture](./02-architecture.md) for RL design
2. See [Feature Extraction](./03-features.md) for scoring dimensions
3. Follow [Training Guide](./04-training.md) to train the model
