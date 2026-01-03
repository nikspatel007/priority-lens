#!/bin/bash
# Download and extract Enron email dataset

set -e

DATA_DIR="./data"
mkdir -p "$DATA_DIR"

echo "========================================"
echo "Downloading Enron Email Dataset"
echo "========================================"

# CMU mirror (most reliable)
URL="https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
FILENAME="enron_mail_20150507.tar.gz"

cd "$DATA_DIR"

if [ -f "$FILENAME" ]; then
    echo "Archive already exists, skipping download..."
else
    echo "Downloading from CMU (~1.7GB)..."
    curl -L -o "$FILENAME" "$URL"
fi

if [ -d "maildir" ]; then
    echo "Maildir already extracted, skipping..."
else
    echo "Extracting archive..."
    tar -xzf "$FILENAME"
fi

echo ""
echo "========================================"
echo "Download complete!"
echo "========================================"
echo "Dataset location: $DATA_DIR/maildir"
echo "Users: $(ls maildir | wc -l)"
echo ""
echo "Next steps:"
echo "  1. python src/preprocess.py"
echo "  2. python src/train.py"
