#!/bin/bash
# Full import pipeline for both Enron and Gmail datasets
#
# This script:
#   1. Ensures databases are running
#   2. Imports schema to both databases
#   3. Imports all data files
#
# Usage:
#   ./scripts/import_all.sh           # Import both datasets
#   ./scripts/import_all.sh enron     # Import only Enron
#   ./scripts/import_all.sh gmail     # Import only Gmail
#
# Prerequisites:
#   - Enron data: data/train.json, data/val.json, data/test.json
#   - Gmail data: data/gmail_labeled.json

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
SCHEMA_FILE="$PROJECT_DIR/db/schema.surql"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

check_db_running() {
    local port=$1
    local db_name=$2

    if ! curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
        echo -e "${RED}Error: $db_name database not running on port $port${NC}"
        echo "Start it first: ./scripts/start_db.sh $db_name"
        return 1
    fi
    return 0
}

import_schema() {
    local port=$1
    local db_name=$2

    echo "Importing schema to $db_name..."
    surreal import \
        --endpoint "http://127.0.0.1:$port" \
        --user root \
        --pass root \
        --ns rl_emails \
        --db "$db_name" \
        "$SCHEMA_FILE"
    echo -e "${GREEN}Schema imported to $db_name${NC}"
}

import_enron() {
    local port=8000

    echo ""
    echo "=== Importing Enron Dataset ==="

    # Check if database is running
    if ! check_db_running $port "enron"; then
        return 1
    fi

    # Check for data files
    local train_file="$DATA_DIR/train.json"
    local val_file="$DATA_DIR/val.json"
    local test_file="$DATA_DIR/test.json"

    local missing_files=()
    [ ! -f "$train_file" ] && missing_files+=("train.json")
    [ ! -f "$val_file" ] && missing_files+=("val.json")
    [ ! -f "$test_file" ] && missing_files+=("test.json")

    if [ ${#missing_files[@]} -gt 0 ]; then
        echo -e "${RED}Error: Missing Enron data files: ${missing_files[*]}${NC}"
        echo "Expected files in: $DATA_DIR"
        return 1
    fi

    # Import schema
    import_schema $port "enron"

    # Import data
    echo "Importing Enron emails..."
    python -m db.import_data enron \
        "$train_file" "$val_file" "$test_file" \
        --url "ws://localhost:$port/rpc" \
        --skip-schema

    echo -e "${GREEN}Enron import complete${NC}"
}

import_gmail() {
    local port=8001

    echo ""
    echo "=== Importing Gmail Dataset ==="

    # Check if database is running
    if ! check_db_running $port "gmail"; then
        return 1
    fi

    # Check for data file
    local gmail_file="$DATA_DIR/gmail_labeled.json"

    if [ ! -f "$gmail_file" ]; then
        echo -e "${RED}Error: Missing Gmail data file: gmail_labeled.json${NC}"
        echo "Expected file: $gmail_file"
        return 1
    fi

    # Import schema
    import_schema $port "gmail"

    # Import data
    echo "Importing Gmail emails..."
    python -m db.import_data gmail \
        "$gmail_file" \
        --url "ws://localhost:$port/rpc" \
        --skip-schema

    echo -e "${GREEN}Gmail import complete${NC}"
}

# Parse arguments
DB_TARGET="${1:-all}"

# Change to project directory for Python module resolution
cd "$PROJECT_DIR"

case "$DB_TARGET" in
    enron)
        import_enron
        ;;
    gmail)
        import_gmail
        ;;
    all)
        import_enron
        import_gmail
        ;;
    *)
        echo "Error: Unknown target '$DB_TARGET'. Use 'enron', 'gmail', or 'all'."
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}=== Import Complete ===${NC}"
echo ""
echo "Query your data:"
echo "  ./scripts/query_db.py enron 'SELECT count() FROM emails GROUP ALL'"
echo "  ./scripts/query_db.py gmail 'SELECT count() FROM emails GROUP ALL'"
