#!/bin/bash
# Start SurrealDB for email RL training
#
# Usage:
#   ./scripts/start_surreal.sh [database]
#
# Examples:
#   ./scripts/start_surreal.sh          # Start Enron database (default)
#   ./scripts/start_surreal.sh enron    # Start Enron database
#   ./scripts/start_surreal.sh gmail    # Start Gmail database

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"

# Default to enron database
DB_NAME="${1:-enron}"

# Validate database name
case "$DB_NAME" in
    enron|gmail)
        ;;
    *)
        echo "Error: Unknown database '$DB_NAME'. Use 'enron' or 'gmail'."
        exit 1
        ;;
esac

DB_FILE="$DATA_DIR/${DB_NAME}.db"

echo "Starting SurrealDB..."
echo "  Database: $DB_NAME"
echo "  Data file: $DB_FILE"
echo "  Endpoint: http://127.0.0.1:8000"
echo ""
echo "Connect with: surreal sql --endpoint http://127.0.0.1:8000 --user root --pass root --ns email --db $DB_NAME"
echo ""

# Ensure data directory exists
mkdir -p "$DATA_DIR"

# Start SurrealDB
exec surreal start "file:$DB_FILE" \
    --user root \
    --pass root \
    --bind 127.0.0.1:8000 \
    --log info
