#!/bin/bash
# Initialize SurrealDB schema for email RL training
#
# Prerequisites:
#   - SurrealDB must be running (./scripts/start_surreal.sh)
#
# Usage:
#   ./scripts/init_surreal.sh [database]
#
# Examples:
#   ./scripts/init_surreal.sh          # Initialize Enron database (default)
#   ./scripts/init_surreal.sh gmail    # Initialize Gmail database

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SCHEMA_FILE="$PROJECT_DIR/db/schema.surql"

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

echo "Initializing SurrealDB schema..."
echo "  Database: $DB_NAME"
echo "  Schema: $SCHEMA_FILE"
echo ""

# Check if SurrealDB is running
if ! curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "Error: SurrealDB is not running."
    echo "Start it first: ./scripts/start_surreal.sh $DB_NAME"
    exit 1
fi

# Apply schema
surreal import \
    --endpoint http://127.0.0.1:8000 \
    --user root \
    --pass root \
    --ns email \
    --db "$DB_NAME" \
    "$SCHEMA_FILE"

echo ""
echo "Schema initialized successfully!"
echo ""
echo "Next steps:"
echo "  1. Import data: python -m db.import_data $DB_NAME data/train.json data/val.json data/test.json"
echo "  2. Query data: surreal sql --endpoint http://127.0.0.1:8000 --user root --pass root --ns email --db $DB_NAME"
