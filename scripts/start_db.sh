#!/bin/bash
# Start both Enron and Gmail SurrealDB databases
#
# Usage:
#   ./scripts/start_db.sh           # Start both databases
#   ./scripts/start_db.sh enron     # Start only Enron database
#   ./scripts/start_db.sh gmail     # Start only Gmail database
#
# Ports:
#   - Enron: 8000
#   - Gmail: 8001

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
PID_DIR="$PROJECT_DIR/.runtime"

# Ensure directories exist
mkdir -p "$DATA_DIR"
mkdir -p "$PID_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

start_db() {
    local db_name=$1
    local port=$2
    local pid_file="$PID_DIR/${db_name}.pid"
    local log_file="$PID_DIR/${db_name}.log"
    local db_file="$DATA_DIR/${db_name}.db"

    # Check if already running
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}$db_name database already running (PID: $pid)${NC}"
            return 0
        else
            # Stale PID file
            rm -f "$pid_file"
        fi
    fi

    echo "Starting $db_name database on port $port..."

    # Start SurrealDB in background
    surreal start "file:$db_file" \
        --user root \
        --pass root \
        --bind "127.0.0.1:$port" \
        --log warn \
        > "$log_file" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"

    # Wait for startup
    local max_wait=10
    local waited=0
    while ! curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1; do
        sleep 0.5
        waited=$((waited + 1))
        if [ $waited -ge $((max_wait * 2)) ]; then
            echo "Error: $db_name database failed to start"
            cat "$log_file"
            return 1
        fi
    done

    echo -e "${GREEN}$db_name database started (PID: $pid, port: $port)${NC}"
}

# Parse arguments
DB_TARGET="${1:-all}"

case "$DB_TARGET" in
    enron)
        start_db "enron" 8000
        ;;
    gmail)
        start_db "gmail" 8001
        ;;
    all)
        start_db "enron" 8000
        start_db "gmail" 8001
        ;;
    *)
        echo "Error: Unknown database '$DB_TARGET'. Use 'enron', 'gmail', or 'all'."
        exit 1
        ;;
esac

echo ""
echo "Connect with:"
if [ "$DB_TARGET" = "all" ] || [ "$DB_TARGET" = "enron" ]; then
    echo "  Enron: surreal sql --endpoint http://127.0.0.1:8000 --user root --pass root --ns rl_emails --db enron"
fi
if [ "$DB_TARGET" = "all" ] || [ "$DB_TARGET" = "gmail" ]; then
    echo "  Gmail: surreal sql --endpoint http://127.0.0.1:8001 --user root --pass root --ns rl_emails --db gmail"
fi
