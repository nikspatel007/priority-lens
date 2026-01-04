#!/bin/bash
# Gracefully stop SurrealDB databases
#
# Usage:
#   ./scripts/stop_db.sh           # Stop all databases
#   ./scripts/stop_db.sh enron     # Stop only Enron database
#   ./scripts/stop_db.sh gmail     # Stop only Gmail database

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_DIR="$PROJECT_DIR/.runtime"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

stop_db() {
    local db_name=$1
    local pid_file="$PID_DIR/${db_name}.pid"

    if [ ! -f "$pid_file" ]; then
        echo -e "${YELLOW}$db_name database not running (no PID file)${NC}"
        return 0
    fi

    local pid=$(cat "$pid_file")

    if ! kill -0 "$pid" 2>/dev/null; then
        echo -e "${YELLOW}$db_name database not running (stale PID: $pid)${NC}"
        rm -f "$pid_file"
        return 0
    fi

    echo "Stopping $db_name database (PID: $pid)..."

    # Send SIGTERM for graceful shutdown
    kill -TERM "$pid" 2>/dev/null || true

    # Wait for process to exit
    local max_wait=10
    local waited=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep 0.5
        waited=$((waited + 1))
        if [ $waited -ge $((max_wait * 2)) ]; then
            echo -e "${RED}$db_name database did not stop gracefully, sending SIGKILL${NC}"
            kill -KILL "$pid" 2>/dev/null || true
            break
        fi
    done

    rm -f "$pid_file"
    echo -e "${GREEN}$db_name database stopped${NC}"
}

# Parse arguments
DB_TARGET="${1:-all}"

case "$DB_TARGET" in
    enron)
        stop_db "enron"
        ;;
    gmail)
        stop_db "gmail"
        ;;
    all)
        stop_db "enron"
        stop_db "gmail"
        ;;
    *)
        echo "Error: Unknown database '$DB_TARGET'. Use 'enron', 'gmail', or 'all'."
        exit 1
        ;;
esac
