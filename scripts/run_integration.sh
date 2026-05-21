#!/bin/bash
# Run integration tests for the local coding agent.
# Usage: ./scripts/run_integration.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================"
echo " Local Coding Agent — Integration Tests"
echo "============================================"
echo ""

# Activate venv
VENV="$PROJECT_ROOT/.venv"
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
    echo "[OK] Virtual environment activated"
else
    echo "[FAIL] Virtual environment not found at $VENV"
    echo "       Run: cd $PROJECT_ROOT && python -m venv .venv && pip install -e ."
    exit 1
fi
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Run pytest with integration marker
echo "Running integration tests..."
echo ""
python -m pytest tests/test_integration.py -v -m integration

echo ""
echo "============================================"
echo " Integration tests complete"
echo "============================================"
