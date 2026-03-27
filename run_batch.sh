#!/bin/bash
# Run blinded Erben trials in parallel batch.
#
# Usage:
#   ./run_batch.sh --word-list words.txt --runs-per-word 5 --parallel 4
#   ./run_batch.sh --word-list words.txt --runs-per-word 5 --include-controls --dry-run
#
# All arguments are passed through to run_batch.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec python3 run_batch.py "$@"
