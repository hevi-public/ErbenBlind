#!/bin/bash
# Run a single blinded Erben trial.
#
# Usage:
#   ./run_blinded_trial.sh --word "rút" --target-domain "SD-15" --trial-type "target_present"
#   ./run_blinded_trial.sh --word "rút" --target-domain "SD-15" --trial-type "target_present" --dry-run
#
# All arguments are passed through to run_trial.py.
# If --run-id is not provided, one is auto-generated.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec python3 run_trial.py "$@"
