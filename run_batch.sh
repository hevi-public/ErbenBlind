#!/bin/bash
# Run blinded Erben trials in batch from a word list.
#
# Usage:
#   ./run_batch.sh --word-list words.txt --runs-per-word 5
#   ./run_batch.sh --word-list words.txt --runs-per-word 5 --include-controls
#   ./run_batch.sh --word-list words.txt --runs-per-word 1 --include-controls --dry-run
#
# Word list format (TSV, header line optional):
#   word	target_domain	actual_meaning
#   rút	SD-15	ugly, foul, repulsive
#   víz	SD-01	water
#
# With --include-controls, each word also gets target_absent, null_trial, and
# random_profile runs (1 each per word, in addition to the N target_present runs).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
WORD_LIST=""
RUNS_PER_WORD=1
INCLUDE_CONTROLS=false
MODEL="sonnet"
NUM_OPTIONS=4
DRY_RUN=""
SEED=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --word-list)
            WORD_LIST="$2"
            shift 2
            ;;
        --runs-per-word)
            RUNS_PER_WORD="$2"
            shift 2
            ;;
        --include-controls)
            INCLUDE_CONTROLS=true
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --num-options)
            NUM_OPTIONS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --word-list FILE --runs-per-word N [--include-controls] [--model MODEL] [--dry-run]"
            echo ""
            echo "Word list format (TSV): word<tab>target_domain<tab>actual_meaning"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$WORD_LIST" ]]; then
    echo "Error: --word-list is required" >&2
    exit 1
fi

if [[ ! -f "$WORD_LIST" ]]; then
    echo "Error: Word list file not found: $WORD_LIST" >&2
    exit 1
fi

# Count words (skip header and empty lines)
WORD_COUNT=0
while IFS=$'\t' read -r word domain meaning; do
    [[ -z "$word" || "$word" == "word" ]] && continue
    WORD_COUNT=$((WORD_COUNT + 1))
done < "$WORD_LIST"

TRIAL_TYPES=("target_present")
if $INCLUDE_CONTROLS; then
    TRIAL_TYPES+=("target_absent" "null_trial" "random_profile")
fi

TOTAL_TRIALS=$((WORD_COUNT * RUNS_PER_WORD))
if $INCLUDE_CONTROLS; then
    TOTAL_TRIALS=$((TOTAL_TRIALS + WORD_COUNT * 3))
fi

echo "============================================"
echo "Erben Blind Batch Runner"
echo "============================================"
echo "Word list:      $WORD_LIST"
echo "Words:          $WORD_COUNT"
echo "Runs per word:  $RUNS_PER_WORD"
echo "Controls:       $INCLUDE_CONTROLS"
echo "Model:          $MODEL"
echo "Total trials:   $TOTAL_TRIALS"
echo "============================================"
echo ""

COMPLETED=0
FAILED=0
RUN_COUNTER=1

# Process each word
while IFS=$'\t' read -r word domain meaning; do
    # Skip header and empty lines
    [[ -z "$word" || "$word" == "word" ]] && continue

    echo ">>> Processing: $word (target: $domain)"

    # Target-present runs
    for ((run=1; run<=RUNS_PER_WORD; run++)); do
        RUN_ID=$(printf "%03d" $RUN_COUNTER)
        SEED_ARG=""
        if [[ -n "$SEED" ]]; then
            SEED_ARG="--seed $((SEED + RUN_COUNTER))"
        fi

        echo "  [${COMPLETED}/${TOTAL_TRIALS}] target_present run ${run}/${RUNS_PER_WORD} (ID: ${RUN_ID})"

        if python3 run_trial.py \
            --word "$word" \
            --target-domain "$domain" \
            --trial-type "target_present" \
            --run-id "$RUN_ID" \
            --model "$MODEL" \
            --num-options "$NUM_OPTIONS" \
            $SEED_ARG \
            $DRY_RUN; then
            COMPLETED=$((COMPLETED + 1))
        else
            echo "  WARNING: Trial failed for $word (target_present, run $RUN_ID)" >&2
            FAILED=$((FAILED + 1))
        fi

        RUN_COUNTER=$((RUN_COUNTER + 1))
        echo ""
    done

    # Control runs (1 each)
    if $INCLUDE_CONTROLS; then
        for trial_type in "target_absent" "null_trial" "random_profile"; do
            RUN_ID=$(printf "%03d" $RUN_COUNTER)
            SEED_ARG=""
            if [[ -n "$SEED" ]]; then
                SEED_ARG="--seed $((SEED + RUN_COUNTER))"
            fi

            echo "  [${COMPLETED}/${TOTAL_TRIALS}] ${trial_type} (ID: ${RUN_ID})"

            if python3 run_trial.py \
                --word "$word" \
                --target-domain "$domain" \
                --trial-type "$trial_type" \
                --run-id "$RUN_ID" \
                --model "$MODEL" \
                --num-options "$NUM_OPTIONS" \
                $SEED_ARG \
                $DRY_RUN; then
                COMPLETED=$((COMPLETED + 1))
            else
                echo "  WARNING: Trial failed for $word (${trial_type}, run $RUN_ID)" >&2
                FAILED=$((FAILED + 1))
            fi

            RUN_COUNTER=$((RUN_COUNTER + 1))
            echo ""
        done
    fi

done < "$WORD_LIST"

echo "============================================"
echo "Batch Complete"
echo "============================================"
echo "Completed: $COMPLETED"
echo "Failed:    $FAILED"
echo "Total:     $TOTAL_TRIALS"
echo "============================================"

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
