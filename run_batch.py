"""Parallel batch runner for blinded Erben trials.

Executes multiple independent trials concurrently using ProcessPoolExecutor.
Each trial is self-contained (separate output directory, separate claude CLI calls),
so there are no shared-state conflicts.

Usage:
    python3 run_batch.py --word-list test_words.txt --runs-per-word 5 --parallel 4
    python3 run_batch.py --word-list test_words.txt --runs-per-word 2 --include-controls --dry-run
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from run_trial import run_single_trial


# Trial descriptor: all info needed to run one trial independently
TrialSpec = Tuple[str, str, str, str, str, int, Optional[int], bool]
# (word, target_domain, trial_type, run_id, model, num_options, seed, dry_run)


def parse_word_list(word_list_path: str) -> List[Dict[str, str]]:
    """Parse a TSV word list file into a list of word entries.

    Skips header lines (starting with 'word') and blank lines.

    Args:
        word_list_path: Path to TSV file with columns: word, target_domain, actual_meaning.

    Returns:
        List of dicts with keys: word, target_domain, actual_meaning.
    """
    entries = []
    with open(word_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            word = parts[0]
            if word == "word":
                continue
            entries.append({
                "word": word,
                "target_domain": parts[1],
                "actual_meaning": parts[2] if len(parts) > 2 else "",
            })
    return entries


def build_trial_specs(
    entries: List[Dict[str, str]],
    runs_per_word: int,
    include_controls: bool,
    model: str,
    num_options: int,
    seed: Optional[int],
    dry_run: bool,
) -> List[TrialSpec]:
    """Build the full list of trial specs from word entries and parameters.

    Each spec is a tuple containing all arguments needed for run_single_trial.

    Args:
        entries: Parsed word list entries.
        runs_per_word: Number of target_present runs per word.
        include_controls: Whether to add control trials (target_absent, null_trial, random_profile).
        model: Claude model name.
        num_options: Number of forced-choice options.
        seed: Base random seed (each trial gets seed + counter). None for no seeding.
        dry_run: Whether to skip actual claude CLI calls.

    Returns:
        List of TrialSpec tuples.
    """
    specs: List[TrialSpec] = []
    counter = 1

    for entry in entries:
        word = entry["word"]
        domain = entry["target_domain"]

        # Target-present runs
        for run in range(runs_per_word):
            run_id = f"{counter:03d}"
            trial_seed = (seed + counter) if seed is not None else None
            specs.append((word, domain, "target_present", run_id, model, num_options, trial_seed, dry_run))
            counter += 1

        # Control runs (1 each)
        if include_controls:
            for trial_type in ("target_absent", "null_trial", "random_profile"):
                run_id = f"{counter:03d}"
                trial_seed = (seed + counter) if seed is not None else None
                specs.append((word, domain, trial_type, run_id, model, num_options, trial_seed, dry_run))
                counter += 1

    return specs


def _run_trial_from_spec(spec: TrialSpec) -> Dict[str, Any]:
    """Unpack a TrialSpec and call run_single_trial.

    This is the worker function submitted to the process pool.
    It must be a top-level function (not a lambda/closure) for pickling.

    Args:
        spec: Tuple of (word, target_domain, trial_type, run_id, model, num_options, seed, dry_run).

    Returns:
        Trial summary dict with an added 'error' key (None on success, str on failure).
    """
    word, target_domain, trial_type, run_id, model, num_options, seed, dry_run = spec
    try:
        summary = run_single_trial(
            word=word,
            target_domain=target_domain,
            trial_type=trial_type,
            run_id=run_id,
            model=model,
            num_options=num_options,
            seed=seed,
            dry_run=dry_run,
        )
        summary["error"] = None
        return summary
    except Exception as e:
        return {
            "word": word,
            "run_id": run_id,
            "trial_type": trial_type,
            "target_domain": target_domain,
            "error": str(e),
        }


def run_batch(
    word_list_path: str,
    runs_per_word: int = 1,
    include_controls: bool = False,
    model: str = "sonnet",
    num_options: int = 4,
    seed: Optional[int] = None,
    dry_run: bool = False,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """Run a batch of blinded trials in parallel.

    Parses the word list, builds trial specs, and submits them to a process pool.
    Progress is printed as trials complete.

    Args:
        word_list_path: Path to TSV word list.
        runs_per_word: Number of target_present runs per word.
        include_controls: Add control trials per word.
        model: Claude model name.
        num_options: Number of forced-choice options.
        seed: Base random seed.
        dry_run: Skip claude CLI calls.
        max_workers: Maximum concurrent trial processes.

    Returns:
        List of trial summary dicts (one per trial).
    """
    entries = parse_word_list(word_list_path)
    if not entries:
        print("Error: No words found in word list.", file=sys.stderr)
        sys.exit(1)

    specs = build_trial_specs(entries, runs_per_word, include_controls, model, num_options, seed, dry_run)
    total = len(specs)

    print("=" * 44)
    print("Erben Blind Batch Runner (parallel)")
    print("=" * 44)
    print(f"Word list:      {word_list_path}")
    print(f"Words:          {len(entries)}")
    print(f"Runs per word:  {runs_per_word}")
    print(f"Controls:       {include_controls}")
    print(f"Model:          {model}")
    print(f"Total trials:   {total}")
    print(f"Max workers:    {max_workers}")
    print("=" * 44)
    print()

    results: List[Dict[str, Any]] = []
    completed = 0
    failed = 0
    start_time = time.monotonic()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_spec = {
            executor.submit(_run_trial_from_spec, spec): spec
            for spec in specs
        }

        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            word, _, trial_type, run_id, *_ = spec

            try:
                summary = future.result()
            except Exception as exc:
                summary = {
                    "word": word,
                    "run_id": run_id,
                    "trial_type": trial_type,
                    "error": str(exc),
                }

            results.append(summary)

            if summary.get("error"):
                failed += 1
                print(f"  FAILED [{completed + failed}/{total}] {word} ({trial_type}, {run_id}): {summary['error']}")
            else:
                completed += 1
                choice_info = ""
                if "step1_choice" in summary:
                    choice_info = f" → {summary['step1_choice']}"
                print(f"  OK [{completed + failed}/{total}] {word} ({trial_type}, {run_id}){choice_info}")

    elapsed = time.monotonic() - start_time

    print()
    print("=" * 44)
    print("Batch Complete")
    print("=" * 44)
    print(f"Completed: {completed}")
    print(f"Failed:    {failed}")
    print(f"Total:     {total}")
    print(f"Time:      {elapsed:.1f}s")
    print("=" * 44)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run blinded Erben trials in parallel batch."
    )
    parser.add_argument("--word-list", required=True, help="Path to TSV word list file")
    parser.add_argument("--runs-per-word", type=int, default=1, help="Number of target_present runs per word (default: 1)")
    parser.add_argument("--include-controls", action="store_true", help="Add control trials per word")
    parser.add_argument("--model", default="sonnet",
                        help="Model to use: 'sonnet', 'opus', 'claude', or 'ollama:model_name' (default: sonnet)")
    parser.add_argument("--num-options", type=int, default=4, help="Number of forced-choice options (default: 4)")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true", help="Skip claude CLI calls, use dummy responses")
    parser.add_argument("--parallel", type=int, default=4, help="Maximum concurrent trials (default: 4)")

    args = parser.parse_args()

    results = run_batch(
        word_list_path=args.word_list,
        runs_per_word=args.runs_per_word,
        include_controls=args.include_controls,
        model=args.model,
        num_options=args.num_options,
        seed=args.seed,
        dry_run=args.dry_run,
        max_workers=args.parallel,
    )

    # Exit with error if any trials failed
    if any(r.get("error") for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
