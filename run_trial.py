"""Orchestrate a single blinded Erben trial.

Runs the full pipeline: decompose → profile → format → call claude CLI → record.
Each of the three steps is a separate claude invocation with fresh context.

Usage:
    python3 run_trial.py --word "rút" --target-domain "SD-15" --trial-type "target_present"
    python3 run_trial.py --word "rút" --target-domain "SD-15" --trial-type "target_present" --dry-run
"""

import argparse
import random
import re
import subprocess
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

from pipeline.phoneme_decomposer import decompose_word
from pipeline.activation_profiler import ActivationProfile, build_activation_profile
from pipeline.prompt_formatter import (
    format_step1_prompt,
    format_step2_prompt,
    format_step3_prompt,
    get_system_prompt,
)
from pipeline.option_selector import select_options
from pipeline.random_profile_generator import generate_random_profile
from pipeline.result_recorder import save_profile, save_step, save_metadata


# Dummy responses for --dry-run mode
DRY_RUN_STEP1 = (
    "Looking at the consonant frame, Position 1 shows strong activation in MC-10 "
    "and MC-05, while Position 3 has strong MC-03, MC-17, and MC-11. The overlap "
    "of MC-05 across both positions suggests a pattern of roughness or disruption. "
    "My choice is A."
)

DRY_RUN_STEP2 = (
    "The vowel data adds strong activations in MC-06, MC-12, and MC-14 at Position 2. "
    "Combined with the consonant frame, the reinforcement of MC-01 and MC-02 across "
    "layers suggests a holistic quality rather than a specific physical process. "
    "The vowel data confirms my earlier reading. My choice is A."
)

DRY_RUN_STEP3 = (
    "1. Semantic field prediction: decay/deterioration — the profile points to "
    "something that is rough, uneven, deep, and soft in a degraded sense.\n"
    "2. Confidence: moderate\n"
    "3. This word likely refers to something ugly, rotten, or in a state of decline."
)


def _call_claude(prompt: str, system_prompt: str, model: str) -> str:
    """Call the claude CLI as a subprocess with fresh context.

    Each invocation is independent — no conversation memory carries over.

    Args:
        prompt: The user-message content to send.
        system_prompt: The system prompt for the session.
        model: Model name or alias (e.g., "sonnet", "opus").

    Returns:
        The model's response text.

    Raises:
        subprocess.CalledProcessError: If the claude CLI fails.
    """
    cmd = [
        "claude",
        "--print",
        "--model", model,
        "--system-prompt", system_prompt,
        "--no-session-persistence",
        prompt,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {result.returncode}):\n"
            f"stderr: {result.stderr}\n"
            f"stdout: {result.stdout}"
        )

    return result.stdout.strip()


def parse_forced_choice(
    response: str,
    options: List[Dict[str, Any]],
) -> str:
    """Extract the chosen semantic domain from a forced-choice response.

    Tries multiple strategies:
    1. Look for explicit letter choice: "My choice is A", "I choose B", "Choice: C"
    2. Look for the last standalone letter A-H in the response
    3. Search for domain labels in the response text

    Args:
        response: The full LLM response text.
        options: The forced-choice options list (in presentation order).

    Returns:
        The chosen domain as "SD-XX (LABEL)" or "UNPARSED" if extraction fails.
    """
    letters = "ABCDEFGH"

    # Strategy 1: Explicit choice patterns
    choice_patterns = [
        r'(?:my\s+)?choice\s+(?:is\s+)?([A-H])\b',
        r'I\s+choose\s+([A-H])\b',
        r'I\s+(?:would\s+)?(?:pick|select)\s+([A-H])\b',
        r'\bchoice:\s*([A-H])\b',
        r'\b([A-H])\)\s',  # "A) " at start of a justification
    ]

    for pattern in choice_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            idx = letters.index(letter)
            if idx < len(options):
                opt = options[idx]
                return f"{opt['id']} ({opt['label']})"

    # Strategy 2: Last standalone capital letter A-H
    # Look for letters that appear standalone (not part of a word)
    standalone = re.findall(r'\b([A-H])\b', response)
    if standalone:
        letter = standalone[-1].upper()
        idx = letters.index(letter)
        if idx < len(options):
            opt = options[idx]
            return f"{opt['id']} ({opt['label']})"

    # Strategy 3: Search for domain labels in the response
    for i, opt in enumerate(options):
        label = opt["label"]
        if label in response or label.replace("_", " ") in response.upper():
            return f"{opt['id']} ({opt['label']})"

    return "UNPARSED"


def _sanitize_word_id(word: str) -> str:
    """Create a filesystem-safe word ID from a Hungarian word."""
    # Keep the original word but replace problematic chars
    safe = word.lower()
    # Replace characters that might cause filesystem issues
    for old, new in [("/", "_"), ("\\", "_"), (" ", "_")]:
        safe = safe.replace(old, new)
    return safe


def run_single_trial(
    word: str,
    target_domain: str,
    trial_type: str,
    run_id: Optional[str] = None,
    model: str = "sonnet",
    num_options: int = 4,
    seed: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a complete blinded trial for one word.

    Executes the full pipeline: decompose → profile → 3 blinded LLM steps → record.

    Args:
        word: Hungarian word to analyze.
        target_domain: Correct semantic domain ID (e.g., "SD-15").
        trial_type: One of "target_present", "target_absent", "null_trial", "random_profile".
        run_id: Unique run identifier. Auto-generated if None.
        model: Claude model to use (default: "sonnet").
        num_options: Number of forced-choice options per step.
        seed: Random seed for reproducibility.
        dry_run: If True, skip claude CLI calls and use dummy responses.

    Returns:
        Summary dict with choices, predictions, and file paths.
    """
    if run_id is None:
        run_id = uuid.uuid4().hex[:8]

    rng = random.Random(seed) if seed is not None else random.Random()
    word_id = _sanitize_word_id(word)
    is_control = trial_type == "random_profile"
    system_prompt = get_system_prompt()

    print(f"[Trial] word={word}, domain={target_domain}, type={trial_type}, run={run_id}")

    # Step 0: Build activation profile
    if trial_type == "random_profile":
        # For random profile trials, generate synthetic profile
        # Use structure similar to a typical word (2 consonants, 1 vowel)
        phonemes = decompose_word(word)
        num_c = sum(1 for p in phonemes if p["type"] == "consonant")
        num_v = sum(1 for p in phonemes if p["type"] == "vowel")
        profile = generate_random_profile(
            num_consonant_positions=num_c,
            num_vowel_positions=num_v,
            rng=rng,
        )
        print(f"  Generated random profile ({num_c}C, {num_v}V)")
    else:
        phonemes = decompose_word(word)
        profile = build_activation_profile(phonemes)
        ipa_seq = [p["ipa"] for p in phonemes]
        print(f"  Decomposed: {ipa_seq}")

    # Select forced-choice options (same for both steps for now)
    step1_options = select_options(target_domain, trial_type, num_options, rng)
    step2_options = select_options(target_domain, trial_type, num_options, rng)

    # Step 1: Consonant frame
    print("  Step 1: Consonant frame...")
    prompt1 = format_step1_prompt(profile, step1_options)
    if dry_run:
        response1 = DRY_RUN_STEP1
    else:
        response1 = _call_claude(prompt1, system_prompt, model)
    choice1 = parse_forced_choice(response1, step1_options)
    print(f"    Choice: {choice1}")

    # Step 2: Vowel fill
    print("  Step 2: Vowel fill...")
    prompt2 = format_step2_prompt(profile, step2_options, response1, choice1)
    if dry_run:
        response2 = DRY_RUN_STEP2
    else:
        response2 = _call_claude(prompt2, system_prompt, model)
    choice2 = parse_forced_choice(response2, step2_options)
    print(f"    Choice: {choice2}")

    # Step 3: Synthesis (open-ended, no forced choice)
    print("  Step 3: Synthesis...")
    prompt3 = format_step3_prompt(profile, response1, choice1, response2, choice2)
    if dry_run:
        response3 = DRY_RUN_STEP3
    else:
        response3 = _call_claude(prompt3, system_prompt, model)
    print(f"    Prediction: {response3[:100]}...")

    # Record everything
    print("  Saving artifacts...")
    save_profile(word_id, run_id, profile, is_control=is_control)
    save_step(word_id, run_id, 1, "consonant", prompt1, response1, is_control=is_control)
    save_step(word_id, run_id, 2, "vowel", prompt2, response2, is_control=is_control)
    save_step(word_id, run_id, 3, "synthesis", prompt3, response3, is_control=is_control)
    save_metadata(
        word_id, run_id,
        word=word,
        target_domain=target_domain,
        trial_type=trial_type,
        model=model,
        forced_choice_options_step1=step1_options,
        forced_choice_options_step2=step2_options,
        step1_choice=choice1,
        step2_choice=choice2,
        step3_prediction=response3,
        extra={"seed": seed, "dry_run": dry_run},
        is_control=is_control,
    )

    summary = {
        "word": word,
        "word_id": word_id,
        "run_id": run_id,
        "trial_type": trial_type,
        "target_domain": target_domain,
        "step1_choice": choice1,
        "step2_choice": choice2,
        "step3_prediction": response3,
    }

    print(f"  Done. Saved to {'controls' if is_control else 'results'}/{word_id}_{run_id}/")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run a single blinded Erben trial."
    )
    parser.add_argument("--word", required=True, help="Hungarian word to analyze")
    parser.add_argument("--target-domain", required=True, help="Correct semantic domain ID (e.g., SD-15)")
    parser.add_argument("--trial-type", required=True,
                        choices=["target_present", "target_absent", "null_trial", "random_profile"],
                        help="Trial type")
    parser.add_argument("--run-id", default=None, help="Run identifier (auto-generated if omitted)")
    parser.add_argument("--model", default="sonnet", help="Claude model to use (default: sonnet)")
    parser.add_argument("--num-options", type=int, default=4, help="Number of forced-choice options")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip claude CLI calls, use dummy responses")

    args = parser.parse_args()

    summary = run_single_trial(
        word=args.word,
        target_domain=args.target_domain,
        trial_type=args.trial_type,
        run_id=args.run_id,
        model=args.model,
        num_options=args.num_options,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    print(f"\n=== Trial Summary ===")
    print(f"Word: {summary['word']}")
    print(f"Target: {summary['target_domain']}")
    print(f"Step 1 choice: {summary['step1_choice']}")
    print(f"Step 2 choice: {summary['step2_choice']}")


if __name__ == "__main__":
    main()
