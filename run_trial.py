"""Orchestrate a single blinded Erben trial.

Runs the full pipeline: decompose → profile → format → call claude CLI → record.
Each of the three steps is a separate claude invocation with fresh context.

Usage:
    python3 run_trial.py --word "rút" --target-domain "SD-15" --trial-type "target_present"
    python3 run_trial.py --word "rút" --target-domain "SD-15" --trial-type "target_present" --dry-run
"""

import argparse
import datetime
import random
import re
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple


def _ts() -> str:
    """Return current local time as [HH:MM:SS] for log lines."""
    return datetime.datetime.now().strftime("[%H:%M:%S]")

from pipeline.phoneme_decomposer import decompose_word
from pipeline.activation_profiler import ActivationProfile, build_activation_profile
from pipeline.prompt_formatter import (
    format_step1_prompt,
    format_step2_prompt,
    format_step3_prompt,
    get_system_prompt,
    get_step3_system_prompt,
)
from pipeline.config_loader import load_config
from pipeline.option_selector import select_options
from pipeline.random_profile_generator import generate_random_profile
from pipeline.result_recorder import save_profile, save_step, save_metadata
from pipeline.model_runner import run_model, is_local_model, strip_thinking_tags


def _generate_dry_run_response(
    step: int,
    profile: ActivationProfile,
    options: Optional[List[Dict[str, Any]]] = None,
    rng: Optional[random.Random] = None,
) -> str:
    """Generate a profile-aware dummy response for dry-run mode.

    Unlike static strings, these responses reference the actual MC codes from
    the profile and pick from the actual forced-choice options, making dry-run
    useful for testing the full pipeline including response parsing.
    """
    if rng is None:
        rng = random.Random()

    letters = "ABCDEFGH"

    if step == 1:
        # Reference actual consonant layer codes
        strong_codes = []
        for pos in profile["consonant_layer"]:
            for act in pos["activations"]:
                if act["tier"] == "strong":
                    strong_codes.append(act["macro_concept_id"])
        codes_str = ", ".join(strong_codes[:4]) if strong_codes else "no strong codes"
        chosen_idx = rng.randint(0, len(options) - 1) if options else 0
        letter = letters[chosen_idx]
        return (
            f"The consonant frame shows strong activations at {codes_str}. "
            f"The pattern across positions suggests a consistent theme. "
            f"My choice is {letter}."
        )

    elif step == 2:
        # Reference actual vowel layer codes
        strong_codes = []
        for pos in profile["vowel_layer"]:
            for act in pos["activations"]:
                if act["tier"] == "strong":
                    strong_codes.append(act["macro_concept_id"])
        codes_str = ", ".join(strong_codes[:4]) if strong_codes else "no strong codes"
        reinforced = ", ".join(profile["reinforced_codes"][:3]) if profile["reinforced_codes"] else "none"
        chosen_idx = rng.randint(0, len(options) - 1) if options else 0
        letter = letters[chosen_idx]
        return (
            f"The vowel data adds {codes_str}. "
            f"Reinforcement across layers ({reinforced}) supports the reading. "
            f"My choice is {letter}."
        )

    else:
        # Step 3: reference decoded concept names
        codes = load_config("concept_codes.json")
        code_to_label = codes["code_to_label"]
        strong_labels = []
        for pos in profile["consonant_layer"] + profile["vowel_layer"]:
            for act in pos["activations"]:
                if act["tier"] == "strong":
                    strong_labels.append(act["macro_concept"])
        unique_labels = list(dict.fromkeys(strong_labels))[:5]
        labels_str = ", ".join(l.lower() for l in unique_labels) if unique_labels else "unclear"
        return (
            f"1. Semantic field prediction: the profile's strong activations "
            f"({labels_str}) suggest a domain related to {unique_labels[0].lower() if unique_labels else 'unknown'}.\n"
            f"2. Confidence: moderate\n"
            f"3. This word likely refers to something connected to {labels_str}."
        )


def parse_forced_choice(
    response: str,
    options: List[Dict[str, Any]],
) -> str:
    """Extract the chosen semantic domain from a forced-choice response.

    Tries multiple strategies:
    0. Look for explicit "CHOICE: <label or letter>" line (reasoning prompt format)
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

    # Strategy 0: CHOICE: line (reasoning prompt format)
    choice_line_match = re.search(r'CHOICE:\s*(.+)', response, re.IGNORECASE)
    if choice_line_match:
        choice_text = choice_line_match.group(1).strip()
        # Sub-strategy: letter (e.g., "CHOICE: A")
        letter_match = re.match(r'^([A-H])\b', choice_text, re.IGNORECASE)
        if letter_match:
            letter = letter_match.group(1).upper()
            idx = letters.index(letter)
            if idx < len(options):
                opt = options[idx]
                return f"{opt['id']} ({opt['label']})"
        # Sub-strategy: label text (e.g., "CHOICE: DECAY_DETERIORATION")
        choice_upper = choice_text.upper()
        for opt in options:
            label = opt.get("label", "")
            if label and label.upper() in choice_upper:
                return f"{opt['id']} ({opt['label']})"

    # Strategy 1: Explicit choice patterns
    choice_patterns = [
        r'(?:my\s+)?choice\s+(?:is\s+)?([A-H])\b',
        r'I\s+choose\s+([A-H])\b',
        r'I\s+(?:would\s+)?(?:pick|select)\s+([A-H])\b',
        r"I'?ll\s+go\s+with\s+([A-H])\b",
        r'(?:most\s+consistent|best\s+match(?:es)?)\s+.*?\b([A-H])\b',
        r'\bchoice:\s*([A-H])\b',
        r'\banswer\s+(?:is\s+)?([A-H])\b',
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
    decode_all_steps: bool = False,
    reasoning_prompt: bool = False,
    timeout: int = 600,
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
        decode_all_steps: If True, use human-readable labels in steps 1-2 instead
            of MC-XX codes.
        reasoning_prompt: If True, use structured counting-based system prompts
            designed for models that struggle with gestalt synthesis.
        timeout: Seconds to wait per model call (Ollama only).

    Returns:
        Summary dict with choices, predictions, and file paths.
    """
    if run_id is None:
        run_id = uuid.uuid4().hex[:8]

    rng = random.Random(seed) if seed is not None else random.Random()
    word_id = _sanitize_word_id(word)
    is_control = trial_type == "random_profile"
    system_prompt = get_system_prompt(decode_all_steps, reasoning_prompt)
    step3_system_prompt = get_step3_system_prompt(reasoning_prompt)

    print(f"{_ts()} [Trial] word={word}, domain={target_domain}, type={trial_type}, run={run_id}")

    # Step 0: Build activation profile
    if trial_type == "random_profile":
        phonemes = decompose_word(word)
        num_c = sum(1 for p in phonemes if p["type"] == "consonant")
        num_v = sum(1 for p in phonemes if p["type"] == "vowel")
        profile = generate_random_profile(
            num_consonant_positions=num_c,
            num_vowel_positions=num_v,
            rng=rng,
        )
        print(f"{_ts()}   Generated random profile ({num_c}C, {num_v}V)")
    else:
        phonemes = decompose_word(word)
        profile = build_activation_profile(phonemes)
        ipa_seq = [p["ipa"] for p in phonemes]
        print(f"{_ts()}   Decomposed: {ipa_seq}")

    # Select forced-choice options (same for both steps for now)
    step1_options = select_options(target_domain, trial_type, num_options, rng)
    step2_options = select_options(target_domain, trial_type, num_options, rng)

    # Step 1: Consonant frame
    print(f"{_ts()}   Step 1: Consonant frame — sending to model...")
    t1 = time.monotonic()
    prompt1 = format_step1_prompt(profile, step1_options, decode_all_steps=decode_all_steps)
    if dry_run:
        response1 = _generate_dry_run_response(1, profile, step1_options, rng)
        response1_for_parse = response1
    else:
        response1 = run_model(prompt1, model, system_prompt, timeout=timeout)
        response1_for_parse, _ = strip_thinking_tags(response1)
    choice1 = parse_forced_choice(response1_for_parse, step1_options)
    print(f"{_ts()}   Step 1 done ({time.monotonic()-t1:.0f}s) → {choice1}")

    # Step 2: Vowel fill
    print(f"{_ts()}   Step 2: Vowel fill — sending to model...")
    t2 = time.monotonic()
    prompt2 = format_step2_prompt(profile, step2_options, response1_for_parse, choice1, decode_all_steps=decode_all_steps)
    if dry_run:
        response2 = _generate_dry_run_response(2, profile, step2_options, rng)
        response2_for_parse = response2
    else:
        if not is_local_model(model):
            time.sleep(5)  # Rate-limit protection between claude calls
        response2 = run_model(prompt2, model, system_prompt, timeout=timeout)
        response2_for_parse, _ = strip_thinking_tags(response2)
    choice2 = parse_forced_choice(response2_for_parse, step2_options)
    print(f"{_ts()}   Step 2 done ({time.monotonic()-t2:.0f}s) → {choice2}")

    # Step 3: Synthesis (open-ended, no forced choice)
    print(f"{_ts()}   Step 3: Synthesis — sending to model...")
    t3 = time.monotonic()
    prompt3 = format_step3_prompt(profile, response1_for_parse, choice1, response2_for_parse, choice2)
    if dry_run:
        response3 = _generate_dry_run_response(3, profile, rng=rng)
    else:
        if not is_local_model(model):
            time.sleep(5)  # Rate-limit protection between claude calls
        response3 = run_model(prompt3, model, step3_system_prompt, timeout=timeout)
    print(f"{_ts()}   Step 3 done ({time.monotonic()-t3:.0f}s) — {response3[:80]}...")

    # Record everything (skipped in dry-run to avoid polluting results/)
    if not dry_run:
        print(f"{_ts()}   Saving artifacts...")
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
            extra={"seed": seed, "dry_run": dry_run, "decode_all_steps": decode_all_steps, "reasoning_prompt": reasoning_prompt},
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

    if dry_run:
        print(f"{_ts()}   Done. (dry-run, no artifacts saved)")
    else:
        print(f"{_ts()}   Done. Saved to {'controls' if is_control else 'results'}/{word_id}_{run_id}/")
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
    parser.add_argument("--model", default="sonnet",
                        help="Model to use: 'sonnet', 'opus', 'claude', or 'ollama:model_name' (default: sonnet)")
    parser.add_argument("--num-options", type=int, default=4, help="Number of forced-choice options")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip claude CLI calls, use dummy responses")
    parser.add_argument("--decode-all-steps", action="store_true",
                        help="Use human-readable concept labels in steps 1-2 instead of MC-XX codes")
    parser.add_argument("--reasoning-prompt", action="store_true",
                        help="Use structured counting-based system prompts instead of gestalt synthesis prompts")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Seconds to wait per model call for Ollama models (default: 600)")

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
        decode_all_steps=args.decode_all_steps,
        reasoning_prompt=args.reasoning_prompt,
        timeout=args.timeout,
    )

    print(f"\n=== Trial Summary ===")
    print(f"Word: {summary['word']}")
    print(f"Target: {summary['target_domain']}")
    print(f"Step 1 choice: {summary['step1_choice']}")
    print(f"Step 2 choice: {summary['step2_choice']}")


if __name__ == "__main__":
    main()
