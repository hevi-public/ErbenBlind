"""Save all trial artifacts to disk for auditability.

Each trial produces a directory containing the activation profile,
all prompts and responses, and run metadata.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.activation_profiler import ActivationProfile


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CONTROLS_DIR = Path(__file__).resolve().parent.parent / "controls"


def _ensure_dir(path: Path) -> None:
    """Create directory and parents if they don't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _trial_dir(
    word_id: str,
    run_id: str,
    is_control: bool = False,
) -> Path:
    """Get the output directory path for a trial."""
    base = CONTROLS_DIR if is_control else RESULTS_DIR
    return base / f"{word_id}_{run_id}"


def save_profile(
    word_id: str,
    run_id: str,
    profile: ActivationProfile,
    is_control: bool = False,
) -> Path:
    """Save the activation profile as profile.json.

    Args:
        word_id: Identifier for the word (e.g., sanitized word string).
        run_id: Unique run identifier.
        profile: The activation profile dict.
        is_control: If True, saves under controls/ instead of results/.

    Returns:
        Path to the saved file.
    """
    out_dir = _trial_dir(word_id, run_id, is_control)
    _ensure_dir(out_dir)

    path = out_dir / "profile.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    return path


def save_step(
    word_id: str,
    run_id: str,
    step_number: int,
    step_name: str,
    prompt: str,
    response: str,
    is_control: bool = False,
) -> tuple:
    """Save a prompt-response pair for one step of a trial.

    Args:
        word_id: Identifier for the word.
        run_id: Unique run identifier.
        step_number: Step number (1, 2, or 3).
        step_name: Short name for the step (e.g., "consonant", "vowel", "synthesis").
        prompt: The exact prompt sent to the LLM.
        response: The full LLM response.
        is_control: If True, saves under controls/.

    Returns:
        Tuple of (prompt_path, response_path).
    """
    out_dir = _trial_dir(word_id, run_id, is_control)
    _ensure_dir(out_dir)

    prompt_path = out_dir / f"step{step_number}_{step_name}_prompt.txt"
    response_path = out_dir / f"step{step_number}_{step_name}_response.txt"

    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    with open(response_path, "w", encoding="utf-8") as f:
        f.write(response)

    return prompt_path, response_path


def save_metadata(
    word_id: str,
    run_id: str,
    word: str,
    target_domain: str,
    trial_type: str,
    model: str = "claude",
    forced_choice_options_step1: Optional[List[Dict[str, Any]]] = None,
    forced_choice_options_step2: Optional[List[Dict[str, Any]]] = None,
    step1_choice: Optional[str] = None,
    step2_choice: Optional[str] = None,
    step3_prediction: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    is_control: bool = False,
) -> Path:
    """Save trial metadata as meta.json.

    Args:
        word_id: Identifier for the word.
        run_id: Unique run identifier.
        word: The actual Hungarian word (for evaluation only, never shown to LLM).
        target_domain: The correct semantic domain ID.
        trial_type: One of target_present, target_absent, null_trial, random_profile.
        model: Model identifier used for the LLM calls.
        forced_choice_options_step1: Options presented in step 1.
        forced_choice_options_step2: Options presented in step 2.
        step1_choice: Domain chosen by LLM in step 1.
        step2_choice: Domain chosen by LLM in step 2.
        step3_prediction: Free-text prediction from step 3.
        extra: Any additional metadata to include.
        is_control: If True, saves under controls/.

    Returns:
        Path to the saved file.
    """
    out_dir = _trial_dir(word_id, run_id, is_control)
    _ensure_dir(out_dir)

    meta: Dict[str, Any] = {
        "word_id": word_id,
        "run_id": run_id,
        "word": word,
        "target_domain": target_domain,
        "trial_type": trial_type,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "forced_choice_options_step1": forced_choice_options_step1,
        "forced_choice_options_step2": forced_choice_options_step2,
        "step1_choice": step1_choice,
        "step2_choice": step2_choice,
        "step3_prediction": step3_prediction,
    }

    if extra:
        meta.update(extra)

    path = out_dir / "meta.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return path


def list_trials(is_control: bool = False) -> List[str]:
    """List all trial directory names under results/ or controls/.

    Returns:
        Sorted list of directory names (e.g., ["rút_001", "víz_002"]).
    """
    base = CONTROLS_DIR if is_control else RESULTS_DIR
    if not base.exists():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and (d / "meta.json").exists()
    )


def trial_exists(word_id: str, run_id: str, is_control: bool = False) -> bool:
    """Return True if a trial's results are already on disk.

    A trial is considered complete when its meta.json exists. This is the
    last file written by save_metadata(), so its presence means the full
    pipeline ran successfully for that trial.
    """
    return (_trial_dir(word_id, run_id, is_control) / "meta.json").exists()


def load_trial(word_id: str, run_id: str, is_control: bool = False) -> Dict[str, Any]:
    """Load all artifacts from a trial directory.

    Returns:
        Dict with keys: profile, meta, and step{N}_{name}_{prompt|response}
        for each saved step.
    """
    out_dir = _trial_dir(word_id, run_id, is_control)

    result: Dict[str, Any] = {}

    profile_path = out_dir / "profile.json"
    if profile_path.exists():
        with open(profile_path, encoding="utf-8") as f:
            result["profile"] = json.load(f)

    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            result["meta"] = json.load(f)

    # Load all step files
    for txt_file in sorted(out_dir.glob("step*_*.txt")):
        key = txt_file.stem  # e.g., "step1_consonant_prompt"
        with open(txt_file, encoding="utf-8") as f:
            result[key] = f.read()

    return result


if __name__ == "__main__":
    import tempfile

    # Demo: save a minimal trial to a temp location
    print("=== Result Recorder Demo ===\n")

    # Override dirs for demo
    original_results = RESULTS_DIR
    with tempfile.TemporaryDirectory() as tmpdir:
        import pipeline.result_recorder as rr
        rr.RESULTS_DIR = Path(tmpdir) / "results"

        save_profile("rút", "001", {
            "consonant_layer": [], "vowel_layer": [],
            "reinforced_codes": [], "activation_density": {},
            "strong_across_positions": [],
            "metadata": {"phoneme_count": 3},
        })  # type: ignore[arg-type]

        save_step("rút", "001", 1, "consonant",
                  "This is the prompt.", "This is the response.")

        save_metadata("rút", "001",
                      word="rút", target_domain="SD-15",
                      trial_type="target_present",
                      step1_choice="SD-15")

        # List and load
        trials = list_trials()
        print(f"Trials found: {trials}")

        data = load_trial("rút", "001")
        print(f"Loaded keys: {sorted(data.keys())}")
        print(f"Meta word: {data['meta']['word']}")
        print(f"Step 1 prompt: {data['step1_consonant_prompt'][:40]}...")

        rr.RESULTS_DIR = original_results

    print("\nDemo complete (temp files cleaned up).")
