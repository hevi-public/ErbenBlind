"""Evaluate blinded Erben trial results.

Computes hit rates, false positive rates, signal strength, and statistical
significance across completed trials.

Usage:
    python3 evaluate.py
    python3 evaluate.py --output evaluation_report.json
"""

import argparse
import json
import re
import sys
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline.result_recorder import list_trials, load_trial


def binomial_p_value(successes: int, trials: int, p_chance: float) -> float:
    """One-sided binomial test: P(X >= successes) under null hypothesis.

    Tests whether the observed hit rate is significantly better than chance.

    Args:
        successes: Number of correct picks.
        trials: Total number of trials.
        p_chance: Probability of correct pick by chance (e.g., 0.25 for 4 options).

    Returns:
        p-value (probability of observing this many or more hits by chance).
    """
    if trials == 0:
        return 1.0
    p_value = sum(
        comb(trials, k) * p_chance**k * (1 - p_chance)**(trials - k)
        for k in range(successes, trials + 1)
    )
    return min(p_value, 1.0)


def _extract_domain_id(choice_str: str) -> Optional[str]:
    """Extract the SD-XX domain ID from a choice string like 'SD-15 (DECAY_DETERIORATION)'."""
    if not choice_str or choice_str == "UNPARSED":
        return None
    # Match SD-XX pattern
    import re
    match = re.match(r'(SD-\d+)', choice_str)
    return match.group(1) if match else None


def _is_hit(choice_str: str, target_domain: str) -> bool:
    """Check if a choice matches the target domain."""
    chosen_id = _extract_domain_id(choice_str)
    return chosen_id == target_domain


def load_all_trials() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load all result and control trials.

    Returns:
        Tuple of (result_trials, control_trials), each a list of trial dicts.
    """
    results = []
    for trial_name in list_trials(is_control=False):
        parts = trial_name.rsplit("_", 1)
        if len(parts) == 2:
            word_id, run_id = parts
        else:
            continue
        try:
            data = load_trial(word_id, run_id, is_control=False)
            if "meta" in data:
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load trial {trial_name}: {e}", file=sys.stderr)

    controls = []
    for trial_name in list_trials(is_control=True):
        parts = trial_name.rsplit("_", 1)
        if len(parts) == 2:
            word_id, run_id = parts
        else:
            continue
        try:
            data = load_trial(word_id, run_id, is_control=True)
            if "meta" in data:
                controls.append(data)
        except Exception as e:
            print(f"Warning: Could not load control {trial_name}: {e}", file=sys.stderr)

    return results, controls


def compute_hit_rates(
    trials: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Compute hit rates grouped by trial type and step.

    Returns:
        Dict keyed by trial_type, each containing:
        - step1_hits, step1_total, step1_rate
        - step2_hits, step2_total, step2_rate
        - unparsed_count
    """
    by_type: Dict[str, Dict[str, Any]] = {}

    for trial in trials:
        meta = trial["meta"]
        trial_type = meta["trial_type"]
        target = meta["target_domain"]

        if trial_type not in by_type:
            by_type[trial_type] = {
                "step1_hits": 0, "step1_total": 0,
                "step2_hits": 0, "step2_total": 0,
                "unparsed_count": 0,
            }

        stats = by_type[trial_type]

        choice1 = meta.get("step1_choice", "")
        choice2 = meta.get("step2_choice", "")

        if choice1 == "UNPARSED":
            stats["unparsed_count"] += 1
        else:
            stats["step1_total"] += 1
            if _is_hit(choice1, target):
                stats["step1_hits"] += 1

        if choice2 == "UNPARSED":
            stats["unparsed_count"] += 1
        else:
            stats["step2_total"] += 1
            if _is_hit(choice2, target):
                stats["step2_hits"] += 1

    # Compute rates
    for trial_type, stats in by_type.items():
        stats["step1_rate"] = (
            stats["step1_hits"] / stats["step1_total"]
            if stats["step1_total"] > 0 else None
        )
        stats["step2_rate"] = (
            stats["step2_hits"] / stats["step2_total"]
            if stats["step2_total"] > 0 else None
        )

    return by_type


def compute_synthesis_consistency(
    trials: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Measure how often step 3 aligns with step 1-2 choices.

    Checks whether the step 3 free-text prediction mentions the same domain
    as the step 2 forced-choice selection.

    Returns:
        Dict with consistent_count, total, rate.
    """
    consistent = 0
    total = 0

    for trial in trials:
        meta = trial["meta"]
        choice2 = meta.get("step2_choice", "")
        prediction = meta.get("step3_prediction", "")

        if not choice2 or choice2 == "UNPARSED" or not prediction:
            continue

        total += 1

        # Check if the step 2 domain label appears in the step 3 prediction
        domain_id = _extract_domain_id(choice2)
        if domain_id and domain_id in prediction:
            consistent += 1
            continue

        # Check for label text
        # Extract label from "SD-15 (DECAY_DETERIORATION)"
        import re
        label_match = re.search(r'\((\w+)\)', choice2)
        if label_match:
            label = label_match.group(1).lower().replace("_", " ")
            if label in prediction.lower() or label.split()[0] in prediction.lower():
                consistent += 1

    return {
        "consistent_count": consistent,
        "total": total,
        "rate": consistent / total if total > 0 else None,
    }


def generate_report(
    results: List[Dict[str, Any]],
    controls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate the full evaluation report.

    Args:
        results: List of result trial dicts (non-control).
        controls: List of control trial dicts (random_profile).

    Returns:
        Complete report dict.
    """
    all_trials = results + controls
    hit_rates = compute_hit_rates(all_trials)

    # Determine chance rate from typical option count
    # Default to 0.25 (4 options) unless we can determine from data
    chance_rate = 0.25
    for trial in all_trials:
        meta = trial["meta"]
        opts = meta.get("forced_choice_options_step1")
        if opts:
            chance_rate = 1.0 / len(opts)
            break

    # Signal strength: target_present hit rate minus random_profile hit rate
    tp_stats = hit_rates.get("target_present", {})
    rp_stats = hit_rates.get("random_profile", {})

    tp_rate = tp_stats.get("step2_rate")
    rp_rate = rp_stats.get("step2_rate")

    signal_strength = None
    if tp_rate is not None and rp_rate is not None:
        signal_strength = tp_rate - rp_rate

    # Statistical significance for target_present
    tp_hits = tp_stats.get("step2_hits", 0)
    tp_total = tp_stats.get("step2_total", 0)
    p_value = binomial_p_value(tp_hits, tp_total, chance_rate)

    # Synthesis consistency (target_present trials only)
    tp_trials = [t for t in all_trials if t["meta"]["trial_type"] == "target_present"]
    consistency = compute_synthesis_consistency(tp_trials)

    # Per-word breakdown for target_present (including per-word synthesis consistency)
    word_breakdown: Dict[str, Dict[str, Any]] = {}
    for trial in tp_trials:
        meta = trial["meta"]
        word = meta["word"]
        if word not in word_breakdown:
            word_breakdown[word] = {
                "target_domain": meta["target_domain"],
                "runs": 0, "step1_hits": 0, "step2_hits": 0,
                "synthesis_consistent": 0, "synthesis_total": 0,
            }
        wb = word_breakdown[word]
        wb["runs"] += 1
        if _is_hit(meta.get("step1_choice", ""), meta["target_domain"]):
            wb["step1_hits"] += 1
        if _is_hit(meta.get("step2_choice", ""), meta["target_domain"]):
            wb["step2_hits"] += 1

        # Per-word synthesis consistency: does step 3 align with step 2?
        choice2 = meta.get("step2_choice", "")
        prediction = meta.get("step3_prediction", "")
        if choice2 and choice2 != "UNPARSED" and prediction:
            wb["synthesis_total"] += 1
            domain_id = _extract_domain_id(choice2)
            if domain_id and domain_id in prediction:
                wb["synthesis_consistent"] += 1
            else:
                label_match = re.search(r'\((\w+)\)', choice2)
                if label_match:
                    label = label_match.group(1).lower().replace("_", " ")
                    if label in prediction.lower() or label.split()[0] in prediction.lower():
                        wb["synthesis_consistent"] += 1

    report = {
        "summary": {
            "total_trials": len(all_trials),
            "result_trials": len(results),
            "control_trials": len(controls),
            "chance_rate": chance_rate,
        },
        "hit_rates_by_trial_type": hit_rates,
        "signal_strength": {
            "target_present_rate": tp_rate,
            "random_profile_rate": rp_rate,
            "difference": signal_strength,
            "interpretation": (
                "positive = real words beat random profiles"
                if signal_strength and signal_strength > 0
                else "no signal detected" if signal_strength is not None
                else "insufficient data"
            ),
        },
        "statistical_significance": {
            "test": "one-sided binomial",
            "hits": tp_hits,
            "trials": tp_total,
            "chance_rate": chance_rate,
            "p_value": p_value,
            "significant_at_0_05": p_value < 0.05 if tp_total > 0 else None,
            "note": f"Requires ~{_min_hits_for_significance(tp_total, chance_rate)} hits out of {tp_total} for p<0.05"
                    if tp_total > 0 else "no trials",
        },
        "synthesis_consistency": consistency,
        "per_word_breakdown": word_breakdown,
    }

    return report


def _min_hits_for_significance(
    n_trials: int, chance_rate: float, alpha: float = 0.05,
) -> int:
    """Find minimum hits needed for p < alpha."""
    if n_trials == 0:
        return 0
    for k in range(n_trials + 1):
        if binomial_p_value(k, n_trials, chance_rate) < alpha:
            return k
    return n_trials + 1


def print_report(report: Dict[str, Any]) -> None:
    """Print a human-readable evaluation report."""
    s = report["summary"]
    print("=" * 60)
    print("ERBEN BLIND ANALYSIS — EVALUATION REPORT")
    print("=" * 60)
    print(f"Total trials:   {s['total_trials']}")
    print(f"  Results:      {s['result_trials']}")
    print(f"  Controls:     {s['control_trials']}")
    print(f"Chance rate:    {s['chance_rate']:.1%}")
    print()

    # Hit rates by trial type
    print("--- Hit Rates by Trial Type ---")
    for trial_type, stats in report["hit_rates_by_trial_type"].items():
        print(f"\n  {trial_type}:")
        if stats["step1_total"] > 0:
            print(f"    Step 1 (consonant): {stats['step1_hits']}/{stats['step1_total']} = {stats['step1_rate']:.1%}")
        if stats["step2_total"] > 0:
            print(f"    Step 2 (full):      {stats['step2_hits']}/{stats['step2_total']} = {stats['step2_rate']:.1%}")
        if stats["unparsed_count"] > 0:
            print(f"    Unparsed responses: {stats['unparsed_count']}")

    # Signal strength
    print("\n--- Signal Strength ---")
    ss = report["signal_strength"]
    if ss["difference"] is not None:
        print(f"  Target-present rate: {ss['target_present_rate']:.1%}")
        print(f"  Random-profile rate: {ss['random_profile_rate']:.1%}")
        print(f"  Difference:          {ss['difference']:+.1%}")
        print(f"  Interpretation:      {ss['interpretation']}")
    else:
        print(f"  {ss['interpretation']}")

    # Statistical significance
    print("\n--- Statistical Significance ---")
    sig = report["statistical_significance"]
    if sig["trials"] > 0:
        print(f"  Binomial test: {sig['hits']}/{sig['trials']} hits, p = {sig['p_value']:.4f}")
        print(f"  Significant at p<0.05: {sig['significant_at_0_05']}")
        print(f"  {sig['note']}")
    else:
        print(f"  {sig['note']}")

    # Synthesis consistency
    print("\n--- Synthesis Consistency ---")
    sc = report["synthesis_consistency"]
    if sc["total"] > 0:
        print(f"  Step 3 aligns with Step 2: {sc['consistent_count']}/{sc['total']} = {sc['rate']:.1%}")
    else:
        print("  No data")

    # Per-word breakdown
    if report["per_word_breakdown"]:
        print("\n--- Per-Word Breakdown (target_present) ---")
        for word, wb in sorted(report["per_word_breakdown"].items()):
            s1 = f"{wb['step1_hits']}/{wb['runs']}"
            s2 = f"{wb['step2_hits']}/{wb['runs']}"
            synth = ""
            if wb.get("synthesis_total", 0) > 0:
                synth = f", synth_consistency={wb['synthesis_consistent']}/{wb['synthesis_total']}"
            print(f"  {word:12s} (target: {wb['target_domain']}): step1={s1}, step2={s2}{synth}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate blinded Erben trial results.")
    parser.add_argument("--output", "-o", default=None,
                        help="Save report as JSON to this path")

    args = parser.parse_args()

    results, controls = load_all_trials()

    if not results and not controls:
        print("No trials found in results/ or controls/.")
        print("Run some trials first with run_trial.py or run_batch.sh.")
        sys.exit(1)

    report = generate_report(results, controls)
    print_report(report)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
