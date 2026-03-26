"""Generate random activation profiles for baseline control trials.

Random profiles establish the false-positive rate: if the LLM finds "coherent"
interpretations in random noise at the same rate as real words, the framework
isn't detecting signal.
"""

import random
from typing import Any, Dict, List, Literal, Optional

from pipeline.config_loader import load_config
from pipeline.activation_profiler import (
    ActivationEntry,
    ActivationProfile,
    PositionActivation,
    compute_reinforcement,
    compute_activation_density,
)


def _random_activations(
    num_activations: int,
    macro_concepts: Dict[str, Any],
    rng: random.Random,
) -> List[ActivationEntry]:
    """Generate a random set of activations with plausible tier distribution.

    Tier distribution approximates real words: ~25% strong, ~30% medium, ~45% weak.
    """
    mc_names = list(macro_concepts.keys())
    selected = rng.sample(mc_names, min(num_activations, len(mc_names)))

    tier_weights = [("strong", 0.25), ("medium", 0.30), ("weak", 0.45)]
    tiers_list = [t for t, _ in tier_weights]
    weights = [w for _, w in tier_weights]

    tier_order = {"strong": 0, "medium": 1, "weak": 2}

    activations: List[ActivationEntry] = []
    for mc_name in selected:
        tier: Literal["strong", "medium", "weak"] = rng.choices(tiers_list, weights=weights, k=1)[0]  # type: ignore[assignment]
        activations.append(ActivationEntry(
            macro_concept=mc_name,
            macro_concept_id=macro_concepts[mc_name]["id"],
            tier=tier,
            matching_sound_groups=["synthetic"],
        ))

    activations.sort(key=lambda a: (tier_order[a["tier"]], a["macro_concept"]))
    return activations


def generate_random_profile(
    num_consonant_positions: int = 2,
    num_vowel_positions: int = 1,
    activations_per_position: tuple = (4, 8),
    rng: Optional[random.Random] = None,
) -> ActivationProfile:
    """Generate a random activation profile that mimics real word structure.

    The profile has the same shape as a real one (consonant/vowel layers,
    reinforcement, density) but with randomly assigned macro-concept activations.

    Args:
        num_consonant_positions: Number of consonant positions (default 2).
        num_vowel_positions: Number of vowel positions (default 1).
        activations_per_position: (min, max) range for activations per position.
        rng: Optional Random instance for reproducibility.

    Returns:
        A synthetic ActivationProfile with is_random=True in metadata.
    """
    if rng is None:
        rng = random.Random()

    erben = load_config("erben_table.json")
    macro_concepts = erben["macro_concepts"]

    min_act, max_act = activations_per_position
    position = 1

    consonant_layer: List[PositionActivation] = []
    for _ in range(num_consonant_positions):
        num_act = rng.randint(min_act, max_act)
        consonant_layer.append(PositionActivation(
            position=position,
            ipa="?",
            type="consonant",
            activations=_random_activations(num_act, macro_concepts, rng),
        ))
        position += 1

    vowel_layer: List[PositionActivation] = []
    for _ in range(num_vowel_positions):
        num_act = rng.randint(min_act, max_act)
        vowel_layer.append(PositionActivation(
            position=position,
            ipa="?",
            type="vowel",
            activations=_random_activations(num_act, macro_concepts, rng),
        ))
        position += 1

    # Interleave positions to mimic real CVCV structure
    # Reassign positions in a plausible order: C1 V1 C2 V2 ...
    all_pos = []
    ci, vi = 0, 0
    for i in range(num_consonant_positions + num_vowel_positions):
        if i % 2 == 0 and ci < num_consonant_positions:
            all_pos.append(consonant_layer[ci])
            ci += 1
        elif vi < num_vowel_positions:
            all_pos.append(vowel_layer[vi])
            vi += 1
        elif ci < num_consonant_positions:
            all_pos.append(consonant_layer[ci])
            ci += 1

    for idx, pos in enumerate(all_pos):
        pos["position"] = idx + 1

    all_positions = consonant_layer + vowel_layer
    reinforced = compute_reinforcement(consonant_layer, vowel_layer)
    density = compute_activation_density(all_positions)

    strong_counts: Dict[str, int] = {}
    for pos in all_positions:
        seen: set = set()
        for act in pos["activations"]:
            if act["tier"] == "strong" and act["macro_concept_id"] not in seen:
                seen.add(act["macro_concept_id"])
                strong_counts[act["macro_concept_id"]] = strong_counts.get(act["macro_concept_id"], 0) + 1
    strong_across = sorted(code for code, count in strong_counts.items() if count >= 2)

    total = num_consonant_positions + num_vowel_positions
    return ActivationProfile(
        consonant_layer=consonant_layer,
        vowel_layer=vowel_layer,
        reinforced_codes=reinforced,
        activation_density=density,
        strong_across_positions=strong_across,
        metadata={
            "phoneme_count": total,
            "consonant_count": num_consonant_positions,
            "vowel_count": num_vowel_positions,
            "is_random": True,
        },
    )


if __name__ == "__main__":
    rng = random.Random(42)

    TIER_NOTATION = {"strong": "***", "medium": "**", "weak": "*"}

    profile = generate_random_profile(
        num_consonant_positions=2,
        num_vowel_positions=1,
        rng=rng,
    )

    print("=== Random Activation Profile ===\n")

    for layer_name in ("consonant_layer", "vowel_layer"):
        label = "CONSONANT LAYER" if "consonant" in layer_name else "VOWEL LAYER"
        print(f"--- {label} ---")
        for pos in profile[layer_name]:  # type: ignore[literal-required]
            acts_str = ", ".join(
                f"{a['macro_concept_id']}{TIER_NOTATION[a['tier']]} ({a['macro_concept']})"
                for a in pos["activations"]
            )
            print(f"  Position {pos['position']}: {acts_str}")
        print()

    print(f"Reinforced: {profile['reinforced_codes']}")
    print(f"Density: {profile['activation_density']}")
    print(f"Metadata: {profile['metadata']}")
