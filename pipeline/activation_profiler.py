"""Map phoneme sequences to tiered Erben activation profiles.

For each phoneme, determines which macro-concepts it activates and at what
strength tier (strong/medium/weak), then organizes into consonant and vowel
layers with reinforcement and density analysis.
"""

from typing import Any, Dict, List, Literal, Optional, Set, TypedDict

from pipeline.config_loader import load_config
from pipeline.phoneme_decomposer import PhonemeToken


class ActivationEntry(TypedDict):
    macro_concept: str
    macro_concept_id: str
    tier: Literal["strong", "medium", "weak"]
    matching_sound_groups: list[str]


class PositionActivation(TypedDict):
    position: int
    ipa: str
    type: Literal["consonant", "vowel"]
    activations: list[ActivationEntry]


class ActivationProfile(TypedDict):
    consonant_layer: list[PositionActivation]
    vowel_layer: list[PositionActivation]
    reinforced_codes: list[str]
    activation_density: dict[str, int]
    strong_across_positions: list[str]
    metadata: dict[str, Any]


# Module-level caches built on first use
_cardinal_sets: Optional[Dict[str, Set[str]]] = None
_broad_groups: Optional[Set[str]] = None
_specific_groups: Optional[Set[str]] = None


def _ensure_strength_caches() -> None:
    """Build module-level caches for strength classification on first call."""
    global _cardinal_sets, _broad_groups, _specific_groups

    if _cardinal_sets is not None:
        return

    rules = load_config("activation_strength_rules.json")
    erben = load_config("erben_table.json")

    cardinal_map = rules["_cardinal_to_ipa_map"]
    _broad_groups = set(rules["broad_sound_groups"])
    _specific_groups = set(rules["specific_sound_groups"])

    # For each macro-concept, expand its primary_cardinal_sounds to actual IPA symbols
    _cardinal_sets = {}
    for mc_name, mc_data in erben["macro_concepts"].items():
        ipa_set: set[str] = set()
        for cardinal_label in mc_data["primary_cardinal_sounds"]:
            if cardinal_label in cardinal_map:
                ipa_set.update(cardinal_map[cardinal_label])
            else:
                # Cardinal label is already an IPA symbol (e.g., "r", "l")
                ipa_set.add(cardinal_label)
        _cardinal_sets[mc_name] = ipa_set


def _classify_activation_tier(
    ipa: str,
    matching_groups: set[str],
    macro_concept_name: str,
) -> Literal["strong", "medium", "weak"]:
    """Classify the strength tier of an activation.

    Priority: strong (primary cardinal) > medium (specific group) > weak (broad only).
    """
    _ensure_strength_caches()
    assert _cardinal_sets is not None
    assert _specific_groups is not None

    # Strong: phoneme is a primary cardinal sound for this macro-concept
    if ipa in _cardinal_sets[macro_concept_name]:
        return "strong"

    # Medium: overlap includes at least one specific sound group
    if matching_groups & _specific_groups:
        return "medium"

    # Weak: only broad groups matched
    return "weak"


def compute_macro_concept_activations(
    ipa: str,
    phoneme_sound_groups: list[str],
) -> list[ActivationEntry]:
    """Determine which macro-concepts a single phoneme activates, with tier classification.

    For each macro-concept in erben_table.json:
    1. Compute intersection of phoneme's sound_groups with macro-concept's sound_groups
    2. If empty → no activation
    3. Classify as strong/medium/weak per activation_strength_rules.json

    Args:
        ipa: The IPA symbol of the phoneme (e.g., "r", "uː").
        phoneme_sound_groups: The phoneme's sound_groups from phoneme_features.json.

    Returns:
        List of ActivationEntry dicts, sorted: strong first, then medium, then weak.
    """
    _ensure_strength_caches()
    erben = load_config("erben_table.json")

    phoneme_groups = set(phoneme_sound_groups)
    activations: list[ActivationEntry] = []

    tier_order = {"strong": 0, "medium": 1, "weak": 2}

    for mc_name, mc_data in erben["macro_concepts"].items():
        mc_groups = set(mc_data["sound_groups"])
        overlap = phoneme_groups & mc_groups

        if not overlap:
            continue

        tier = _classify_activation_tier(ipa, overlap, mc_name)
        activations.append(ActivationEntry(
            macro_concept=mc_name,
            macro_concept_id=mc_data["id"],
            tier=tier,
            matching_sound_groups=sorted(overlap),
        ))

    activations.sort(key=lambda a: (tier_order[a["tier"]], a["macro_concept"]))
    return activations


def compute_reinforcement(
    consonant_layer: list[PositionActivation],
    vowel_layer: list[PositionActivation],
) -> list[str]:
    """Find macro-concept codes activated in BOTH the consonant and vowel layers.

    Returns:
        Sorted list of MC-XX codes that appear in both layers.
    """
    def _collect_codes(layer: list[PositionActivation]) -> set[str]:
        codes: set[str] = set()
        for pos in layer:
            for act in pos["activations"]:
                codes.add(act["macro_concept_id"])
        return codes

    consonant_codes = _collect_codes(consonant_layer)
    vowel_codes = _collect_codes(vowel_layer)
    return sorted(consonant_codes & vowel_codes)


def compute_activation_density(
    all_positions: list[PositionActivation],
) -> dict[str, int]:
    """Count how many distinct positions activate each macro-concept.

    Returns:
        Dict mapping MC-XX code to count of positions where it appears.
    """
    density: dict[str, int] = {}
    for pos in all_positions:
        seen_in_position: set[str] = set()
        for act in pos["activations"]:
            code = act["macro_concept_id"]
            if code not in seen_in_position:
                seen_in_position.add(code)
                density[code] = density.get(code, 0) + 1
    return dict(sorted(density.items()))


def build_activation_profile(phonemes: list[PhonemeToken]) -> ActivationProfile:
    """Build a complete tiered activation profile from a phoneme sequence.

    For each phoneme, determines which macro-concepts it activates and at what
    strength tier, then organizes into consonant and vowel layers with
    reinforcement and density analysis.

    Args:
        phonemes: Ordered list of PhonemeToken dicts from phoneme_decomposer.

    Returns:
        Complete ActivationProfile with layers, reinforcement, and density.
    """
    features = load_config("phoneme_features.json")

    consonant_layer: list[PositionActivation] = []
    vowel_layer: list[PositionActivation] = []

    for phoneme in phonemes:
        ipa = phoneme["ipa"]
        ptype = phoneme["type"]

        # Look up sound groups from phoneme_features.json
        section = "consonants" if ptype == "consonant" else "vowels"
        phoneme_data = features[section].get(ipa)
        if phoneme_data is None:
            raise ValueError(f"IPA '{ipa}' not found in phoneme_features.json {section}")

        sound_groups = phoneme_data["sound_groups"]
        activations = compute_macro_concept_activations(ipa, sound_groups)

        pos_act = PositionActivation(
            position=phoneme["position"],
            ipa=ipa,
            type=ptype,
            activations=activations,
        )

        if ptype == "consonant":
            consonant_layer.append(pos_act)
        else:
            vowel_layer.append(pos_act)

    all_positions = consonant_layer + vowel_layer
    density = compute_activation_density(all_positions)
    reinforced = compute_reinforcement(consonant_layer, vowel_layer)

    # Codes with strong activation at 2+ positions
    strong_counts: dict[str, int] = {}
    for pos in all_positions:
        seen: set[str] = set()
        for act in pos["activations"]:
            if act["tier"] == "strong" and act["macro_concept_id"] not in seen:
                seen.add(act["macro_concept_id"])
                strong_counts[act["macro_concept_id"]] = strong_counts.get(act["macro_concept_id"], 0) + 1
    strong_across = sorted(code for code, count in strong_counts.items() if count >= 2)

    return ActivationProfile(
        consonant_layer=consonant_layer,
        vowel_layer=vowel_layer,
        reinforced_codes=reinforced,
        activation_density=density,
        strong_across_positions=strong_across,
        metadata={
            "phoneme_count": len(phonemes),
            "consonant_count": len(consonant_layer),
            "vowel_count": len(vowel_layer),
        },
    )


if __name__ == "__main__":
    from pipeline.phoneme_decomposer import decompose_word

    TIER_NOTATION = {"strong": "***", "medium": "**", "weak": "*"}

    word = "rút"
    phonemes = decompose_word(word)
    profile = build_activation_profile(phonemes)

    print(f"=== Activation Profile for '{word}' ===\n")

    for layer_name in ("consonant_layer", "vowel_layer"):
        label = "CONSONANT LAYER" if "consonant" in layer_name else "VOWEL LAYER"
        print(f"--- {label} ---")
        for pos in profile[layer_name]:  # type: ignore[literal-required]
            acts_str = ", ".join(
                f"{a['macro_concept_id']}{TIER_NOTATION[a['tier']]} ({a['macro_concept']})"
                for a in pos["activations"]
            )
            print(f"  Position {pos['position']} ({pos['ipa']}): {acts_str}")
        print()

    print(f"Reinforced codes: {profile['reinforced_codes']}")
    print(f"Activation density: {profile['activation_density']}")
    print(f"Strong at 2+ positions: {profile['strong_across_positions']}")
    print(f"Metadata: {profile['metadata']}")
