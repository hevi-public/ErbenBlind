"""Format activation profiles into blinded prompts for the claude CLI.

Steps 1-2 use MC-XX codes (blinded). Step 3 decodes to human-readable labels.
Each function returns the user-message content; the system prompt is returned
separately by get_system_prompt().
"""

from typing import Any

from pipeline.config_loader import load_config
from pipeline.activation_profiler import ActivationProfile, PositionActivation


TIER_NOTATION = {"strong": "***", "medium": "**", "weak": "*"}
TIER_LABEL = {"strong": "strong", "medium": "medium", "weak": "weak"}


_REASONING_SYSTEM_PROMPT_STEPS12 = (
    "You are a pattern matching analyst. You compare abstract feature profiles against category options.\n\n"
    "Your method:\n"
    "1. List each activation code from the profile\n"
    "2. For EACH category option, count how many activation codes could relate to it\n"
    "3. Show your counts\n"
    "4. Pick the category with the highest count\n\n"
    "Always show your counting work before stating your final choice. "
    "End your response with a line that says \"CHOICE: [your pick]\" using the exact category label."
)

_REASONING_SYSTEM_PROMPT_STEP3 = (
    "You are a semantic analyst. You have been given a set of conceptual features activated by an unknown word.\n\n"
    "Your method:\n"
    "1. Group the features that point in similar semantic directions\n"
    "2. Identify the largest coherent cluster\n"
    "3. Ask: what kind of thing or concept sits at the intersection of this cluster?\n"
    "4. Commit to a single specific prediction\n\n"
    "Show your grouping work, then state your prediction."
)


def get_system_prompt(decode_all_steps: bool = False, reasoning_prompt: bool = False) -> str:
    """Return the system prompt for blinded LLM calls (steps 1-2).

    When reasoning_prompt=True, returns a structured counting-based prompt
    designed for models that struggle with gestalt synthesis.
    """
    if reasoning_prompt:
        return _REASONING_SYSTEM_PROMPT_STEPS12
    templates = load_config("prompt_templates.json")
    key = "system_prompt_decoded" if decode_all_steps else "system_prompt"
    return templates[key]


def get_step3_system_prompt(reasoning_prompt: bool = False) -> str:
    """Return the system prompt for Step 3 (synthesis).

    Separate from steps 1-2 because the reasoning prompt uses a different
    instruction style for the open-ended synthesis step.
    """
    if reasoning_prompt:
        return _REASONING_SYSTEM_PROMPT_STEP3
    templates = load_config("prompt_templates.json")
    return templates["system_prompt"]


def _format_position_activations_coded(
    positions: list[PositionActivation],
) -> str:
    """Format positional activations using MC-XX codes with tier notation.

    Output format:
      Position 1: MC-10***, MC-05***, MC-08**, MC-16*, MC-19*
    """
    lines: list[str] = []
    for pos in positions:
        acts = ", ".join(
            f"{a['macro_concept_id']}{TIER_NOTATION[a['tier']]}"
            for a in pos["activations"]
        )
        lines.append(f"Position {pos['position']}: {acts}")
    return "\n".join(lines)


def _format_position_activations_decoded(
    positions: list[PositionActivation],
) -> str:
    """Format positional activations using human-readable labels (for Step 3).

    Output format:
      Position 1: TURN(strong), UNEVEN(strong), TONGUE(medium), MOTHER(weak)
    """
    lines: list[str] = []
    for pos in positions:
        acts = ", ".join(
            f"{a['macro_concept']}({TIER_LABEL[a['tier']]})"
            for a in pos["activations"]
        )
        lines.append(f"Position {pos['position']}: {acts}")
    return "\n".join(lines)


def _format_position_activations_decoded_tiered(
    positions: list[PositionActivation],
) -> str:
    """Format positional activations using human-readable labels with tier notation.

    Used for decode_all_steps mode in Steps 1-2.

    Output format:
      Position 1: TURN***, UNEVEN***, TONGUE**, MOTHER*
    """
    lines: list[str] = []
    for pos in positions:
        acts = ", ".join(
            f"{a['macro_concept']}{TIER_NOTATION[a['tier']]}"
            for a in pos["activations"]
        )
        lines.append(f"Position {pos['position']}: {acts}")
    return "\n".join(lines)


def _format_combined_activations_decoded(profile: ActivationProfile) -> str:
    """Format all positions (both layers) with decoded labels, indicating layer.

    Used for decode_all_steps mode in Step 2.

    Output format:
      Position 1 [consonant]: TURN***, UNEVEN***
      Position 2 [vowel]: ROUNDNESS***, DEEP***
    """
    all_positions: list[tuple[int, str, PositionActivation]] = []
    for pos in profile["consonant_layer"]:
        all_positions.append((pos["position"], "consonant", pos))
    for pos in profile["vowel_layer"]:
        all_positions.append((pos["position"], "vowel", pos))
    all_positions.sort(key=lambda x: x[0])

    lines: list[str] = []
    for _, layer_type, pos in all_positions:
        acts = ", ".join(
            f"{a['macro_concept']}{TIER_NOTATION[a['tier']]}"
            for a in pos["activations"]
        )
        lines.append(f"Position {pos['position']} [{layer_type}]: {acts}")
    return "\n".join(lines)


def _format_forced_choice_options(options: list[dict[str, Any]]) -> str:
    """Format semantic domain options as a lettered list.

    Output format:
      A) WATER_LIQUID — Water, liquids, flowing, wet, rain, river
      B) DECAY_DETERIORATION — Decay, rot, deterioration, ugly, foul
    """
    letters = "ABCDEFGH"
    lines: list[str] = []
    for i, opt in enumerate(options):
        letter = letters[i] if i < len(letters) else str(i + 1)
        label = opt.get("label", opt.get("id", "???"))
        desc = opt.get("description_for_evaluation", "")
        if desc:
            lines.append(f"{letter}) {label} — {desc}")
        else:
            lines.append(f"{letter}) {label}")
    return "\n".join(lines)


def _format_reinforcement_summary(profile: ActivationProfile) -> str:
    """Format the reinforcement codes and density for prompt insertion."""
    codes = load_config("concept_codes.json")
    code_to_label = codes["code_to_label"]

    parts: list[str] = []
    for mc_code in profile["reinforced_codes"]:
        label = code_to_label.get(mc_code, mc_code)
        count = profile["activation_density"].get(mc_code, 0)
        parts.append(f"{label} ({mc_code}): reinforced, {count} positions")

    return "; ".join(parts) if parts else "No reinforcement detected"


def _format_combined_activations_coded(profile: ActivationProfile) -> str:
    """Format all positions (both layers) with coded labels, indicating layer.

    Output format:
      Position 1 [consonant]: MC-10***, MC-05***
      Position 2 [vowel]: MC-06***, MC-12***
    """
    all_positions: list[tuple[int, str, PositionActivation]] = []
    for pos in profile["consonant_layer"]:
        all_positions.append((pos["position"], "consonant", pos))
    for pos in profile["vowel_layer"]:
        all_positions.append((pos["position"], "vowel", pos))
    all_positions.sort(key=lambda x: x[0])

    lines: list[str] = []
    for _, layer_type, pos in all_positions:
        acts = ", ".join(
            f"{a['macro_concept_id']}{TIER_NOTATION[a['tier']]}"
            for a in pos["activations"]
        )
        lines.append(f"Position {pos['position']} [{layer_type}]: {acts}")
    return "\n".join(lines)


def format_step1_prompt(
    profile: ActivationProfile,
    forced_choice_options: list[dict[str, Any]],
    decode_all_steps: bool = False,
) -> str:
    """Format the Step 1 (consonant frame) prompt for the blinded LLM.

    Uses MC-XX codes by default. When decode_all_steps=True, uses human-readable
    concept labels with tier notation instead.

    Args:
        profile: Full activation profile from activation_profiler.
        forced_choice_options: List of semantic domain dicts with 'id', 'label',
            and optionally 'description_for_evaluation'.
        decode_all_steps: If True, show concept names instead of MC-XX codes.

    Returns:
        Complete formatted prompt string.
    """
    templates = load_config("prompt_templates.json")
    template = templates["step_1_consonant_frame"]["template"]

    if decode_all_steps:
        consonant_activations = _format_position_activations_decoded_tiered(profile["consonant_layer"])
    else:
        consonant_activations = _format_position_activations_coded(profile["consonant_layer"])
    options_text = _format_forced_choice_options(forced_choice_options)

    return template.format(
        consonant_activations=consonant_activations,
        forced_choice_options=options_text,
    )


def format_step2_prompt(
    profile: ActivationProfile,
    forced_choice_options: list[dict[str, Any]],
    step1_response: str,
    step1_choice: str,
    decode_all_steps: bool = False,
) -> str:
    """Format the Step 2 (vowel fill) prompt.

    Includes quoted Step 1 response as prior commitment. Uses MC-XX codes by
    default; when decode_all_steps=True, shows human-readable concept labels.

    Args:
        profile: Full activation profile.
        forced_choice_options: Semantic domain options for this step.
        step1_response: The full text response from Step 1.
        step1_choice: The semantic domain chosen in Step 1 (e.g., "SD-15").
        decode_all_steps: If True, show concept names instead of MC-XX codes.

    Returns:
        Complete formatted prompt string.
    """
    templates = load_config("prompt_templates.json")
    template = templates["step_2_vowel_fill"]["template"]

    if decode_all_steps:
        codes = load_config("concept_codes.json")
        code_to_label = codes["code_to_label"]
        vowel_activations = _format_position_activations_decoded_tiered(profile["vowel_layer"])
        combined_activations = _format_combined_activations_decoded(profile)
        reinforced = ", ".join(
            code_to_label.get(c, c) for c in profile["reinforced_codes"]
        ) if profile["reinforced_codes"] else "None"
    else:
        vowel_activations = _format_position_activations_coded(profile["vowel_layer"])
        combined_activations = _format_combined_activations_coded(profile)
        reinforced = ", ".join(profile["reinforced_codes"]) if profile["reinforced_codes"] else "None"
    options_text = _format_forced_choice_options(forced_choice_options)

    return template.format(
        step_1_response=step1_response,
        step_1_choice=step1_choice,
        vowel_activations=vowel_activations,
        combined_activations=combined_activations,
        reinforced_codes=reinforced,
        forced_choice_options=options_text,
    )


def format_step3_prompt(
    profile: ActivationProfile,
    step1_response: str,
    step1_choice: str,
    step2_response: str,
    step2_choice: str,
) -> str:
    """Format the Step 3 (synthesis) prompt with decoded human-readable labels.

    This is the only step where macro-concept names are visible. No forced-choice
    options — this step is open-ended.

    Args:
        profile: Full activation profile.
        step1_response: Full text response from Step 1.
        step1_choice: Domain chosen in Step 1.
        step2_response: Full text response from Step 2.
        step2_choice: Domain chosen in Step 2.

    Returns:
        Complete formatted prompt string.
    """
    templates = load_config("prompt_templates.json")
    codes = load_config("concept_codes.json")
    code_to_label = codes["code_to_label"]
    template = templates["step_3_synthesis"]["template"]

    all_positions = profile["consonant_layer"] + profile["vowel_layer"]
    all_positions_sorted = sorted(all_positions, key=lambda p: p["position"])
    decoded_profile = _format_position_activations_decoded(all_positions_sorted)

    # Strong activations: codes with density >= 2 at strong tier
    strong_counts: dict[str, int] = {}
    for pos in all_positions:
        seen: set[str] = set()
        for act in pos["activations"]:
            if act["tier"] == "strong" and act["macro_concept_id"] not in seen:
                seen.add(act["macro_concept_id"])
                strong_counts[act["macro_concept_id"]] = strong_counts.get(act["macro_concept_id"], 0) + 1
    strong_acts = sorted(
        code for code, count in strong_counts.items() if count >= 2
    )
    strong_text = ", ".join(
        f"{code_to_label.get(c, c)} ({c})" for c in strong_acts
    ) if strong_acts else "None"

    # Weak/unique activations: codes at only 1 position total
    density = profile["activation_density"]
    weak_acts = sorted(code for code, count in density.items() if count == 1)
    weak_text = ", ".join(
        f"{code_to_label.get(c, c)} ({c})" for c in weak_acts
    ) if weak_acts else "None"

    reinforcement = _format_reinforcement_summary(profile)

    return template.format(
        step_1_choice=step1_choice,
        step_2_choice=step2_choice,
        step_1_response=step1_response,
        step_2_response=step2_response,
        decoded_full_profile=decoded_profile,
        strong_activations=strong_text,
        weak_activations=weak_text,
        reinforcement_summary=reinforcement,
    )


if __name__ == "__main__":
    from pipeline.phoneme_decomposer import decompose_word
    from pipeline.activation_profiler import build_activation_profile

    word = "rút"
    phonemes = decompose_word(word)
    profile = build_activation_profile(phonemes)

    # Dummy forced-choice options
    dummy_options = [
        {"id": "SD-15", "label": "DECAY_DETERIORATION", "description_for_evaluation": "Decay, rot, deterioration, ugly, foul, broken"},
        {"id": "SD-01", "label": "WATER_LIQUID", "description_for_evaluation": "Water, liquids, flowing, wet, rain, river"},
        {"id": "SD-03", "label": "FIRE_HEAT", "description_for_evaluation": "Fire, heat, burning, warmth, cooking, sun"},
        {"id": "SD-20", "label": "ROTATION_TWISTING", "description_for_evaluation": "Turning, twisting, spinning, circular motion"},
    ]

    print("=" * 60)
    print("SYSTEM PROMPT:")
    print("=" * 60)
    print(get_system_prompt())
    print()

    print("=" * 60)
    print("STEP 1 PROMPT (consonant frame, coded):")
    print("=" * 60)
    prompt1 = format_step1_prompt(profile, dummy_options)
    print(prompt1)
    print()

    # Simulate step 1 response
    fake_step1 = "The consonant frame shows strong activation in codes associated with positions 1 and 3. I choose DECAY_DETERIORATION."

    print("=" * 60)
    print("STEP 2 PROMPT (vowel fill, coded):")
    print("=" * 60)
    prompt2 = format_step2_prompt(profile, dummy_options, fake_step1, "SD-15 (DECAY_DETERIORATION)")
    print(prompt2)
    print()

    # Simulate step 2 response
    fake_step2 = "The vowel data reinforces the consonant reading. I choose DECAY_DETERIORATION."

    print("=" * 60)
    print("STEP 3 PROMPT (synthesis, decoded):")
    print("=" * 60)
    prompt3 = format_step3_prompt(
        profile, fake_step1, "SD-15 (DECAY_DETERIORATION)",
        fake_step2, "SD-15 (DECAY_DETERIORATION)",
    )
    print(prompt3)
