"""Select forced-choice semantic domain options for blinded trials.

Controls which domains appear as options based on trial type:
- target_present: target IS one of the options
- target_absent: target deliberately EXCLUDED
- null_trial: all options semantically distant from target
- random_profile: any options (no real target)
"""

import random
from typing import Any, Dict, List, Literal, Optional

from pipeline.config_loader import load_config


TrialType = Literal["target_present", "target_absent", "null_trial", "random_profile"]

# Manually defined semantic adjacency: domains that are conceptually close.
# Each key maps to a set of domain IDs that are "adjacent" (sharing semantic overlap).
# This is deliberately conservative — only clear overlaps are marked.
ADJACENCY: Dict[str, List[str]] = {
    "SD-01": ["SD-04"],              # WATER ↔ AIR (fluidity)
    "SD-03": ["SD-09"],              # FIRE ↔ VISION (light/brightness)
    "SD-04": ["SD-01", "SD-22"],     # AIR ↔ WATER, EXPELLING (blowing)
    "SD-05": ["SD-02"],              # GROWTH ↔ EARTH
    "SD-02": ["SD-05"],              # EARTH ↔ GROWTH
    "SD-06": ["SD-07"],              # ANIMAL ↔ BODY
    "SD-07": ["SD-06", "SD-11"],     # BODY ↔ ANIMAL, TOUCH
    "SD-08": ["SD-20"],              # MOVEMENT ↔ ROTATION
    "SD-09": ["SD-03"],              # VISION ↔ FIRE
    "SD-11": ["SD-07", "SD-13"],     # TOUCH ↔ BODY, CUTTING
    "SD-12": ["SD-23"],              # SIZE ↔ SWELLING
    "SD-13": ["SD-11"],              # CUTTING ↔ TOUCH
    "SD-15": ["SD-17"],              # DECAY ↔ FOOD (rot/taste)
    "SD-16": ["SD-19"],              # KINSHIP ↔ EMOTION
    "SD-17": ["SD-15"],              # FOOD ↔ DECAY
    "SD-19": ["SD-16"],              # EMOTION ↔ KINSHIP
    "SD-20": ["SD-08"],              # ROTATION ↔ MOVEMENT
    "SD-22": ["SD-04"],              # EXPELLING ↔ AIR
    "SD-23": ["SD-12"],              # SWELLING ↔ SIZE
    "SD-24": ["SD-08"],              # STEALING ↔ MOVEMENT (sneaking)
}


def _get_adjacent_ids(domain_id: str) -> set:
    """Get domain IDs that are semantically adjacent to the given domain."""
    return set(ADJACENCY.get(domain_id, []))


def _get_distant_ids(domain_id: str, all_domains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get domains that are semantically distant from the given domain.

    Distant means: not the target itself, and not in its adjacency set.
    """
    adjacent = _get_adjacent_ids(domain_id)
    return [d for d in all_domains if d["id"] != domain_id and d["id"] not in adjacent]


def select_options(
    target_domain_id: str,
    trial_type: TrialType,
    num_options: int = 4,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    """Select forced-choice semantic domain options for a trial.

    The selection strategy depends on trial_type:
    - target_present: target included, rest are non-adjacent distractors
    - target_absent: target excluded, mix of adjacent and distant domains
    - null_trial: all options are semantically distant from target
    - random_profile: random selection from all domains (no real target)

    Args:
        target_domain_id: The correct semantic domain ID (e.g., "SD-15").
            Ignored for random_profile trials.
        trial_type: One of the four trial types.
        num_options: Number of options to present (default 4).
        rng: Optional Random instance for reproducibility. If None, uses
            module-level random.

    Returns:
        Shuffled list of semantic domain dicts, each with 'id', 'label',
        and 'description_for_evaluation'.

    Raises:
        ValueError: If target_domain_id is not found or not enough domains available.
    """
    if rng is None:
        rng = random.Random()

    domains_config = load_config("semantic_domains.json")
    all_domains = domains_config["domains"]

    # Lookup target domain
    target_domain = None
    for d in all_domains:
        if d["id"] == target_domain_id:
            target_domain = d
            break

    if target_domain is None and trial_type != "random_profile":
        raise ValueError(f"Target domain '{target_domain_id}' not found in semantic_domains.json")

    if trial_type == "target_present":
        # Include target, fill rest with non-adjacent domains
        non_adjacent = _get_distant_ids(target_domain_id, all_domains)
        if len(non_adjacent) < num_options - 1:
            raise ValueError(f"Not enough distant domains for {num_options} options")
        distractors = rng.sample(non_adjacent, num_options - 1)
        options = [target_domain] + distractors

    elif trial_type == "target_absent":
        # Exclude target, include some adjacent and some distant
        all_except_target = [d for d in all_domains if d["id"] != target_domain_id]
        adjacent = [d for d in all_domains if d["id"] in _get_adjacent_ids(target_domain_id)]
        distant = _get_distant_ids(target_domain_id, all_domains)

        # Try to include 1 adjacent if available, rest distant
        options: List[Dict[str, Any]] = []
        if adjacent:
            options.append(rng.choice(adjacent))
            remaining_pool = [d for d in distant if d["id"] not in {o["id"] for o in options}]
            needed = num_options - len(options)
            if len(remaining_pool) < needed:
                remaining_pool = [d for d in all_except_target if d["id"] not in {o["id"] for o in options}]
            options.extend(rng.sample(remaining_pool, needed))
        else:
            options = rng.sample(all_except_target, num_options)

    elif trial_type == "null_trial":
        # All options are semantically distant from target
        distant = _get_distant_ids(target_domain_id, all_domains)
        if len(distant) < num_options:
            raise ValueError(f"Not enough distant domains for null trial with {num_options} options")
        options = rng.sample(distant, num_options)

    elif trial_type == "random_profile":
        # Any options — there's no real target
        options = rng.sample(all_domains, num_options)

    else:
        raise ValueError(f"Unknown trial_type: {trial_type}")

    rng.shuffle(options)
    return options


if __name__ == "__main__":
    rng = random.Random(42)

    for trial_type in ("target_present", "target_absent", "null_trial", "random_profile"):
        print(f"--- {trial_type} (target: SD-15 DECAY_DETERIORATION) ---")
        options = select_options("SD-15", trial_type, num_options=4, rng=rng)  # type: ignore[arg-type]
        for opt in options:
            print(f"  {opt['id']}: {opt['label']}")
        target_ids = [o["id"] for o in options]
        print(f"  Target present: {'SD-15' in target_ids}")
        print()
