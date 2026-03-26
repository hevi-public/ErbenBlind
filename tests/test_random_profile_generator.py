"""Tests for the random profile generator module."""

import random
import unittest

from pipeline.random_profile_generator import generate_random_profile


class TestProfileStructure(unittest.TestCase):
    """Random profiles must have the same structure as real ones."""

    def setUp(self):
        self.profile = generate_random_profile(
            num_consonant_positions=2,
            num_vowel_positions=1,
            rng=random.Random(42),
        )

    def test_has_consonant_layer(self):
        self.assertEqual(len(self.profile["consonant_layer"]), 2)

    def test_has_vowel_layer(self):
        self.assertEqual(len(self.profile["vowel_layer"]), 1)

    def test_has_reinforced_codes(self):
        self.assertIsInstance(self.profile["reinforced_codes"], list)

    def test_has_activation_density(self):
        self.assertIsInstance(self.profile["activation_density"], dict)

    def test_metadata_is_random(self):
        self.assertTrue(self.profile["metadata"]["is_random"])

    def test_metadata_counts(self):
        self.assertEqual(self.profile["metadata"]["phoneme_count"], 3)
        self.assertEqual(self.profile["metadata"]["consonant_count"], 2)
        self.assertEqual(self.profile["metadata"]["vowel_count"], 1)


class TestActivationEntries(unittest.TestCase):
    """Each position should have valid activation entries."""

    def setUp(self):
        self.profile = generate_random_profile(
            num_consonant_positions=2,
            num_vowel_positions=2,
            rng=random.Random(42),
        )

    def test_activations_have_required_fields(self):
        for layer in ("consonant_layer", "vowel_layer"):
            for pos in self.profile[layer]:  # type: ignore[literal-required]
                for act in pos["activations"]:
                    self.assertIn("macro_concept", act)
                    self.assertIn("macro_concept_id", act)
                    self.assertIn("tier", act)
                    self.assertIn("matching_sound_groups", act)

    def test_tiers_are_valid(self):
        valid_tiers = {"strong", "medium", "weak"}
        for layer in ("consonant_layer", "vowel_layer"):
            for pos in self.profile[layer]:  # type: ignore[literal-required]
                for act in pos["activations"]:
                    self.assertIn(act["tier"], valid_tiers)

    def test_activations_sorted_by_tier(self):
        tier_order = {"strong": 0, "medium": 1, "weak": 2}
        for layer in ("consonant_layer", "vowel_layer"):
            for pos in self.profile[layer]:  # type: ignore[literal-required]
                tiers = [tier_order[a["tier"]] for a in pos["activations"]]
                self.assertEqual(tiers, sorted(tiers))

    def test_mc_ids_are_valid(self):
        from pipeline.config_loader import load_config
        erben = load_config("erben_table.json")
        valid_ids = {mc["id"] for mc in erben["macro_concepts"].values()}
        for layer in ("consonant_layer", "vowel_layer"):
            for pos in self.profile[layer]:  # type: ignore[literal-required]
                for act in pos["activations"]:
                    self.assertIn(act["macro_concept_id"], valid_ids)


class TestReproducibility(unittest.TestCase):
    """Same seed should produce identical profiles."""

    def test_same_seed_same_output(self):
        p1 = generate_random_profile(rng=random.Random(123))
        p2 = generate_random_profile(rng=random.Random(123))
        # Compare activation IDs at each position
        for layer in ("consonant_layer", "vowel_layer"):
            for pos1, pos2 in zip(p1[layer], p2[layer]):  # type: ignore[literal-required]
                ids1 = [a["macro_concept_id"] for a in pos1["activations"]]
                ids2 = [a["macro_concept_id"] for a in pos2["activations"]]
                self.assertEqual(ids1, ids2)

    def test_different_seed_different_output(self):
        p1 = generate_random_profile(rng=random.Random(1))
        p2 = generate_random_profile(rng=random.Random(2))
        ids1 = [a["macro_concept_id"]
                for pos in p1["consonant_layer"] for a in pos["activations"]]
        ids2 = [a["macro_concept_id"]
                for pos in p2["consonant_layer"] for a in pos["activations"]]
        self.assertNotEqual(ids1, ids2)


class TestVariableStructure(unittest.TestCase):
    """Test with different position counts."""

    def test_single_consonant_single_vowel(self):
        p = generate_random_profile(
            num_consonant_positions=1, num_vowel_positions=1,
            rng=random.Random(42),
        )
        self.assertEqual(len(p["consonant_layer"]), 1)
        self.assertEqual(len(p["vowel_layer"]), 1)

    def test_three_consonants_two_vowels(self):
        p = generate_random_profile(
            num_consonant_positions=3, num_vowel_positions=2,
            rng=random.Random(42),
        )
        self.assertEqual(len(p["consonant_layer"]), 3)
        self.assertEqual(len(p["vowel_layer"]), 2)
        self.assertEqual(p["metadata"]["phoneme_count"], 5)


if __name__ == "__main__":
    unittest.main()
