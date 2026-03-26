"""Tests for the activation profiler module."""

import unittest

from pipeline.phoneme_decomposer import decompose_word
from pipeline.activation_profiler import (
    build_activation_profile,
    compute_macro_concept_activations,
    compute_reinforcement,
    compute_activation_density,
)
from pipeline.config_loader import load_config


class TestTierClassification(unittest.TestCase):
    """Verify strong/medium/weak tier assignment follows the rules."""

    def _get_activations_for(self, ipa: str):
        """Helper: get activations for an IPA phoneme."""
        features = load_config("phoneme_features.json")
        for section in ("consonants", "vowels"):
            if ipa in features.get(section, {}):
                return compute_macro_concept_activations(
                    ipa, features[section][ipa]["sound_groups"]
                )
        raise ValueError(f"IPA '{ipa}' not found")

    def _find_activation(self, activations, mc_name):
        """Helper: find a specific macro-concept in activations list."""
        for a in activations:
            if a["macro_concept"] == mc_name:
                return a
        return None

    def test_r_turn_is_strong(self):
        """/r/ is a primary cardinal sound for TURN → strong."""
        acts = self._get_activations_for("r")
        turn = self._find_activation(acts, "TURN")
        self.assertIsNotNone(turn)
        self.assertEqual(turn["tier"], "strong")

    def test_r_uneven_is_strong(self):
        """/r/ is a primary cardinal sound for UNEVEN → strong."""
        acts = self._get_activations_for("r")
        uneven = self._find_activation(acts, "UNEVEN")
        self.assertIsNotNone(uneven)
        self.assertEqual(uneven["tier"], "strong")

    def test_r_tongue_is_medium(self):
        """/r/ matches alveolar_voiced (specific) for TONGUE but l is primary → medium."""
        acts = self._get_activations_for("r")
        tongue = self._find_activation(acts, "TONGUE")
        self.assertIsNotNone(tongue)
        self.assertEqual(tongue["tier"], "medium")

    def test_r_mother_is_weak(self):
        """/r/ matches only broad 'voiced' for MOTHER → weak."""
        acts = self._get_activations_for("r")
        mother = self._find_activation(acts, "MOTHER")
        self.assertIsNotNone(mother)
        self.assertEqual(mother["tier"], "weak")

    def test_u_long_roundness_is_strong(self):
        """/uː/ is primary cardinal for ROUNDNESS → strong."""
        acts = self._get_activations_for("uː")
        roundness = self._find_activation(acts, "ROUNDNESS")
        self.assertIsNotNone(roundness)
        self.assertEqual(roundness["tier"], "strong")

    def test_u_long_deep_is_strong(self):
        """/uː/ is primary cardinal for DEEP → strong."""
        acts = self._get_activations_for("uː")
        deep = self._find_activation(acts, "DEEP")
        self.assertIsNotNone(deep)
        self.assertEqual(deep["tier"], "strong")

    def test_u_long_softness_is_strong(self):
        """/uː/ is primary cardinal for SOFTNESS → strong."""
        acts = self._get_activations_for("uː")
        softness = self._find_activation(acts, "SOFTNESS")
        self.assertIsNotNone(softness)
        self.assertEqual(softness["tier"], "strong")

    def test_t_smallness_is_strong(self):
        """/t/ is primary cardinal for SMALLNESS → strong."""
        acts = self._get_activations_for("t")
        smallness = self._find_activation(acts, "SMALLNESS")
        self.assertIsNotNone(smallness)
        self.assertEqual(smallness["tier"], "strong")

    def test_t_hardness_is_weak(self):
        """/t/ matches only broad groups for HARDNESS (k is primary) → weak."""
        acts = self._get_activations_for("t")
        hardness = self._find_activation(acts, "HARDNESS")
        self.assertIsNotNone(hardness)
        self.assertEqual(hardness["tier"], "weak")

    def test_k_hardness_is_strong(self):
        """/k/ is primary cardinal for HARDNESS → strong."""
        acts = self._get_activations_for("k")
        hardness = self._find_activation(acts, "HARDNESS")
        self.assertIsNotNone(hardness)
        self.assertEqual(hardness["tier"], "strong")

    def test_palatal_stop_not_primary(self):
        """/ɟ/ and /c/ are not primary for any macro-concept → no strong tiers."""
        for ipa in ("ɟ", "c"):
            acts = self._get_activations_for(ipa)
            for a in acts:
                self.assertNotEqual(
                    a["tier"], "strong",
                    f"{ipa} should not have strong activation for {a['macro_concept']}"
                )

    def test_palatal_stop_medium_via_specific_groups(self):
        """/c/ should get medium for concepts matching voiceless_stop or palatal_voiceless."""
        acts = self._get_activations_for("c")
        tiers = {a["macro_concept"]: a["tier"] for a in acts}
        # SMALLNESS has voiceless_stop in its sound_groups — /c/ has voiceless_stop → medium
        self.assertEqual(tiers.get("SMALLNESS"), "medium")


class TestActivationProfileStructure(unittest.TestCase):
    """Test the full profile structure for 'rút'."""

    def setUp(self):
        self.phonemes = decompose_word("rút")
        self.profile = build_activation_profile(self.phonemes)

    def test_consonant_layer_count(self):
        """'rút' has 2 consonants: r and t."""
        self.assertEqual(len(self.profile["consonant_layer"]), 2)

    def test_vowel_layer_count(self):
        """'rút' has 1 vowel: uː."""
        self.assertEqual(len(self.profile["vowel_layer"]), 1)

    def test_metadata(self):
        self.assertEqual(self.profile["metadata"]["phoneme_count"], 3)
        self.assertEqual(self.profile["metadata"]["consonant_count"], 2)
        self.assertEqual(self.profile["metadata"]["vowel_count"], 1)

    def test_consonant_positions(self):
        positions = [p["position"] for p in self.profile["consonant_layer"]]
        self.assertEqual(positions, [1, 3])

    def test_vowel_positions(self):
        positions = [p["position"] for p in self.profile["vowel_layer"]]
        self.assertEqual(positions, [2])

    def test_activations_sorted_by_tier(self):
        """Within each position, activations should be sorted strong > medium > weak."""
        tier_order = {"strong": 0, "medium": 1, "weak": 2}
        for layer in ("consonant_layer", "vowel_layer"):
            for pos in self.profile[layer]:
                tiers = [tier_order[a["tier"]] for a in pos["activations"]]
                self.assertEqual(tiers, sorted(tiers),
                    f"Position {pos['position']} activations not sorted by tier")


class TestReinforcement(unittest.TestCase):
    """Reinforcement: codes active in BOTH consonant and vowel layers."""

    def test_rut_reinforced_codes(self):
        """'rút' should have reinforcement for codes appearing in both layers."""
        phonemes = decompose_word("rút")
        profile = build_activation_profile(phonemes)
        reinforced = profile["reinforced_codes"]
        # Both layers must share these codes
        consonant_codes = set()
        for pos in profile["consonant_layer"]:
            for a in pos["activations"]:
                consonant_codes.add(a["macro_concept_id"])
        vowel_codes = set()
        for pos in profile["vowel_layer"]:
            for a in pos["activations"]:
                vowel_codes.add(a["macro_concept_id"])
        expected = sorted(consonant_codes & vowel_codes)
        self.assertEqual(reinforced, expected)


class TestActivationDensity(unittest.TestCase):
    """Density: count of positions activating each macro-concept."""

    def test_density_values_are_positive(self):
        phonemes = decompose_word("rút")
        profile = build_activation_profile(phonemes)
        for code, count in profile["activation_density"].items():
            self.assertGreater(count, 0)

    def test_density_max_is_phoneme_count(self):
        """No code can be activated at more positions than total phonemes."""
        phonemes = decompose_word("rút")
        profile = build_activation_profile(phonemes)
        for code, count in profile["activation_density"].items():
            self.assertLessEqual(count, len(phonemes))


class TestNoActivation(unittest.TestCase):
    """Verify that non-matching phoneme-concept pairs produce no activation."""

    def test_r_does_not_activate_smallness(self):
        """/r/ (voiced alveolar vibrant) should not activate SMALLNESS (voiceless, stop)."""
        features = load_config("phoneme_features.json")
        acts = compute_macro_concept_activations("r", features["consonants"]["r"]["sound_groups"])
        mc_names = [a["macro_concept"] for a in acts]
        self.assertNotIn("SMALLNESS", mc_names)

    def test_vowel_does_not_activate_nose(self):
        """/uː/ should not activate NOSE (nasal_voiced only)."""
        features = load_config("phoneme_features.json")
        acts = compute_macro_concept_activations("uː", features["vowels"]["uː"]["sound_groups"])
        mc_names = [a["macro_concept"] for a in acts]
        self.assertNotIn("NOSE", mc_names)


if __name__ == "__main__":
    unittest.main()
