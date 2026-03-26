"""Tests for the prompt formatter module."""

import unittest

from pipeline.phoneme_decomposer import decompose_word
from pipeline.activation_profiler import build_activation_profile
from pipeline.prompt_formatter import (
    format_step1_prompt,
    format_step2_prompt,
    format_step3_prompt,
    get_system_prompt,
)


DUMMY_OPTIONS = [
    {"id": "SD-15", "label": "DECAY_DETERIORATION",
     "description_for_evaluation": "Decay, rot, deterioration, ugly, foul"},
    {"id": "SD-01", "label": "WATER_LIQUID",
     "description_for_evaluation": "Water, liquids, flowing, wet, rain"},
    {"id": "SD-03", "label": "FIRE_HEAT",
     "description_for_evaluation": "Fire, heat, burning, warmth"},
    {"id": "SD-20", "label": "ROTATION_TWISTING",
     "description_for_evaluation": "Turning, twisting, spinning, circular motion"},
]


class TestSystemPrompt(unittest.TestCase):

    def test_system_prompt_is_string(self):
        prompt = get_system_prompt()
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 50)

    def test_system_prompt_does_not_reveal_language(self):
        prompt = get_system_prompt()
        self.assertNotIn("Hungarian", prompt)
        self.assertNotIn("hungarian", prompt)


class TestStep1Blinding(unittest.TestCase):
    """Step 1 must use coded labels only — no concept names."""

    def setUp(self):
        phonemes = decompose_word("rút")
        self.profile = build_activation_profile(phonemes)
        self.prompt = format_step1_prompt(self.profile, DUMMY_OPTIONS)

    def test_contains_mc_codes(self):
        self.assertIn("MC-", self.prompt)

    def test_does_not_contain_concept_names(self):
        """Concept names like ROUNDNESS, TONGUE etc must not appear."""
        forbidden = [
            "ROUNDNESS", "TONGUE", "TURN", "SMALLNESS", "HARDNESS",
            "DEEP", "SOFTNESS", "AIRFLOW", "UNEVEN", "EXPULSION",
        ]
        for name in forbidden:
            self.assertNotIn(name, self.prompt,
                f"Step 1 prompt should not contain concept name '{name}'")

    def test_contains_tier_notation(self):
        """Should have *** / ** / * tier markers."""
        self.assertIn("***", self.prompt)

    def test_contains_forced_choice_options(self):
        self.assertIn("DECAY_DETERIORATION", self.prompt)
        self.assertIn("WATER_LIQUID", self.prompt)

    def test_contains_position_labels(self):
        self.assertIn("Position 1:", self.prompt)

    def test_does_not_contain_word(self):
        self.assertNotIn("rút", self.prompt)
        self.assertNotIn("rut", self.prompt.lower())

    def test_does_not_contain_phonemes(self):
        """IPA symbols should not leak into the prompt."""
        self.assertNotIn("/r/", self.prompt)
        self.assertNotIn("uː", self.prompt)


class TestStep2Blinding(unittest.TestCase):
    """Step 2 uses coded labels + quotes step 1."""

    def setUp(self):
        phonemes = decompose_word("rút")
        self.profile = build_activation_profile(phonemes)
        self.step1_response = "I pick option A based on the pattern."
        self.prompt = format_step2_prompt(
            self.profile, DUMMY_OPTIONS,
            self.step1_response, "SD-15 (DECAY_DETERIORATION)",
        )

    def test_contains_quoted_step1(self):
        self.assertIn(self.step1_response, self.prompt)

    def test_contains_step1_choice(self):
        self.assertIn("SD-15", self.prompt)

    def test_contains_vowel_activations(self):
        self.assertIn("MC-", self.prompt)

    def test_contains_reinforcement(self):
        self.assertIn("Reinforcement", self.prompt)

    def test_contains_combined_profile(self):
        """Should show both consonant and vowel positions."""
        self.assertIn("[consonant]", self.prompt)
        self.assertIn("[vowel]", self.prompt)

    def test_does_not_contain_concept_names(self):
        forbidden = ["ROUNDNESS", "TONGUE", "TURN", "SMALLNESS", "DEEP"]
        for name in forbidden:
            self.assertNotIn(name, self.prompt)


class TestStep3Decoded(unittest.TestCase):
    """Step 3 decodes to human-readable labels — names should be visible."""

    def setUp(self):
        phonemes = decompose_word("rút")
        self.profile = build_activation_profile(phonemes)
        self.prompt = format_step3_prompt(
            self.profile,
            "I pick A.", "SD-15",
            "Vowels confirm.", "SD-15",
        )

    def test_contains_decoded_labels(self):
        """Step 3 should have human-readable concept names."""
        # At least some of these should appear
        found = any(
            name in self.prompt
            for name in ["ROUNDNESS", "TURN", "SMALLNESS", "DEEP", "SOFTNESS"]
        )
        self.assertTrue(found, "Step 3 should contain decoded concept names")

    def test_contains_tier_labels(self):
        """Step 3 uses (strong)/(medium)/(weak) instead of ***."""
        self.assertIn("(strong)", self.prompt)

    def test_contains_prior_commitments(self):
        self.assertIn("I pick A.", self.prompt)
        self.assertIn("Vowels confirm.", self.prompt)

    def test_contains_activation_summary(self):
        self.assertIn("Strongest activations", self.prompt)
        self.assertIn("Unique activations", self.prompt)

    def test_is_open_ended(self):
        """Step 3 should NOT have forced-choice lettered options."""
        # Should not have the A) B) C) D) format
        self.assertNotIn("A) ", self.prompt)
        self.assertNotIn("B) ", self.prompt)


class TestTemplateCompleteness(unittest.TestCase):
    """Verify no unfilled template variables remain in formatted prompts."""

    def setUp(self):
        phonemes = decompose_word("rút")
        self.profile = build_activation_profile(phonemes)

    def test_step1_no_unfilled_vars(self):
        prompt = format_step1_prompt(self.profile, DUMMY_OPTIONS)
        self.assertNotIn("{", prompt)
        self.assertNotIn("}", prompt)

    def test_step2_no_unfilled_vars(self):
        prompt = format_step2_prompt(
            self.profile, DUMMY_OPTIONS, "response1", "SD-01"
        )
        self.assertNotIn("{", prompt)
        self.assertNotIn("}", prompt)

    def test_step3_no_unfilled_vars(self):
        prompt = format_step3_prompt(
            self.profile, "r1", "SD-01", "r2", "SD-01"
        )
        self.assertNotIn("{", prompt)
        self.assertNotIn("}", prompt)


if __name__ == "__main__":
    unittest.main()
