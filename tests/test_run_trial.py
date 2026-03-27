"""Tests for the trial orchestrator."""

import tempfile
import unittest
from pathlib import Path

from run_trial import (
    parse_forced_choice, _sanitize_word_id, _generate_dry_run_response,
    run_single_trial,
)
from pipeline.phoneme_decomposer import decompose_word
from pipeline.activation_profiler import build_activation_profile
import pipeline.result_recorder as rr


SAMPLE_OPTIONS = [
    {"id": "SD-15", "label": "DECAY_DETERIORATION"},
    {"id": "SD-01", "label": "WATER_LIQUID"},
    {"id": "SD-03", "label": "FIRE_HEAT"},
    {"id": "SD-20", "label": "ROTATION_TWISTING"},
]


class TestParseForChoice(unittest.TestCase):
    """Test forced-choice response parsing."""

    def test_explicit_choice_is_a(self):
        response = "Based on the pattern, I think A is most consistent. My choice is A."
        result = parse_forced_choice(response, SAMPLE_OPTIONS)
        self.assertEqual(result, "SD-15 (DECAY_DETERIORATION)")

    def test_explicit_choice_is_b(self):
        response = "The profile suggests water. My choice is B."
        result = parse_forced_choice(response, SAMPLE_OPTIONS)
        self.assertEqual(result, "SD-01 (WATER_LIQUID)")

    def test_i_choose_c(self):
        response = "Given the strong activations at position 3, I choose C."
        result = parse_forced_choice(response, SAMPLE_OPTIONS)
        self.assertEqual(result, "SD-03 (FIRE_HEAT)")

    def test_i_pick_d(self):
        response = "I would pick D based on the reinforcement pattern."
        result = parse_forced_choice(response, SAMPLE_OPTIONS)
        self.assertEqual(result, "SD-20 (ROTATION_TWISTING)")

    def test_choice_colon(self):
        response = "The data points to turning. Choice: D"
        result = parse_forced_choice(response, SAMPLE_OPTIONS)
        self.assertEqual(result, "SD-20 (ROTATION_TWISTING)")

    def test_standalone_letter_fallback(self):
        response = "After analysis, the answer is clearly B"
        result = parse_forced_choice(response, SAMPLE_OPTIONS)
        self.assertEqual(result, "SD-01 (WATER_LIQUID)")

    def test_domain_label_fallback(self):
        response = "The profile most resembles FIRE_HEAT based on the activation pattern."
        result = parse_forced_choice(response, SAMPLE_OPTIONS)
        self.assertEqual(result, "SD-03 (FIRE_HEAT)")

    def test_unparsed_when_no_match(self):
        response = "I cannot determine which option is best from this profile."
        result = parse_forced_choice(response, SAMPLE_OPTIONS)
        self.assertEqual(result, "UNPARSED")

    def test_case_insensitive(self):
        response = "my choice is a"
        result = parse_forced_choice(response, SAMPLE_OPTIONS)
        self.assertEqual(result, "SD-15 (DECAY_DETERIORATION)")

    def test_last_letter_wins(self):
        """When multiple letters appear, prefer the last one (final answer)."""
        response = "A seems possible but B is stronger. My choice is B."
        result = parse_forced_choice(response, SAMPLE_OPTIONS)
        self.assertEqual(result, "SD-01 (WATER_LIQUID)")


class TestSanitizeWordId(unittest.TestCase):

    def test_simple_word(self):
        self.assertEqual(_sanitize_word_id("rút"), "rút")

    def test_lowercase(self):
        self.assertEqual(_sanitize_word_id("RÚT"), "rút")

    def test_spaces(self):
        self.assertEqual(_sanitize_word_id("két szó"), "két_szó")


class TestDryRunResponseGenerator(unittest.TestCase):
    """Test that dry-run responses are profile-aware and vary between words."""

    def test_step1_references_actual_codes(self):
        """Step 1 dry-run should mention actual consonant MC codes from the profile."""
        import random
        profile = build_activation_profile(decompose_word("rút"))
        response = _generate_dry_run_response(1, profile, SAMPLE_OPTIONS, random.Random(42))
        # Should reference some actual MC codes from the consonant layer
        consonant_codes = set()
        for pos in profile["consonant_layer"]:
            for act in pos["activations"]:
                if act["tier"] == "strong":
                    consonant_codes.add(act["macro_concept_id"])
        found = any(code in response for code in consonant_codes)
        self.assertTrue(found, f"Response should reference consonant codes. Got: {response}")

    def test_step2_references_reinforcement(self):
        """Step 2 dry-run should mention reinforced codes."""
        import random
        profile = build_activation_profile(decompose_word("rút"))
        response = _generate_dry_run_response(2, profile, SAMPLE_OPTIONS, random.Random(42))
        # Should contain reinforcement info
        self.assertIn("Reinforcement", response)

    def test_step3_references_concept_names(self):
        """Step 3 dry-run should mention decoded concept names, not MC codes."""
        import random
        profile = build_activation_profile(decompose_word("rút"))
        response = _generate_dry_run_response(3, profile, rng=random.Random(42))
        # Should have human-readable names
        found_name = any(
            name.lower() in response.lower()
            for name in ["turn", "uneven", "roundness", "deep", "smallness"]
        )
        self.assertTrue(found_name, f"Step 3 should reference concept names. Got: {response}")

    def test_different_words_produce_different_responses(self):
        """Different words should produce different dry-run responses."""
        import random
        profile_rut = build_activation_profile(decompose_word("rút"))
        profile_viz = build_activation_profile(decompose_word("víz"))
        resp_rut = _generate_dry_run_response(1, profile_rut, SAMPLE_OPTIONS, random.Random(42))
        resp_viz = _generate_dry_run_response(1, profile_viz, SAMPLE_OPTIONS, random.Random(42))
        self.assertNotEqual(resp_rut, resp_viz,
            "Different words should produce different dry-run responses")

    def test_choice_is_parseable(self):
        """Dry-run responses should be parseable by parse_forced_choice."""
        import random
        profile = build_activation_profile(decompose_word("rút"))
        response = _generate_dry_run_response(1, profile, SAMPLE_OPTIONS, random.Random(42))
        choice = parse_forced_choice(response, SAMPLE_OPTIONS)
        self.assertNotEqual(choice, "UNPARSED",
            f"Dry-run response should be parseable. Got: {response}")

    def test_does_not_always_pick_a(self):
        """Over multiple seeds, dry-run should pick different options."""
        import random
        profile = build_activation_profile(decompose_word("rút"))
        choices = set()
        for seed in range(20):
            resp = _generate_dry_run_response(1, profile, SAMPLE_OPTIONS, random.Random(seed))
            choice = parse_forced_choice(resp, SAMPLE_OPTIONS)
            choices.add(choice)
        self.assertGreater(len(choices), 1,
            "Dry-run should pick different options across seeds")


class TestDryRunIntegration(unittest.TestCase):
    """Test dry-run mode end-to-end with result saving."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_results = rr.RESULTS_DIR
        self._orig_controls = rr.CONTROLS_DIR
        rr.RESULTS_DIR = Path(self._tmpdir.name) / "results"
        rr.CONTROLS_DIR = Path(self._tmpdir.name) / "controls"

    def tearDown(self):
        rr.RESULTS_DIR = self._orig_results
        rr.CONTROLS_DIR = self._orig_controls
        self._tmpdir.cleanup()

    def test_dry_run_creates_all_artifacts(self):
        summary = run_single_trial(
            word="rút", target_domain="SD-15",
            trial_type="target_present",
            run_id="dry001", dry_run=True, seed=42,
        )
        trial_dir = rr.RESULTS_DIR / "rút_dry001"
        self.assertTrue(trial_dir.exists())
        self.assertTrue((trial_dir / "profile.json").exists())
        self.assertTrue((trial_dir / "meta.json").exists())
        self.assertTrue((trial_dir / "step1_consonant_prompt.txt").exists())
        self.assertTrue((trial_dir / "step1_consonant_response.txt").exists())
        self.assertTrue((trial_dir / "step2_vowel_prompt.txt").exists())
        self.assertTrue((trial_dir / "step2_vowel_response.txt").exists())
        self.assertTrue((trial_dir / "step3_synthesis_prompt.txt").exists())
        self.assertTrue((trial_dir / "step3_synthesis_response.txt").exists())

    def test_dry_run_returns_summary(self):
        summary = run_single_trial(
            word="víz", target_domain="SD-01",
            trial_type="target_present",
            run_id="dry002", dry_run=True, seed=42,
        )
        self.assertEqual(summary["word"], "víz")
        self.assertEqual(summary["target_domain"], "SD-01")
        self.assertIn("step1_choice", summary)
        self.assertIn("step2_choice", summary)

    def test_random_profile_saves_to_controls(self):
        summary = run_single_trial(
            word="rút", target_domain="SD-15",
            trial_type="random_profile",
            run_id="ctrl001", dry_run=True, seed=42,
        )
        ctrl_dir = rr.CONTROLS_DIR / "rút_ctrl001"
        self.assertTrue(ctrl_dir.exists())
        result_dir = rr.RESULTS_DIR / "rút_ctrl001"
        self.assertFalse(result_dir.exists())

    def test_different_words_save_different_responses(self):
        """Two different words in dry-run should produce different saved response files."""
        run_single_trial(
            word="rút", target_domain="SD-15",
            trial_type="target_present",
            run_id="cmp001", dry_run=True, seed=42,
        )
        run_single_trial(
            word="víz", target_domain="SD-01",
            trial_type="target_present",
            run_id="cmp002", dry_run=True, seed=42,
        )
        resp1 = (rr.RESULTS_DIR / "rút_cmp001" / "step1_consonant_response.txt").read_text()
        resp2 = (rr.RESULTS_DIR / "víz_cmp002" / "step1_consonant_response.txt").read_text()
        self.assertNotEqual(resp1, resp2,
            "Different words should have different response files in dry-run")


if __name__ == "__main__":
    unittest.main()
