"""Tests for the trial orchestrator."""

import tempfile
import unittest
from pathlib import Path

from run_trial import parse_forced_choice, _sanitize_word_id, run_single_trial
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


class TestDryRun(unittest.TestCase):
    """Test dry-run mode saves correct artifacts."""

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
        # Should NOT be in results
        result_dir = rr.RESULTS_DIR / "rút_ctrl001"
        self.assertFalse(result_dir.exists())


if __name__ == "__main__":
    unittest.main()
