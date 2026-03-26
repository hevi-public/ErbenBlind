"""Tests for the result recorder module."""

import json
import tempfile
import unittest
from pathlib import Path

import pipeline.result_recorder as rr
from pipeline.result_recorder import (
    save_profile,
    save_step,
    save_metadata,
    list_trials,
    load_trial,
)


class TestResultRecorder(unittest.TestCase):
    """All recorder tests use a temporary directory."""

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


class TestSaveProfile(TestResultRecorder):

    def test_creates_profile_json(self):
        dummy_profile = {
            "consonant_layer": [], "vowel_layer": [],
            "reinforced_codes": [], "activation_density": {},
            "strong_across_positions": [],
            "metadata": {"phoneme_count": 3},
        }
        path = save_profile("test", "001", dummy_profile)  # type: ignore[arg-type]
        self.assertTrue(path.exists())
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["metadata"]["phoneme_count"], 3)

    def test_control_saves_under_controls(self):
        path = save_profile("test", "001", {
            "consonant_layer": [], "vowel_layer": [],
            "reinforced_codes": [], "activation_density": {},
            "strong_across_positions": [], "metadata": {},
        }, is_control=True)  # type: ignore[arg-type]
        self.assertIn("controls", str(path))


class TestSaveStep(TestResultRecorder):

    def test_creates_prompt_and_response_files(self):
        prompt_path, response_path = save_step(
            "test", "001", 1, "consonant",
            "the prompt", "the response",
        )
        self.assertTrue(prompt_path.exists())
        self.assertTrue(response_path.exists())
        self.assertEqual(prompt_path.read_text(), "the prompt")
        self.assertEqual(response_path.read_text(), "the response")

    def test_file_naming(self):
        prompt_path, response_path = save_step(
            "test", "001", 2, "vowel",
            "p", "r",
        )
        self.assertEqual(prompt_path.name, "step2_vowel_prompt.txt")
        self.assertEqual(response_path.name, "step2_vowel_response.txt")


class TestSaveMetadata(TestResultRecorder):

    def test_creates_meta_json(self):
        path = save_metadata(
            "test", "001", word="rút", target_domain="SD-15",
            trial_type="target_present",
        )
        self.assertTrue(path.exists())
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["word"], "rút")
        self.assertEqual(data["trial_type"], "target_present")

    def test_includes_timestamp(self):
        path = save_metadata(
            "test", "001", word="rút", target_domain="SD-15",
            trial_type="target_present",
        )
        with open(path) as f:
            data = json.load(f)
        self.assertIn("timestamp", data)

    def test_extra_fields_merged(self):
        path = save_metadata(
            "test", "001", word="rút", target_domain="SD-15",
            trial_type="target_present",
            extra={"ordering": "consonant_first"},
        )
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["ordering"], "consonant_first")


class TestListTrials(TestResultRecorder):

    def test_empty_when_no_results(self):
        self.assertEqual(list_trials(), [])

    def test_lists_saved_trials(self):
        save_metadata("aaa", "001", word="a", target_domain="SD-01",
                      trial_type="target_present")
        save_metadata("bbb", "002", word="b", target_domain="SD-02",
                      trial_type="target_absent")
        trials = list_trials()
        self.assertEqual(trials, ["aaa_001", "bbb_002"])

    def test_lists_controls_separately(self):
        save_metadata("aaa", "001", word="a", target_domain="SD-01",
                      trial_type="target_present")
        save_metadata("ctrl", "001", word="ctrl", target_domain="SD-01",
                      trial_type="random_profile", is_control=True)
        self.assertEqual(list_trials(is_control=False), ["aaa_001"])
        self.assertEqual(list_trials(is_control=True), ["ctrl_001"])


class TestLoadTrial(TestResultRecorder):

    def test_round_trip(self):
        """Save profile, step, and metadata, then load them all back."""
        save_profile("test", "001", {
            "consonant_layer": [{"position": 1, "ipa": "r", "type": "consonant", "activations": []}],
            "vowel_layer": [],
            "reinforced_codes": [], "activation_density": {},
            "strong_across_positions": [], "metadata": {},
        })  # type: ignore[arg-type]
        save_step("test", "001", 1, "consonant", "prompt1", "response1")
        save_step("test", "001", 2, "vowel", "prompt2", "response2")
        save_metadata("test", "001", word="test", target_domain="SD-01",
                      trial_type="target_present")

        data = load_trial("test", "001")
        self.assertIn("profile", data)
        self.assertIn("meta", data)
        self.assertEqual(data["step1_consonant_prompt"], "prompt1")
        self.assertEqual(data["step1_consonant_response"], "response1")
        self.assertEqual(data["step2_vowel_prompt"], "prompt2")
        self.assertEqual(data["step2_vowel_response"], "response2")


if __name__ == "__main__":
    unittest.main()
