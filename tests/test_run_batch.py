"""Tests for the parallel batch runner."""

import tempfile
import unittest
from pathlib import Path

import pipeline.result_recorder as rr
from run_batch import parse_word_list, build_trial_specs, run_batch, _run_trial_from_spec


class TestParseWordList(unittest.TestCase):
    """Test TSV word list parsing."""

    def test_parses_basic_tsv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("word\ttarget_domain\tactual_meaning\n")
            f.write("rút\tSD-15\tugly, foul\n")
            f.write("víz\tSD-01\twater\n")
            f.flush()
            entries = parse_word_list(f.name)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["word"], "rút")
        self.assertEqual(entries[0]["target_domain"], "SD-15")
        self.assertEqual(entries[1]["word"], "víz")

    def test_skips_blank_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("rút\tSD-15\tugly\n")
            f.write("\n")
            f.write("víz\tSD-01\twater\n")
            f.flush()
            entries = parse_word_list(f.name)
        self.assertEqual(len(entries), 2)

    def test_skips_header(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("word\ttarget_domain\tactual_meaning\n")
            f.write("rút\tSD-15\tugly\n")
            f.flush()
            entries = parse_word_list(f.name)
        self.assertEqual(len(entries), 1)


class TestBuildTrialSpecs(unittest.TestCase):
    """Test trial spec generation."""

    def setUp(self):
        self.entries = [
            {"word": "rút", "target_domain": "SD-15", "actual_meaning": "ugly"},
            {"word": "víz", "target_domain": "SD-01", "actual_meaning": "water"},
        ]

    def test_target_present_only(self):
        specs = build_trial_specs(self.entries, runs_per_word=2, include_controls=False,
                                  model="sonnet", num_options=4, seed=42, dry_run=True)
        self.assertEqual(len(specs), 4)  # 2 words × 2 runs
        self.assertTrue(all(s[2] == "target_present" for s in specs))

    def test_with_controls(self):
        specs = build_trial_specs(self.entries, runs_per_word=1, include_controls=True,
                                  model="sonnet", num_options=4, seed=42, dry_run=True)
        # 2 words × (1 target_present + 3 controls) = 8
        self.assertEqual(len(specs), 8)
        types = [s[2] for s in specs]
        self.assertEqual(types.count("target_present"), 2)
        self.assertEqual(types.count("target_absent"), 2)
        self.assertEqual(types.count("null_trial"), 2)
        self.assertEqual(types.count("random_profile"), 2)

    def test_run_ids_are_sequential(self):
        specs = build_trial_specs(self.entries, runs_per_word=2, include_controls=False,
                                  model="sonnet", num_options=4, seed=None, dry_run=True)
        run_ids = [s[3] for s in specs]
        self.assertEqual(run_ids, ["001", "002", "003", "004"])

    def test_seeds_are_offset(self):
        specs = build_trial_specs(self.entries, runs_per_word=1, include_controls=False,
                                  model="sonnet", num_options=4, seed=100, dry_run=True)
        seeds = [s[6] for s in specs]
        self.assertEqual(seeds, [101, 102])

    def test_no_seed_gives_none(self):
        specs = build_trial_specs(self.entries, runs_per_word=1, include_controls=False,
                                  model="sonnet", num_options=4, seed=None, dry_run=True)
        seeds = [s[6] for s in specs]
        self.assertEqual(seeds, [None, None])


class TestRunBatchDryRun(unittest.TestCase):
    """Integration test: dry-run parallel batch."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_results = rr.RESULTS_DIR
        self._orig_controls = rr.CONTROLS_DIR
        rr.RESULTS_DIR = Path(self._tmpdir.name) / "results"
        rr.CONTROLS_DIR = Path(self._tmpdir.name) / "controls"

        # Write a temp word list
        self._wl = Path(self._tmpdir.name) / "words.txt"
        self._wl.write_text(
            "word\ttarget_domain\tactual_meaning\n"
            "rút\tSD-15\tugly, foul\n"
            "víz\tSD-01\twater\n"
            "tűz\tSD-03\tfire\n",
            encoding="utf-8",
        )

    def tearDown(self):
        rr.RESULTS_DIR = self._orig_results
        rr.CONTROLS_DIR = self._orig_controls
        self._tmpdir.cleanup()

    def test_parallel_dry_run_completes(self):
        """All trials should complete without errors in dry-run mode."""
        results = run_batch(
            word_list_path=str(self._wl),
            runs_per_word=2,
            include_controls=False,
            dry_run=True,
            seed=42,
            max_workers=2,
        )
        self.assertEqual(len(results), 6)  # 3 words × 2 runs
        errors = [r for r in results if r.get("error")]
        self.assertEqual(len(errors), 0, f"Unexpected errors: {errors}")

    def test_parallel_with_controls(self):
        """Controls should also work in parallel."""
        results = run_batch(
            word_list_path=str(self._wl),
            runs_per_word=1,
            include_controls=True,
            dry_run=True,
            seed=42,
            max_workers=3,
        )
        # 3 words × (1 target_present + 3 controls) = 12
        self.assertEqual(len(results), 12)
        errors = [r for r in results if r.get("error")]
        self.assertEqual(len(errors), 0, f"Unexpected errors: {errors}")

    def test_creates_result_directories(self):
        """Each trial should create its own output directory.

        Runs specs in-process (not forked) so the monkey-patched rr.RESULTS_DIR
        is visible to result_recorder.
        """
        entries = parse_word_list(str(self._wl))
        specs = build_trial_specs(entries, 1, False, "sonnet", 4, 42, True)
        for spec in specs:
            _run_trial_from_spec(spec)
        result_dirs = list(rr.RESULTS_DIR.iterdir())
        self.assertEqual(len(result_dirs), 3)

    def test_controls_in_controls_dir(self):
        """Random profile trials should save to controls/ directory.

        Runs specs in-process (not forked) so the monkey-patched rr.CONTROLS_DIR
        is visible to result_recorder.
        """
        entries = parse_word_list(str(self._wl))
        specs = build_trial_specs(entries, 1, True, "sonnet", 4, 42, True)
        for spec in specs:
            _run_trial_from_spec(spec)
        control_dirs = list(rr.CONTROLS_DIR.iterdir())
        self.assertEqual(len(control_dirs), 3)  # 3 words × 1 random_profile each

    def test_single_worker_also_works(self):
        """Batch should work with max_workers=1 (sequential fallback)."""
        results = run_batch(
            word_list_path=str(self._wl),
            runs_per_word=1,
            include_controls=False,
            dry_run=True,
            seed=42,
            max_workers=1,
        )
        self.assertEqual(len(results), 3)
        errors = [r for r in results if r.get("error")]
        self.assertEqual(len(errors), 0)


if __name__ == "__main__":
    unittest.main()
