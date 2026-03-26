"""Tests for the option selector module."""

import random
import unittest

from pipeline.option_selector import select_options, ADJACENCY


class TestTargetPresent(unittest.TestCase):
    """target_present: target domain must be in the options."""

    def test_target_is_included(self):
        rng = random.Random(42)
        options = select_options("SD-15", "target_present", num_options=4, rng=rng)
        ids = [o["id"] for o in options]
        self.assertIn("SD-15", ids)

    def test_correct_option_count(self):
        rng = random.Random(42)
        options = select_options("SD-15", "target_present", num_options=4, rng=rng)
        self.assertEqual(len(options), 4)

    def test_no_adjacent_distractors(self):
        """Distractors should be non-adjacent to the target."""
        rng = random.Random(42)
        adjacent = set(ADJACENCY.get("SD-15", []))
        options = select_options("SD-15", "target_present", num_options=4, rng=rng)
        distractor_ids = [o["id"] for o in options if o["id"] != "SD-15"]
        for did in distractor_ids:
            self.assertNotIn(did, adjacent,
                f"Distractor {did} is adjacent to target SD-15")

    def test_no_duplicate_options(self):
        rng = random.Random(42)
        options = select_options("SD-01", "target_present", num_options=4, rng=rng)
        ids = [o["id"] for o in options]
        self.assertEqual(len(ids), len(set(ids)))


class TestTargetAbsent(unittest.TestCase):
    """target_absent: target domain must NOT be in the options."""

    def test_target_is_excluded(self):
        rng = random.Random(42)
        options = select_options("SD-15", "target_absent", num_options=4, rng=rng)
        ids = [o["id"] for o in options]
        self.assertNotIn("SD-15", ids)

    def test_correct_option_count(self):
        rng = random.Random(42)
        options = select_options("SD-15", "target_absent", num_options=4, rng=rng)
        self.assertEqual(len(options), 4)

    def test_includes_adjacent_when_available(self):
        """Should include at least one adjacent domain when available."""
        rng = random.Random(42)
        adjacent = set(ADJACENCY.get("SD-15", []))
        if not adjacent:
            self.skipTest("SD-15 has no adjacent domains defined")
        # Run multiple times to confirm adjacency inclusion is possible
        found_adjacent = False
        for seed in range(100):
            options = select_options("SD-15", "target_absent", num_options=4,
                                     rng=random.Random(seed))
            ids = {o["id"] for o in options}
            if ids & adjacent:
                found_adjacent = True
                break
        self.assertTrue(found_adjacent, "Should include adjacent domain at least sometimes")


class TestNullTrial(unittest.TestCase):
    """null_trial: all options must be semantically distant from target."""

    def test_target_is_excluded(self):
        rng = random.Random(42)
        options = select_options("SD-15", "null_trial", num_options=4, rng=rng)
        ids = [o["id"] for o in options]
        self.assertNotIn("SD-15", ids)

    def test_no_adjacent_options(self):
        rng = random.Random(42)
        adjacent = set(ADJACENCY.get("SD-15", []))
        options = select_options("SD-15", "null_trial", num_options=4, rng=rng)
        ids = {o["id"] for o in options}
        self.assertFalse(ids & adjacent,
            f"Null trial should have no adjacent domains, found {ids & adjacent}")

    def test_correct_option_count(self):
        rng = random.Random(42)
        options = select_options("SD-15", "null_trial", num_options=4, rng=rng)
        self.assertEqual(len(options), 4)


class TestRandomProfile(unittest.TestCase):
    """random_profile: any options, no target constraints."""

    def test_correct_option_count(self):
        rng = random.Random(42)
        options = select_options("SD-15", "random_profile", num_options=4, rng=rng)
        self.assertEqual(len(options), 4)

    def test_no_duplicate_options(self):
        rng = random.Random(42)
        options = select_options("SD-15", "random_profile", num_options=4, rng=rng)
        ids = [o["id"] for o in options]
        self.assertEqual(len(ids), len(set(ids)))


class TestOptionsAreShuffled(unittest.TestCase):
    """Options should be shuffled so target position varies."""

    def test_target_not_always_first(self):
        positions = []
        for seed in range(20):
            rng = random.Random(seed)
            options = select_options("SD-15", "target_present", num_options=4, rng=rng)
            ids = [o["id"] for o in options]
            positions.append(ids.index("SD-15"))
        unique_positions = set(positions)
        self.assertGreater(len(unique_positions), 1,
            "Target should appear at different positions across runs")


class TestOptionStructure(unittest.TestCase):
    """Each option should have the required keys."""

    def test_options_have_required_keys(self):
        rng = random.Random(42)
        options = select_options("SD-15", "target_present", num_options=4, rng=rng)
        for opt in options:
            self.assertIn("id", opt)
            self.assertIn("label", opt)
            self.assertIn("description_for_evaluation", opt)


class TestEdgeCases(unittest.TestCase):

    def test_unknown_target_raises(self):
        with self.assertRaises(ValueError):
            select_options("SD-99", "target_present", rng=random.Random(42))

    def test_unknown_trial_type_raises(self):
        with self.assertRaises(ValueError):
            select_options("SD-15", "nonexistent", rng=random.Random(42))  # type: ignore[arg-type]

    def test_three_options(self):
        rng = random.Random(42)
        options = select_options("SD-15", "target_present", num_options=3, rng=rng)
        self.assertEqual(len(options), 3)

    def test_works_for_all_domains(self):
        """Every domain should work as a target for all trial types."""
        from pipeline.config_loader import load_config
        domains = load_config("semantic_domains.json")["domains"]
        for domain in domains:
            for trial_type in ("target_present", "target_absent", "null_trial"):
                options = select_options(
                    domain["id"], trial_type,  # type: ignore[arg-type]
                    num_options=4, rng=random.Random(42),
                )
                self.assertEqual(len(options), 4,
                    f"Failed for {domain['id']} / {trial_type}")


if __name__ == "__main__":
    unittest.main()
