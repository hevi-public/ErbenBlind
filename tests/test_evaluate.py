"""Tests for the evaluation module."""

import unittest

from evaluate import (
    binomial_p_value,
    _extract_domain_id,
    _is_hit,
    compute_hit_rates,
    compute_synthesis_consistency,
    generate_report,
    _min_hits_for_significance,
)


def _make_trial(trial_type, target, step1_choice, step2_choice, step3_pred=""):
    """Helper to create a minimal trial dict."""
    return {
        "meta": {
            "trial_type": trial_type,
            "target_domain": target,
            "step1_choice": step1_choice,
            "step2_choice": step2_choice,
            "step3_prediction": step3_pred,
            "word": "test",
            "forced_choice_options_step1": [
                {"id": "SD-01"}, {"id": "SD-02"}, {"id": "SD-03"}, {"id": "SD-04"},
            ],
        }
    }


class TestBinomialPValue(unittest.TestCase):

    def test_all_correct(self):
        """All hits should give very low p-value."""
        p = binomial_p_value(10, 10, 0.25)
        self.assertLess(p, 0.001)

    def test_none_correct(self):
        """No hits should give p-value near 1."""
        p = binomial_p_value(0, 10, 0.25)
        self.assertAlmostEqual(p, 1.0, places=5)

    def test_at_chance(self):
        """Hits at chance level should give p > 0.5."""
        p = binomial_p_value(25, 100, 0.25)
        self.assertGreater(p, 0.4)

    def test_zero_trials(self):
        p = binomial_p_value(0, 0, 0.25)
        self.assertEqual(p, 1.0)

    def test_known_value(self):
        """4 out of 4 with p=0.25: 0.25^4 = 0.00390625."""
        p = binomial_p_value(4, 4, 0.25)
        self.assertAlmostEqual(p, 0.25**4, places=6)


class TestExtractDomainId(unittest.TestCase):

    def test_standard_format(self):
        self.assertEqual(_extract_domain_id("SD-15 (DECAY_DETERIORATION)"), "SD-15")

    def test_id_only(self):
        self.assertEqual(_extract_domain_id("SD-01"), "SD-01")

    def test_unparsed(self):
        self.assertIsNone(_extract_domain_id("UNPARSED"))

    def test_empty(self):
        self.assertIsNone(_extract_domain_id(""))


class TestIsHit(unittest.TestCase):

    def test_correct_hit(self):
        self.assertTrue(_is_hit("SD-15 (DECAY_DETERIORATION)", "SD-15"))

    def test_wrong_choice(self):
        self.assertFalse(_is_hit("SD-01 (WATER_LIQUID)", "SD-15"))

    def test_unparsed_is_miss(self):
        self.assertFalse(_is_hit("UNPARSED", "SD-15"))


class TestComputeHitRates(unittest.TestCase):

    def test_target_present_all_hits(self):
        trials = [
            _make_trial("target_present", "SD-01", "SD-01 (X)", "SD-01 (X)"),
            _make_trial("target_present", "SD-01", "SD-01 (X)", "SD-01 (X)"),
        ]
        rates = compute_hit_rates(trials)
        self.assertEqual(rates["target_present"]["step1_hits"], 2)
        self.assertEqual(rates["target_present"]["step1_rate"], 1.0)
        self.assertEqual(rates["target_present"]["step2_rate"], 1.0)

    def test_target_present_no_hits(self):
        trials = [
            _make_trial("target_present", "SD-01", "SD-02 (Y)", "SD-03 (Z)"),
        ]
        rates = compute_hit_rates(trials)
        self.assertEqual(rates["target_present"]["step1_hits"], 0)
        self.assertEqual(rates["target_present"]["step1_rate"], 0.0)

    def test_unparsed_excluded_from_totals(self):
        trials = [
            _make_trial("target_present", "SD-01", "UNPARSED", "SD-01 (X)"),
        ]
        rates = compute_hit_rates(trials)
        self.assertEqual(rates["target_present"]["step1_total"], 0)
        self.assertEqual(rates["target_present"]["step2_total"], 1)
        self.assertEqual(rates["target_present"]["unparsed_count"], 1)

    def test_multiple_trial_types(self):
        trials = [
            _make_trial("target_present", "SD-01", "SD-01 (X)", "SD-01 (X)"),
            _make_trial("target_absent", "SD-01", "SD-02 (Y)", "SD-02 (Y)"),
        ]
        rates = compute_hit_rates(trials)
        self.assertIn("target_present", rates)
        self.assertIn("target_absent", rates)


class TestSynthesisConsistency(unittest.TestCase):

    def test_consistent_when_label_in_prediction(self):
        trials = [
            _make_trial("target_present", "SD-01",
                        "SD-01", "SD-15 (DECAY_DETERIORATION)",
                        step3_pred="The profile suggests decay and deterioration."),
        ]
        result = compute_synthesis_consistency(trials)
        self.assertEqual(result["consistent_count"], 1)

    def test_inconsistent_when_no_match(self):
        trials = [
            _make_trial("target_present", "SD-01",
                        "SD-01", "SD-15 (DECAY_DETERIORATION)",
                        step3_pred="This clearly indicates water flow."),
        ]
        result = compute_synthesis_consistency(trials)
        self.assertEqual(result["consistent_count"], 0)


class TestGenerateReport(unittest.TestCase):

    def test_report_has_all_sections(self):
        results = [
            _make_trial("target_present", "SD-01", "SD-01 (X)", "SD-01 (X)"),
            _make_trial("target_absent", "SD-01", "SD-02 (Y)", "SD-02 (Y)"),
        ]
        controls = [
            _make_trial("random_profile", "SD-01", "SD-03 (Z)", "SD-04 (W)"),
        ]
        report = generate_report(results, controls)
        self.assertIn("summary", report)
        self.assertIn("hit_rates_by_trial_type", report)
        self.assertIn("signal_strength", report)
        self.assertIn("statistical_significance", report)
        self.assertIn("synthesis_consistency", report)
        self.assertIn("per_word_breakdown", report)

    def test_report_total_counts(self):
        results = [_make_trial("target_present", "SD-01", "SD-01", "SD-01")]
        controls = [_make_trial("random_profile", "SD-01", "SD-02", "SD-03")]
        report = generate_report(results, controls)
        self.assertEqual(report["summary"]["total_trials"], 2)
        self.assertEqual(report["summary"]["result_trials"], 1)
        self.assertEqual(report["summary"]["control_trials"], 1)


class TestMinHitsForSignificance(unittest.TestCase):

    def test_small_sample(self):
        """With 4 options and 10 trials, need ~5 hits for p<0.05."""
        min_hits = _min_hits_for_significance(10, 0.25)
        self.assertGreater(min_hits, 2)
        self.assertLessEqual(min_hits, 10)

    def test_zero_trials(self):
        self.assertEqual(_min_hits_for_significance(0, 0.25), 0)


if __name__ == "__main__":
    unittest.main()
