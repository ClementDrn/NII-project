"""Checks for the estimands and validation changed during the notebook audit."""

import tempfile
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from stats.adversarial_analysis import (
    BASELINE,
    holm_adjust,
    load_csv,
    matched_budget_overlap,
    paired_mean_comparisons,
    session_summaries,
    setting_summary,
    model_summary_tables,
    format_model_summary,
)


class AnalysisTests(unittest.TestCase):
    def test_publication_summary_preserves_counts_rates_and_single_campaign_sd(self):
        rows = []
        for model, method, repetitions in [
            ("baseline", "rule-based", 1),
            ("generator", "one-shot", 3),
        ]:
            for rep in range(repetitions):
                for scenario in [0, 1]:
                    redacted = int(model == "generator" and rep == 1 and scenario == 1)
                    error = int(model == "generator" and rep == 2 and scenario == 1)
                    rows.append(
                        dict(
                            model=model,
                            method=method,
                            setting=f"{model} / {method}",
                            repetition=rep,
                            scenario=scenario,
                            adversarial=int(rep == 0 and scenario == 0),
                            redacted=redacted,
                            errored=error,
                            usable=not (redacted or error),
                        )
                    )
        _, table, _ = model_summary_tables(pd.DataFrame(rows), [0, 1])
        baseline, generator = table.iloc[0], table.iloc[1]
        self.assertTrue(np.isnan(baseline["Adversarial rate SD (pp)"]))
        self.assertTrue(np.isnan(baseline["Redacted rate SD (pp)"]))
        self.assertAlmostEqual(
            generator["Redacted rate SD (pp)"], np.std([0, 50, 0], ddof=1)
        )
        self.assertEqual(generator["Redacted"], 1)
        self.assertEqual(generator["Errors"], 1)
        self.assertAlmostEqual(
            generator["Adversarial rate SD (pp)"], np.std([50, 0, 0], ddof=1)
        )
        self.assertAlmostEqual(generator["Redacted rate (%)"], 100 / 6)
        shown = format_model_summary(table)
        self.assertEqual(shown.iloc[1]["Failure scenarios"], "1/2")
        self.assertIn("Errors", shown.columns)
        self.assertEqual(shown.iloc[1]["Redacted rate (%)"], "16.67 ± 28.87")

    def test_equal_scenario_weight_despite_unequal_counts(self):
        frame = pd.DataFrame(
            {
                "setting": [BASELINE] * 12,
                "scenario": [0] * 10 + [1] * 2,
                "adversarial": [0] * 10 + [1] * 2,
                "redacted": [0] * 12,
                "errored": [0] * 12,
                "usable": [True] * 12,
            }
        )
        result = setting_summary(frame).loc[BASELINE]
        self.assertAlmostEqual(result.mean_adversarial_rate, 0.5)
        self.assertAlmostEqual(result.pooled_adversarial_rate, 1 / 6)

    def test_holm_known_values(self):
        np.testing.assert_allclose(
            holm_adjust([0.03, 0.001, 0.02]), [0.04, 0.003, 0.04]
        )

    def test_loader_rejects_mixed_settings_and_duplicate_keys(self):
        frame = pd.DataFrame(
            {
                "model": ["a", "b"],
                "method": ["one-shot"] * 2,
                "repetition": [0, 0],
                "scenario": [0, 0],
                "run": [0, 1],
                "redacted?": [0, 0],
                "errored?": [0, 0],
                "adversarial?": [0, 1],
                "instruction": ["a", "b"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                load_csv(path, [0])
            frame.model = "a"
            frame.run = 0
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_csv(path, [0])

    def test_paired_alignment_and_missing_scenarios(self):
        table = pd.DataFrame(
            {
                "setting": ["b", "b", "a", "a"],
                "scenario": [1, 0, 0, 1],
                "score": [2, 1, 3, 4],
            }
        )
        result = paired_mean_comparisons(table, "score", [("a", "b")])
        self.assertEqual(result.mean_difference.iloc[0], 2)
        with self.assertRaisesRegex(ValueError, "every scenario"):
            paired_mean_comparisons(table.iloc[:-1], "score", [("a", "b")])

    def test_rarefaction_matches_enumerated_discovery_probability(self):
        # Baseline: one success in four attempts; sample two => P(discovery)=1/2.
        frame = pd.DataFrame(
            {
                "setting": [BASELINE] * 4 + ["target"] * 2,
                "scenario": [0] * 6,
                "adversarial": [1, 0, 0, 0, 1, 0],
                "redacted": [0] * 6,
                "errored": [0] * 6,
                "usable": [True] * 6,
            }
        )
        result = matched_budget_overlap(frame, "target", k=2, draws=1000)
        self.assertAlmostEqual(result["expected_baseline_failure_scenarios_at_k"], 0.5)
        self.assertAlmostEqual(result["expected_target_only_fraction_at_k"], 0.5)
        self.assertEqual(result["target_only_vs_full_baseline"], 0)
        with self.assertRaises(ValueError):
            matched_budget_overlap(frame, "target", k=3)

    def test_session_success_distinguishes_concentrated_failures(self):
        rows = []
        for model, setting, outcomes in [
            ("baseline", BASELINE, [1, 0, 0]),
            ("a", "a", [1, 1, 0, 0, 0, 0]),
            ("b", "b", [1, 0, 0, 1, 0, 0]),
        ]:
            for i, outcome in enumerate(outcomes):
                rows.append(
                    {
                        "model": model,
                        "setting": setting,
                        "scenario": 0,
                        "repetition": i // 3,
                        "adversarial": outcome,
                        "redacted": 0,
                        "errored": 0,
                        "usable": True,
                    }
                )
        _, summary, _, baseline_three = session_summaries(pd.DataFrame(rows))
        self.assertAlmostEqual(summary.loc["a", "attempt_yield"], 1 / 3)
        self.assertAlmostEqual(summary.loc["b", "attempt_yield"], 1 / 3)
        self.assertEqual(summary.loc["a", "session_success"], 0.5)
        self.assertEqual(summary.loc["b", "session_success"], 1.0)
        self.assertEqual(baseline_three, 1.0)


if __name__ == "__main__":
    unittest.main()
