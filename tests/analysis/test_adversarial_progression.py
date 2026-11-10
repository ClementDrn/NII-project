"""Check progression estimands, feedback order, and finite-pool allocation."""

from itertools import combinations
import unittest

import numpy as np
import pandas as pd

from stats.adversarial_analysis import BASELINE
from stats.adversarial_progression import failure_progression


def fixture():
    rows = []
    for scenario, capacity in [(0, 10), (1, 13)]:
        for index in range(capacity):
            rows.append(
                dict(
                    setting=BASELINE,
                    scenario=scenario,
                    repetition=index % 2,
                    run=index // 2,
                    adversarial=int(index in (0, 3)),
                )
            )
    for method in ("one-shot", "incremental"):
        for repetition in range(3):
            for scenario in (0, 1):
                for run in range(3):
                    rows.append(
                        dict(
                            setting=f"claude-opus-4.6 / {method}",
                            repetition=repetition,
                            scenario=scenario,
                            run=run,
                            adversarial=int(run == 1),
                        )
                    )
    return pd.DataFrame(rows)


class ProgressionTests(unittest.TestCase):
    def test_matched_allocation_partial_blocks_and_exhaustion(self):
        curves, allocation = failure_progression(fixture(), [0, 1])
        for attempts, expected in [
            (6, [3, 3]),
            (12, [6, 6]),
            (18, [9, 9]),
            (22, [10, 12]),
            (23, [10, 13]),
        ]:
            counts = allocation.head(attempts).groupby("scenario").size()
            self.assertEqual(counts.tolist(), expected)
        self.assertEqual(allocation.scenario.head(6).tolist(), [0, 0, 0, 1, 1, 1])
        self.assertFalse(allocation.duplicated(["scenario", "scenario_attempt"]).any())
        for _, group in curves.groupby("setting"):
            self.assertTrue(group.attempts.eq(np.arange(len(group))).all())
            self.assertTrue(group.cumulative_failures.diff().dropna().ge(-1e-12).all())
            self.assertTrue(group.discovered_scenarios.diff().dropna().ge(-1e-12).all())
        baseline = curves[curves.setting.eq(BASELINE)].iloc[-1]
        self.assertEqual(baseline.cumulative_failures, 4)
        self.assertEqual(baseline.discovered_scenarios, 2)

    def test_exact_expectations_agree_with_enumerated_subsets(self):
        frame = fixture()
        curves, _ = failure_progression(frame, [0, 1])
        # At attempt 2 only scenario 0 is sampled: enumerate all size-2 subsets.
        pool = frame.loc[
            frame.setting.eq(BASELINE) & frame.scenario.eq(0), "adversarial"
        ]
        subsets = list(combinations(pool, 2))
        expected_failures = np.mean([sum(s) for s in subsets])
        expected_discovery = np.mean([any(s) for s in subsets])
        point = curves.loc[curves.setting.eq(BASELINE) & curves.attempts.eq(2)].iloc[0]
        self.assertAlmostEqual(point.cumulative_failures, expected_failures)
        self.assertAlmostEqual(point.discovered_scenarios, expected_discovery)

    def test_digit_order_ignores_input_row_order_and_preserves_sessions(self):
        frame = fixture().sample(frac=1, random_state=17)
        curves, _ = failure_progression(frame, [0, 1])
        target = curves[curves.setting.eq("claude-opus-4.6 / incremental")]
        np.testing.assert_array_equal(
            target.cumulative_failures,
            [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6],
        )
        self.assertEqual(target.discovered_scenarios.iloc[-1], 2)

    def test_zero_failure_pools(self):
        frame = fixture()
        frame.adversarial = 0
        curves, _ = failure_progression(frame, [0, 1])
        self.assertTrue(curves.cumulative_failures.eq(0).all())
        self.assertTrue(curves.discovered_scenarios.eq(0).all())

    def test_rejects_incomplete_grids_duplicate_keys_and_missing_outcomes(self):
        frame = fixture()
        for invalid in (
            frame.iloc[:-1],
            pd.concat([frame, frame.iloc[[0]]]),
            frame.assign(adversarial=np.nan),
        ):
            with self.assertRaises(ValueError):
                failure_progression(invalid, [0, 1])
        with self.assertRaises(ValueError):
            failure_progression(frame, [0, 2])


if __name__ == "__main__":
    unittest.main()
