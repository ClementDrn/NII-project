import unittest

import numpy as np
import pandas as pd

from stats.adversarial_analysis import BASELINE, matched_budget_overlap
from stats.budget_novelty import (
    SIM_SECONDS,
    baseline_budget_counts,
    budget_comparisons,
    conditional_novelty,
    timing_schedules,
)


def fixture():
    rows = []
    for setting in [
        BASELINE,
        "claude-opus-4.6 / one-shot",
        "claude-opus-4.6 / incremental",
    ]:
        for scenario in [0, 1]:
            for rep in range(3):
                for run in range(3):
                    rows.append(
                        dict(
                            setting=setting,
                            scenario=scenario,
                            repetition=rep,
                            run=run,
                            adversarial=int(rep == 0 and run == 0),
                            usable=True,
                            redacted=0,
                            errored=0,
                        )
                    )
    return pd.DataFrame(rows)


class BudgetTests(unittest.TestCase):
    def test_round_robin_deadline_and_finite_pool(self):
        frame = fixture()
        self.assertEqual(baseline_budget_counts(frame, [0, 1], 13).sum(), 0)
        self.assertEqual(
            baseline_budget_counts(frame, [0, 1], 14 + SIM_SECONDS - 0.01).sum(), 0
        )
        self.assertEqual(
            baseline_budget_counts(frame, [0, 1], 14 + 3 * SIM_SECONDS).tolist(), [2, 1]
        )
        self.assertEqual(baseline_budget_counts(frame, [0, 1], 1e8).tolist(), [9, 9])

    def test_same_attempt_expectation_agrees_with_previous_analysis(self):
        frame = fixture()
        target = "claude-opus-4.6 / one-shot"
        original = matched_budget_overlap(frame, target, k=9, draws=50)
        updated = conditional_novelty(
            frame,
            target,
            frame[frame.setting == target],
            pd.Series(9, index=[0, 1]),
            draws=50,
        )
        self.assertAlmostEqual(
            original["expected_target_only_fraction_at_k"],
            updated["expected_novelty_fraction"],
        )
        with self.assertRaises(ValueError):
            conditional_novelty(
                frame,
                target,
                frame[frame.setting == target],
                pd.Series(10, index=[0, 1]),
            )

    def test_empty_target_discoveries_have_undefined_fraction(self):
        frame = fixture()
        result = conditional_novelty(
            frame, "unused", frame.iloc[:0], pd.Series(0, index=[0, 1]), draws=50
        )
        self.assertTrue(np.isnan(result["expected_novelty_fraction"]))
        self.assertEqual(result["expected_target_only_scenarios"], 0)

    def test_225_timing_and_repetition_modes(self):
        table, schedules = timing_schedules(list(range(25)))
        self.assertTrue(table.attempts.eq(225).all())
        times = table.set_index("schedule").hours
        self.assertAlmostEqual(times["baseline"], (14 + 225 * SIM_SECONDS) / 3600)
        self.assertAlmostEqual(times["incremental_sequential"], 80.70208333333333)
        self.assertGreater(
            times["incremental_pipeline_1"], times["incremental_joint_pipeline_1"]
        )
        for _, schedule in schedules.values():
            self.assertFalse(
                schedule.duplicated(["scenario", "repetition", "run"]).any()
            )
            self.assertEqual(set(schedule.repetition), {0, 1, 2})

    def test_common_deadline_censors_targets_and_preserves_attempt_matching(self):
        frame = fixture()
        _, schedules = timing_schedules([0, 1])
        result = budget_comparisons(
            frame, [0, 1], schedules, common_deadline_hours=0, draws=50
        )
        same_time = result[result.comparison == "same_time_common_deadline"]
        self.assertTrue(same_time.target_completed_attempts.eq(0).all())
        self.assertTrue(same_time.baseline_completed_attempts.eq(0).all())
        attempts = result[result.comparison == "same_attempt_count"]
        self.assertTrue(attempts.target_completed_attempts.eq(18).all())
        self.assertTrue(attempts.baseline_completed_attempts.eq(18).all())


if __name__ == "__main__":
    unittest.main()
