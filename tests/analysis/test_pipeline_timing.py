import unittest

from stats.pipeline_timing import seconds, simulate_pipeline


class PipelineTests(unittest.TestCase):
    def test_time_parsing(self):
        self.assertEqual(seconds("1:55:7"), 6907)
        self.assertEqual(seconds("8:08:17"), 29297)
        with self.assertRaises(ValueError):
            seconds("00:75:00")

    def test_single_chain_cannot_overlap_its_dependencies(self):
        summary, _ = simulate_pipeline(
            scenarios=1,
            generation_seconds=[2, 3],
            simulation_seconds=5,
            generation_workers=10,
            simulation_workers=10,
        )
        self.assertEqual(summary["makespan_seconds"], 15)

    def test_two_chains_hand_computed_schedule(self):
        summary, _ = simulate_pipeline(
            scenarios=2, generation_seconds=[2, 3], simulation_seconds=5
        )
        self.assertEqual(summary["makespan_seconds"], 22)

    def test_batch_arrivals_and_feedback_order(self):
        summary, rows = simulate_pipeline(
            scenarios=2,
            generation_seconds=[2, 3],
            simulation_seconds=5,
            first_batch_seconds=7,
        )
        # S0: 7-12, 12-17; G1: 12-15, 17-20; S1: 17-22, 22-27.
        self.assertEqual(summary["makespan_seconds"], 27)
        self.assertFalse(
            any(r["kind"] == "generation" and r["iteration"] == 0 for r in rows)
        )

    def test_workers_never_overlap_and_feedback_is_ready(self):
        for batch in [None, 4]:
            _, rows = simulate_pipeline(
                scenarios=7,
                repetitions=2,
                generation_seconds=[2, 7, 4],
                simulation_seconds=3,
                generation_workers=3,
                simulation_workers=2,
                first_batch_seconds=batch,
                first_batch_spread=0 if batch is None else 10,
            )
            tasks = {
                (r["scenario"], r["repetition"], r["iteration"], r["kind"]): r
                for r in rows
            }
            for r in rows:
                key = (r["scenario"], r["repetition"])
                if r["kind"] == "simulation":
                    parent_kind = (
                        "batch_response"
                        if batch is not None and r["iteration"] == 0
                        else "generation"
                    )
                    parent = tasks[(*key, r["iteration"], parent_kind)]
                    self.assertGreaterEqual(r["start_seconds"], parent["end_seconds"])
                elif r["kind"] == "generation" and r["iteration"] > 0:
                    parent = tasks[(*key, r["iteration"] - 1, "simulation")]
                    self.assertGreaterEqual(r["start_seconds"], parent["end_seconds"])
            for kind in ["generation", "simulation"]:
                for worker in range(3):
                    assigned = sorted(
                        [
                            r
                            for r in rows
                            if r["kind"] == kind and r["worker"] == worker
                        ],
                        key=lambda r: r["start_seconds"],
                    )
                    for previous, current in zip(assigned, assigned[1:]):
                        self.assertLessEqual(
                            previous["end_seconds"], current["start_seconds"]
                        )

    def test_repetitions_preserve_job_count(self):
        summary, _ = simulate_pipeline(repetitions=3)
        self.assertEqual(summary["simulations"], 225)


if __name__ == "__main__":
    unittest.main()
