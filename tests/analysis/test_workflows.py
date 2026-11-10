"""Regression and integration checks for the portable analysis workflows."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from stats.calculations import effectiveness
from stats.cli import COMMANDS, run_analysis
from stats.config import load_config
from stats.naturalness import load_sample, sampled_estimates, selected_examples
from stats.scene import analyze_scene
from stats.validation import load_results

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "scripts/stats/paper.yaml"


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(PROFILE)
        cls.frame = load_results(cls.config)

    def test_raw_success_on_unusable_attempt_is_preserved_but_not_counted(self):
        gpt = self.frame.loc[
            self.frame.model.eq("gpt-5.2") & self.frame.method.eq("incremental")
        ]
        self.assertEqual(gpt.adversarial_raw.sum(), 4)
        self.assertEqual(gpt.adversarial.sum(), 1)
        self.assertEqual(len(gpt), 225)
        self.assertTrue(gpt.loc[~gpt.usable, "adversarial"].eq(0).all())

    def test_both_holm_families_match_known_sensitive_comparison(self):
        table = effectiveness(self.frame, self.config)["configuration_comparisons"]
        self.assertTrue(table.family_size.eq(30).all())
        self.assertTrue(table.endpoint_family_size.eq(15).all())
        row = table.loc[
            table.target.eq("claude-opus-4.6 / one-shot")
            & table.reference.eq("gpt-5.2 / one-shot")
            & table.metric.eq("adversarial_rate")
        ].iloc[0]
        self.assertAlmostEqual(row.p_t_Holm_all, 0.051931, places=6)
        self.assertAlmostEqual(row.p_t_Holm_endpoint, 0.033755, places=6)

    def test_baseline_estimands_remain_distinct(self):
        baseline = effectiveness(self.frame, self.config)["all_model_summary"].loc[
            "baseline / rule-based"
        ]
        self.assertAlmostEqual(baseline.mean_repetition_rate * 100, 9.158883, places=6)
        self.assertAlmostEqual(baseline.mean_adversarial_rate * 100, 9.183535, places=6)
        self.assertAlmostEqual(baseline.sd_repetition_rate * 100, 0.112800, places=6)

    def test_result_loader_rejects_missing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            config = deepcopy(self.config)
            config["results"] = [str(ROOT / "data" / p) for p in config["results"]]
            original = pd.read_csv(config["results"][1])
            path = Path(directory) / "missing.csv"
            original.iloc[1:].to_csv(path, index=False)
            config["results"][1] = str(path)
            with self.assertRaisesRegex(ValueError, "run grid"):
                load_results(config)

    def test_profile_rejects_invalid_timing_and_duplicate_scenarios(self):
        import yaml

        for key, value in [("scenarios", [0, 0]), ("runs_per_session", 0)]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                config = deepcopy(self.config)
                config[key] = value
                path = Path(directory) / "profile.yaml"
                path.write_text(yaml.safe_dump(config))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_sample_uses_explicit_methods_and_keeps_duplicate_text_frequency(self):
        scores, human = load_sample(self.config)
        self.assertEqual(len(scores), 80)
        self.assertEqual(scores.instruction.nunique(), 78)
        self.assertEqual(human.user_id.nunique(), 11)
        self.assertEqual(
            scores.groupby("method").size().to_dict(), {"baseline": 40, "digit": 40}
        )
        examples = selected_examples(scores)
        self.assertEqual(examples.instruction_id.tolist(), [23, 74, 4, 19, 65, 67])

    def test_sample_rejects_fractional_missing_duplicate_and_unknown_ratings(self):
        spec = self.config["naturalness"]
        for problem in [
            "fractional",
            "missing_item",
            "duplicate",
            "unknown",
            "incomplete_human",
        ]:
            with (
                self.subTest(problem=problem),
                tempfile.TemporaryDirectory() as directory,
            ):
                config = deepcopy(self.config)
                config["data_dir"] = directory
                for key in ["instructions", "human", "llm"]:
                    shutil.copyfile(
                        ROOT / "data" / spec[key], Path(directory) / spec[key]
                    )
                key = "human" if problem == "incomplete_human" else "llm"
                path = Path(directory) / spec[key]
                panel = pd.read_csv(path, encoding="utf-8-sig")
                if problem == "fractional":
                    panel["rating"] = panel.rating.astype(float)
                    panel.loc[0, "rating"] = 2.5
                elif problem in {"missing_item", "incomplete_human"}:
                    panel = panel.iloc[1:]
                elif problem == "duplicate":
                    panel = pd.concat([panel, panel.iloc[:1]])
                else:
                    panel.loc[0, "instruction_id"] = 9999
                panel.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    load_sample(config)

    def test_finite_population_interval_and_paired_discrepancy(self):
        scores = pd.DataFrame(
            {
                "method": ["baseline"] * 2 + ["digit"] * 2,
                "human_mean": [1.0, 3.0, 3.0, 5.0],
                "llm_score": [1.0, 3.0, 3.0, 5.0],
                "bias": [0.0, 0.0, 0.0, 0.0],
            }
        )
        report = sampled_estimates(scores, {"baseline": 4, "digit": 4})
        row = report.loc["human_difference"]
        self.assertAlmostEqual(row.estimate, 2.0)
        self.assertAlmostEqual(row.CI_95_low, 2 - stats.t.ppf(0.975, 2))
        self.assertAlmostEqual(row.CI_95_high, 2 + stats.t.ppf(0.975, 2))
        discrepancy = report.loc["comparison_error_E"]
        np.testing.assert_array_equal(discrepancy.to_numpy(), [0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            sampled_estimates(scores, {"baseline": 1, "digit": 4})

    def test_cli_from_another_directory_uses_explicit_paths_and_no_model_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analysis"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stats/analyze.py"),
                    "effectiveness",
                    "--config",
                    str(PROFILE),
                    "--data-dir",
                    str(ROOT / "data"),
                    "--output-dir",
                    str(output),
                    "--no-figures",
                ],
                cwd=directory,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / "tables/table_I.csv").exists())
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import stats.cli, sys; "
                "assert not any(x in sys.modules for x in ['torch','spacy','litellm','digit'])",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_command_exports_results_and_preserves_inputs(self):
        paths = list((ROOT / "data").glob("*.csv"))
        hashes = {p: sha256(p.read_bytes()).hexdigest() for p in paths}
        with tempfile.TemporaryDirectory() as directory:
            for command in COMMANDS:
                with self.subTest(command=command):
                    output = Path(directory) / command
                    tables, figures = run_analysis(
                        command, self.config, output, figures=False
                    )
                    self.assertTrue(tables)
                    self.assertFalse(figures)
                    manifest = json.loads((output / "manifest.json").read_text())
                    self.assertTrue(manifest["inputs"])
                    self.assertTrue(manifest["analysis_sources"])
                    for filename in manifest["outputs"]:
                        self.assertTrue((output / filename).is_file(), filename)
                    # The manifest must inventory every generated output.
                    actual_files = {
                        str(path.relative_to(output))
                        for path in output.rglob("*")
                        if path.is_file()
                    }
                    self.assertEqual(set(manifest["outputs"]), actual_files)
        self.assertEqual(hashes, {p: sha256(p.read_bytes()).hexdigest() for p in paths})


class SceneTests(unittest.TestCase):
    def fixture(self):
        config = load_config(PROFILE)
        config["scenarios"] = [0, 1]
        rows = []
        types = "car motorbike bicycle pedestrian traffic_sign traffic_light road driving_lane parking_lane bike_lane shoulder sidewalk crosswalk intersection junction construction accident misc unknown".split()
        for model, method, offset in [
            ("baseline", "rule-based", 0),
            (config["selected_model"], "one-shot", 1),
            (config["selected_model"], "incremental", 1),
        ]:
            for scenario in [0, 1]:
                row = dict(
                    model=model,
                    method=method,
                    setting=f"{model} / {method}",
                    scenario=scenario,
                    usable=True,
                    object_count=1 + 2 * scenario + offset,
                    modifier_count=1,
                    object_hallucination_count=scenario,
                    modifier_hallucination_count=0,
                )
                row.update(dict.fromkeys(types, 0))
                row["car"] = row["object_count"]
                rows.append(row)
        return pd.DataFrame(rows), config

    def test_paired_bootstrap_retains_constant_difference_and_ratio_of_means(self):
        frame, config = self.fixture()
        tables = analyze_scene(frame, config)
        summary = tables["scene_relevance_summary"]
        self.assertEqual(
            summary.loc["baseline / rule-based", "object_hallucination_pct"], 25
        )
        contrasts = tables["scene_relevance_contrasts"]
        row = contrasts.loc[
            contrasts.target.eq(f"{config['selected_model']} / one-shot")
            & contrasts.reference.eq("baseline / rule-based")
            & contrasts.metric.eq("object_count")
        ].iloc[0]
        self.assertEqual(
            (row.mean_difference, row.CI_95_low, row.CI_95_high), (1.0, 1.0, 1.0)
        )

    def test_invalid_scene_annotations_are_not_silently_dropped(self):
        for problem in [
            "missing",
            "negative",
            "fractional",
            "flagged_exceeds_total",
            "type_sum",
        ]:
            frame, config = self.fixture()
            if problem == "missing":
                frame.loc[0, "object_count"] = np.nan
            elif problem == "negative":
                frame.loc[0, "object_count"] = -1
            elif problem == "fractional":
                frame["object_count"] = frame.object_count.astype(float)
                frame.loc[0, "object_count"] = 1.5
            elif problem == "flagged_exceeds_total":
                frame.loc[0, "object_hallucination_count"] = 2
            else:
                frame.loc[0, "unknown"] = 1
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                analyze_scene(frame, config)


if __name__ == "__main__":
    unittest.main()
