import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "save_experiment_data.py"
spec = importlib.util.spec_from_file_location("save_experiment_data", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def export_job(job_dir, cache_dir_with_ground_truth):
    run_dir = job_dir / "sce_005" / "run_000"
    prompt = {"text": "Follow the red car ahead", "injection_time": 1, "injection_duration": 2}
    (run_dir / "custom_prompts.json").write_text(json.dumps([prompt]))
    return job_dir, cache_dir_with_ground_truth, run_dir


def write_metadata(run_dir, step, metadata):
    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / f"{step:06d}.json").write_text(json.dumps(metadata))


def read_csv(path):
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        return reader.fieldnames, rows


def test_default_export_preserves_columns_and_skips_checks(export_job, tmp_path, monkeypatch):
    job_dir, cache_dir, _ = export_job
    output = tmp_path / "default.csv"

    def unexpected_check(*args):
        pytest.fail("Hallucination checks must be opt-in")

    monkeypatch.setattr(module, "check_hallucinations_in_instruction", unexpected_check)
    assert module.save_experiment_data(job_dir, output, cache_dir) == 1
    fields, rows = read_csv(output)
    assert fields == module.FIELDNAMES
    assert [rows[0][field] for field in ["model", "method", "repetition"]] == ["N/A"] * 3
    assert rows[0]["start"] == "1"
    assert rows[0]["end"] == "3.0"


def test_export_hallucinations_and_labels(export_job, tmp_path):
    job_dir, cache_dir, run_dir = export_job
    output = tmp_path / "checked.csv"
    metadata = {"scene_objects": [
        {"class": "car", "color_name": "red", "position": [-10, 0, 0]},
        {"class": "car", "color_name": "blue", "position": [10, 0, 0]},
    ]}
    write_metadata(run_dir, 20, metadata)

    # This instruction cannot be satisfied by combining two different cars.
    assert module.save_experiment_data(
        job_dir, output, cache_dir, hallucinations=True,
        model="test-model", method="one-shot", repetition="2",
    ) == 1
    fields, rows = read_csv(output)
    assert fields == module.FIELDNAMES + module.HALLUCINATION_FIELDNAMES
    assert [rows[0][field] for field in ["model", "method", "repetition"]] == ["test-model", "one-shot", "2"]
    assert [rows[0][field] for field in module.HALLUCINATION_FIELDNAMES] == ["1", "2", "0", "1"]


@pytest.mark.parametrize("source", ["experiment", "local_ground_truth", "cache"])
def test_metadata_at_injection_uses_latest_prior_frame(export_job, source):
    job_dir, cache_dir, run_dir = export_job
    source_dirs = {
        "experiment": run_dir,
        "local_ground_truth": job_dir / "ground_truths/sce_005/run_000",
        "cache": cache_dir / "sce_005/run_000",
    }
    metadata_dir = source_dirs[source]
    write_metadata(metadata_dir, 5, {"scene_objects": [], "selected": "earlier"})
    write_metadata(metadata_dir, 15, {"scene_objects": [], "selected": "latest"})
    write_metadata(metadata_dir, 25, {"scene_objects": [], "selected": "future"})
    result = module._load_injection_metadata(run_dir, job_dir, cache_dir, 5, 0, 1)
    assert result["selected"] == "latest"


def test_experiment_metadata_takes_precedence(export_job):
    job_dir, cache_dir, run_dir = export_job
    write_metadata(run_dir, 15, {"scene": [], "selected": "experiment"})
    write_metadata(cache_dir / "sce_005/run_000", 20, {"scene": [], "selected": "cache"})
    result = module._load_injection_metadata(run_dir, job_dir, cache_dir, 5, 0, 1)
    assert result["selected"] == "experiment"


def test_ground_truth_index_from_experiment_info(export_job):
    job_dir, cache_dir, run_dir = export_job
    (run_dir / "experiment_info.json").write_text(json.dumps({"ground_truth_index": 1}))
    write_metadata(cache_dir / "sce_005/run_001", 20, {"scene_objects": [], "selected": 1})
    result = module._load_injection_metadata(run_dir, job_dir, cache_dir, 5, None, 1)
    assert result["selected"] == 1


def test_missing_metadata_exports_na(export_job, tmp_path):
    job_dir, cache_dir, _ = export_job
    output = tmp_path / "missing.csv"
    module.save_experiment_data(job_dir, output, cache_dir, hallucinations=True)
    _, rows = read_csv(output)
    assert [rows[0][field] for field in module.HALLUCINATION_FIELDNAMES] == ["N/A"] * 4


def test_cli_options_are_forwarded(export_job, tmp_path, monkeypatch):
    job_dir, cache_dir, _ = export_job
    output = tmp_path / "cli.csv"
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT_PATH), str(job_dir.parent), str(output), "--gt-dir", str(cache_dir),
        "--hallucinations", "--model", "my-model", "--method", "incremental", "--repetition", "3",
    ])
    module.main()
    fields, rows = read_csv(tmp_path / "cli_job_001.csv")
    assert fields == module.FIELDNAMES + module.HALLUCINATION_FIELDNAMES
    assert [rows[0][field] for field in ["model", "method", "repetition"]] == ["my-model", "incremental", "3"]
