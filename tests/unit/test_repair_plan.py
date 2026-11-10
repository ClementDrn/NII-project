import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/run/rerun_incremental_experiments.py"
)
spec = importlib.util.spec_from_file_location("repair_runs", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "payload", [{}, {"220": [1]}, {"1": [-1]}, {"1": [1, 1]}, {"1": []}]
)
def test_invalid_repair_plans_are_rejected(tmp_path, payload):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        module.load_plan(path)


def test_dry_run_preserves_existing_results_and_uses_current_paths(
    tmp_path, monkeypatch, capsys
):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"40": [1]}))
    experiment = tmp_path / "experiment"
    target = experiment / "job_002/sce_040/run_001"
    target.mkdir(parents=True)
    (target / "sentinel.txt").write_text("existing result")
    cache = tmp_path / "cache"
    (cache / "sce_040/run_000").mkdir(parents=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--experiment-dir",
            str(experiment),
            "--plan",
            str(plan),
            "--cache-dir",
            str(cache),
            "--filter-id",
            "filter",
            "--generator-id",
            "generator",
            "--model-instance-id",
            "model",
            "--job-index",
            "2",
            "--dry-run",
        ],
    )
    assert module.main() == 0
    output = capsys.readouterr().out
    assert "scripts/vlm/generate_next_instruction.py" in output
    assert "scripts/run/run_simulation.py" in output
    assert "--job-index 2" in output
    assert (target / "sentinel.txt").read_text() == "existing result"
    assert not list(target.parent.glob("*.backup-*"))
    assert not list(target.parent.glob("rerun_instruction*"))
