"""Shared fixtures: no private configuration, model downloads, or network calls."""
import os
import json
import socket
from pathlib import Path
from unittest.mock import patch, create_autospec

import pytest

from digit.core.settings import Settings

# Import-time application configuration must not read the developer's .env/hosts.
# load_config itself is exercised with temporary configuration files in its tests.
with patch("digit.core.settings.load_config", return_value=Settings()):
    from digit.core.base import CONFIG

from digit.generator.providers.provider_adapter import ProviderAdapter
from digit.simulator.simulation_data import SimulationData, SimulationFrameData, SimulationMetadata
from tests.fakes import make_res_json, make_experiment_info, make_custom_prompts


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    # load_config writes directly to os.environ. Restore it even after failures.
    with patch.dict(os.environ):
        for name in ("CARLA_PATH", "CONDA_PATH", "OUTPUT_PATH", "CARLA_PORT",
                     "CARLA_TRAFFIC_MANAGER_PORT", "HOSTS_PATH"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(CONFIG, "generator", None)
        monkeypatch.setattr(CONFIG, "filter", None)
        monkeypatch.setattr(CONFIG, "simulator", None)
        monkeypatch.setattr(CONFIG, "active_model_instances", [])

        def unexpected_connection(*args, **kwargs):
            raise AssertionError("Tests must fake network boundaries")

        monkeypatch.setattr(socket.socket, "connect", unexpected_connection)
        yield


@pytest.fixture
def provider_adapter():
    adapter = create_autospec(ProviderAdapter, instance=True, spec_set=True)
    adapter.test_connection.return_value = True
    return adapter


@pytest.fixture
def simulation_data(tmp_path):
    frames = [SimulationFrameData(
        image_name=f"frame_{step}.jpg", timestamp=step / 20,
        metadata=SimulationMetadata(
            step=step, control={}, speed=10.0, position=[0, 0, 0],
            rotation=[0, 0, 0], forward_vector=[1, 0, 0], right_vector=[0, 1, 0],
            velocity=[0, 0, 0], angular_velocity=[0, 0, 0], acceleration=[0, 0, 0], scene=[],
        ),
    ) for step in (0, 20)]
    return SimulationData(image_dir=tmp_path, frame_count=len(frames), frames=frames)


@pytest.fixture
def job_dir(tmp_path: Path):
    job_path = tmp_path / "job_001"
    run_dir = job_path / "sce_005" / "run_000"
    run_dir.mkdir(parents=True)
    (run_dir / "res.json").write_text(json.dumps(make_res_json()))
    (run_dir / "custom_prompts.json").write_text(json.dumps(make_custom_prompts()))
    (run_dir / "experiment_info.json").write_text(json.dumps(make_experiment_info()))
    return job_path


@pytest.fixture
def cache_dir_with_ground_truth(tmp_path: Path):
    gt_dir = tmp_path / "cache" / "sce_005" / "run_000"
    gt_dir.mkdir(parents=True)
    (gt_dir / "res.json").write_text(json.dumps(make_res_json()))
    return tmp_path / "cache"
