"""Test the real simulator filesystem and run indexing without launching CARLA."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from digit.core.base import CONFIG
from digit.core.settings import CARLAEnvConfig
from digit.simulator import simulator as simulator_module
from digit.simulator.simulator import Simulator


@pytest.fixture
def simulator(tmp_path, monkeypatch):
    benchmark = tmp_path / 'benchmark'
    routes = tmp_path / 'routes'
    routes.mkdir()
    for index in (1, 2):
        (routes / f'bench2drive_{index:02d}.xml').write_text('<routes></routes>')
    (benchmark / 'leaderboard/leaderboard').mkdir(parents=True)
    (benchmark / 'leaderboard/leaderboard/leaderboard_evaluator.py').touch()
    (benchmark / 'scenario_runner').mkdir()
    monkeypatch.setattr(simulator_module, 'BENCHMARK_ROOT', benchmark)
    monkeypatch.setattr(simulator_module, 'ROUTES_ROOT', routes)
    monkeypatch.setattr(CONFIG, 'carla_env', CARLAEnvConfig(
        CARLA_PATH=tmp_path / 'carla', CONDA_PATH=str(tmp_path / 'conda'),
        OUTPUT_PATH=tmp_path / 'output', CARLA_PORT=2000, CARLA_TRAFFIC_MANAGER_PORT=2050,
    ))
    ads = MagicMock()
    ads.agent_name = 'test_agent'
    ads.get_agent_args.return_value = []
    ads.get_agent_python_path.return_value = tmp_path / 'agent'
    return Simulator(ads=ads, debug=False, base_dir=tmp_path / 'output')


@pytest.mark.parametrize(('subdir', 'include_subdir', 'expected'), [
    ('', True, 'job_123/sce_001/run_002'),
    ('ground_truths', True, 'job_123/ground_truths/sce_001/run_002'),
    ('ground_truths', False, 'job_123/sce_001/run_002'),
])
def test_run_directory_layout(simulator, subdir, include_subdir, expected):
    simulator.set_base_job_subdir(Path(subdir))
    simulator.set_job_context(123, 1)
    assert simulator.get_run_dir(1, 2, has_job_subdir=include_subdir) == simulator.base_dir / expected


def test_run_ids_are_independent_per_job_scenario_and_subdir(simulator):
    simulator.set_job_context(123, 1)
    simulator.set_scenario(1)
    assert [simulator._next_run_id(), simulator._next_run_id()] == [0, 1]
    simulator.set_job_context(123, 2)
    simulator.set_scenario(2)
    assert simulator._next_run_id() == 0
    simulator.set_job_context(123, 1)
    simulator.set_scenario(1)
    assert simulator._next_run_id() == 2
    simulator.set_base_job_subdir(Path('ground_truths'))
    assert simulator._next_run_id() == 0
    simulator.set_job_context(124, 1)
    assert simulator._next_run_id() == 0


def test_run_script_creates_real_output_hierarchy_and_prompt(simulator):
    import json
    from digit.generator.instruction import Instruction, InjectionWindow

    simulator.set_job_context(123, 1)
    simulator.set_scenario(1)
    instruction = Instruction(rank=1, text='Turn left', reasoning='test', target_module_logic='planning',
                              injection_window=InjectionWindow(start_timestamp=1, end_timestamp=3))
    script = Path(simulator._create_run_script(instruction))
    run_dir = simulator.get_run_dir(1, 0)
    assert script.is_file()
    assert (run_dir / 'images').is_dir()
    assert (run_dir / 'metadata').is_dir()
    prompts = json.loads((run_dir / 'custom_prompts.json').read_text())
    assert prompts[0]['text'] == 'Turn left'
    assert prompts[0]['injection_time'] == 1
    assert prompts[0]['injection_duration'] == 2


@pytest.mark.parametrize('scenario_id', [-1, 220])
def test_out_of_range_scenario_is_rejected(simulator, scenario_id):
    with pytest.raises(ValueError, match='out of valid range'):
        simulator.set_scenario(scenario_id)
