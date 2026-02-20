"""
Shared pytest fixtures for the test suite.

This module is automatically discovered by pytest and provides fixtures
for all tests in the tests/ directory and its subdirectories.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.fakes import (
    FakeSimulator,
    FakeModelClient,
    ModelType,
    make_simulation_result,
)


# =============================================================================
# Fake/Stub Fixtures
# =============================================================================

@pytest.fixture
def fake_simulator(tmp_path: Path) -> FakeSimulator:
    """
    Provide a FakeSimulator instance with a real temp directory for output.
    
    The fake simulator is pre-configured with:
    - base_dir pointing to a real temporary directory
    - Default preset_run_id of 0
    - Empty preset_results (will return default PERFECT results)
    
    Usage:
        def test_something(fake_simulator):
            fake_simulator.preset_run_id = 5
            result = code_under_test(fake_simulator)
            assert fake_simulator.run_scenario_calls[0]["instruction"] is not None
    """
    return FakeSimulator(base_dir=tmp_path / "simulation_output")


@pytest.fixture
def fake_simulator_with_fs(temp_base_dir: Path) -> FakeSimulator:
    """
    Provide a FakeSimulator that creates actual directories.
    
    This fixture is for tests that need to verify filesystem operations.
    The simulator's base_dir points to temp_base_dir/output and will
    create directories when run_scenario() is called.
    
    Usage:
        def test_directory_creation(fake_simulator_with_fs, temp_base_dir):
            fake_simulator_with_fs.set_job_context(1, 1)
            fake_simulator_with_fs.run_scenario()
            assert (temp_base_dir / "output" / "job_001" / "sce_001" / "run_000").exists()
    """
    return FakeSimulator(
        base_dir=temp_base_dir / "output",
        create_directories=True,
    )


@pytest.fixture
def fake_model_client() -> FakeModelClient:
    """
    Provide a FakeModelClient instance ready for testing.
    
    The fake model client is pre-configured with:
    - Default preset_responses containing one generic response
    - VLM model type
    - State: DISABLED (call start() or open_connection() to activate)
    
    Usage:
        @pytest.mark.asyncio
        async def test_generation(fake_model_client):
            fake_model_client.preset_responses = ["turn left", "accelerate"]
            await fake_model_client.open_connection()
            response = await fake_model_client.prompt("sys", "user", [])
            assert response == "turn left"
    """
    return FakeModelClient(model_type=ModelType.VLM)


@pytest.fixture
def fake_llm_client() -> FakeModelClient:
    """
    Provide a FakeModelClient configured as an LLM (no vision).
    """
    return FakeModelClient(model_type=ModelType.LLM)


# =============================================================================
# Mock Fixtures (for patching global state)
# =============================================================================

@pytest.fixture
def mock_config():
    """
    Provide a MagicMock replacement for the global CONFIG object.
    
    This patches untitled.core.base.CONFIG to avoid CARLA environment
    validation and other config-dependent behavior during tests.
    
    Usage:
        def test_with_config(mock_config):
            mock_config.simulator.debug = True
            mock_config.carla_env.path = "/fake/carla"
            # Code under test will see mocked config
    """
    with patch("untitled.core.base.CONFIG") as mock:
        # Set up common nested attributes
        mock.carla_env = MagicMock()
        mock.carla_env.path = "/fake/carla/path"
        mock.carla_env.conda_path = "/fake/conda/path"
        mock.carla_env.output_path = Path("/fake/output")
        
        mock.simulator = MagicMock()
        mock.simulator.debug = False
        
        mock.generator = MagicMock()
        
        yield mock


# =============================================================================
# Filesystem Fixtures
# =============================================================================

@pytest.fixture
def temp_base_dir(tmp_path: Path) -> Path:
    """
    Provide a temporary base directory with standard simulation structure.
    
    Creates the following structure:
        tmp_path/
            cache/
            logs/
            output/
    
    Usage:
        def test_file_operations(temp_base_dir):
            result_file = temp_base_dir / "cache" / "sce_001" / "run_000" / "res.json"
            result_file.parent.mkdir(parents=True)
            result_file.write_text("{}")
    """
    (tmp_path / "cache").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "output").mkdir()
    return tmp_path


@pytest.fixture
def temp_scenario_dir(tmp_path: Path) -> Path:
    """
    Provide a temporary directory mimicking a scenario run directory.
    
    Creates:
        tmp_path/sce_001/run_000/
            images/
            metadata/
    
    Returns the run_000 directory path.
    """
    run_dir = tmp_path / "sce_001" / "run_000"
    (run_dir / "images").mkdir(parents=True)
    (run_dir / "metadata").mkdir(parents=True)
    return run_dir


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def sample_perfect_result():
    """Provide a sample SimulationResult with PERFECT status."""
    return make_simulation_result()


@pytest.fixture
def sample_failed_result():
    """Provide a sample SimulationResult with FAILED status."""
    from untitled.simulator.simulation_result import SimulationStatus
    from tests.fakes import make_simulation_infractions, make_simulation_scores
    
    return make_simulation_result(
        status=SimulationStatus.FAILED,
        infractions=make_simulation_infractions(
            collisions_vehicle=["Collision with vehicle at frame 150"]
        ),
        scores=make_simulation_scores(
            score_route=75.0,
            score_penalty=0.5,
            score_composed=37.5,
        ),
    )
