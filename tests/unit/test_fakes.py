"""
Tests for the fake/stub implementations.

Verifies that the test doubles work correctly and provide
the expected interface for use in other tests.
"""
import pytest
from pathlib import Path

from tests.fakes import (
    FakeSimulator,
    FakeModelClient,
    ModelType,
    make_simulation_result,
    make_simulation_infractions,
    make_simulation_scores,
    make_simulation_meta,
    make_failed_simulation_result,
)
from untitled.simulator.simulation_result import SimulationStatus


class TestFakeSimulator:
    """Tests for FakeSimulator."""
    
    def test_run_scenario_returns_preset_run_id(self):
        """run_scenario should return the preset run_id."""
        fake = FakeSimulator()
        fake.preset_run_id = 42
        
        result = fake.run_scenario(instruction=None)
        
        assert result == 42
    
    def test_run_scenario_tracks_calls(self):
        """run_scenario should record each call with parameters."""
        fake = FakeSimulator()
        fake.set_job_context(job_id=1, scenario_id=5)
        
        fake.run_scenario(instruction=None)
        fake.run_scenario(instruction="test instruction")
        
        assert len(fake.run_scenario_calls) == 2
        assert fake.run_scenario_calls[0]["instruction"] is None
        assert fake.run_scenario_calls[0]["job_id"] == 1
        assert fake.run_scenario_calls[0]["scenario_id"] == 5
        assert fake.run_scenario_calls[1]["instruction"] == "test instruction"
    
    def test_get_result_returns_preset_result(self):
        """get_result should return preset result when configured."""
        fake = FakeSimulator()
        custom_result = make_simulation_result(
            scenario_id=5,
            status=SimulationStatus.FAILED,
        )
        fake.preset_results[(1, 5, 0)] = custom_result
        
        result = fake.get_result(job_id=1, scenario_id=5, run_id=0)
        
        assert result is custom_result
        assert result.status == SimulationStatus.FAILED
    
    def test_get_result_returns_default_when_no_preset(self):
        """get_result should return default PERFECT result when no preset."""
        fake = FakeSimulator()
        
        result = fake.get_result(job_id=0, scenario_id=3, run_id=0)
        
        assert result is not None
        assert result.status == SimulationStatus.PERFECT
        assert result.scenario_id == 3
    
    def test_get_result_tracks_calls(self):
        """get_result should record each call with parameters."""
        fake = FakeSimulator()
        
        fake.get_result(job_id=1, scenario_id=2, run_id=3)
        
        assert len(fake.get_result_calls) == 1
        assert fake.get_result_calls[0] == {
            "job_id": 1,
            "scenario_id": 2,
            "run_id": 3,
            "run_base_dir": None,
        }
    
    def test_set_job_context_updates_state(self):
        """set_job_context should update internal state and track calls."""
        fake = FakeSimulator()
        
        fake.set_job_context(job_id=10, scenario_id=20)
        
        assert fake._job_id == 10
        assert fake._scenario_id == 20
        assert len(fake.set_job_context_calls) == 1
    
    def test_calculate_run_base_dir_with_subdir(self):
        """_calculate_run_base_dir should include subdir when set."""
        fake = FakeSimulator()
        fake.set_base_job_subdir("experiments")
        
        result = fake._calculate_run_base_dir(
            base_dir=Path("/base"),
            job_id=1,
            scenario_id=5,
            run_id=0,
            has_job_subdir=True,
        )
        
        assert result == "/base/job_001/experiments/sce_005/run_000"
    
    def test_calculate_run_base_dir_without_subdir(self):
        """_calculate_run_base_dir should skip subdir when has_job_subdir=False."""
        fake = FakeSimulator()
        fake.set_base_job_subdir("experiments")
        
        result = fake._calculate_run_base_dir(
            base_dir=Path("/base"),
            job_id=1,
            scenario_id=5,
            run_id=0,
            has_job_subdir=False,
        )
        
        assert result == "/base/job_001/sce_005/run_000"
    
    def test_free_tracks_calls(self):
        """free should increment call counter."""
        fake = FakeSimulator()
        
        fake.free()
        fake.free()
        
        assert fake.free_calls == 2
    
    def test_reset_tracking_clears_all(self):
        """reset_tracking should clear all call history."""
        fake = FakeSimulator()
        fake.run_scenario(instruction=None)
        fake.get_result(job_id=0, scenario_id=0, run_id=0)
        fake.set_job_context(job_id=0, scenario_id=0)
        fake.set_scenario(scenario_id=0)
        fake.free()
        
        fake.reset_tracking()
        
        assert len(fake.run_scenario_calls) == 0
        assert len(fake.get_result_calls) == 0
        assert len(fake.set_job_context_calls) == 0
        assert len(fake.set_scenario_calls) == 0
        assert fake.free_calls == 0


class TestFakeModelClient:
    """Tests for FakeModelClient."""
    
    @pytest.mark.asyncio
    async def test_prompt_returns_preset_response(self):
        """prompt should return the preset response."""
        fake = FakeModelClient()
        fake.preset_responses = ["Hello, world!"]
        
        response = await fake.prompt("system", "user", [])
        
        assert response == "Hello, world!"
    
    @pytest.mark.asyncio
    async def test_prompt_cycles_through_responses(self):
        """prompt should cycle through preset responses."""
        fake = FakeModelClient()
        fake.preset_responses = ["first", "second", "third"]
        
        r1 = await fake.prompt("s", "u", [])
        r2 = await fake.prompt("s", "u", [])
        r3 = await fake.prompt("s", "u", [])
        r4 = await fake.prompt("s", "u", [])  # Should wrap around
        
        assert r1 == "first"
        assert r2 == "second"
        assert r3 == "third"
        assert r4 == "first"
    
    @pytest.mark.asyncio
    async def test_prompt_uses_callback_when_provided(self):
        """prompt should use callback for dynamic responses."""
        fake = FakeModelClient()
        fake.response_callback = lambda sys, usr, imgs: f"Echo: {usr}"
        
        response = await fake.prompt("system", "hello", [])
        
        assert response == "Echo: hello"
    
    @pytest.mark.asyncio
    async def test_prompt_tracks_calls(self):
        """prompt should record each call with parameters."""
        fake = FakeModelClient()
        image_paths = [Path("/img1.jpg"), Path("/img2.jpg")]
        
        await fake.prompt("sys text", "user text", image_paths)
        
        assert len(fake.prompt_calls) == 1
        assert fake.prompt_calls[0]["system_text"] == "sys text"
        assert fake.prompt_calls[0]["user_text"] == "user text"
        assert fake.prompt_calls[0]["user_image_paths"] == image_paths
    
    @pytest.mark.asyncio
    async def test_start_updates_state(self):
        """start should update state to STARTED."""
        fake = FakeModelClient()
        assert fake.state == FakeModelClient.State.DISABLED
        
        await fake.start()
        
        assert fake.state == FakeModelClient.State.STARTED
        assert fake.start_calls == 1
    
    @pytest.mark.asyncio
    async def test_open_connection_updates_state(self):
        """open_connection should update state to CONNECTED."""
        fake = FakeModelClient()
        
        await fake.open_connection()
        
        assert fake.state == FakeModelClient.State.CONNECTED
        assert fake.open_connection_calls == 1
    
    @pytest.mark.asyncio
    async def test_open_connection_starts_if_disabled(self):
        """open_connection should call start if currently disabled."""
        fake = FakeModelClient()
        
        await fake.open_connection()
        
        assert fake.start_calls == 1
    
    @pytest.mark.asyncio
    async def test_stop_updates_state(self):
        """stop should update state to DISABLED."""
        fake = FakeModelClient()
        await fake.start()
        
        await fake.stop()
        
        assert fake.state == FakeModelClient.State.DISABLED
        assert fake.stop_calls == 1
    
    def test_is_remote_returns_false(self):
        """is_remote should always return False for fake."""
        fake = FakeModelClient()
        assert fake.is_remote() is False
    
    @pytest.mark.asyncio
    async def test_test_connection_returns_true(self):
        """test_connection should always return True for fake."""
        fake = FakeModelClient()
        assert await fake.test_connection() is True
    
    @pytest.mark.asyncio
    async def test_reset_tracking_clears_all(self):
        """reset_tracking should clear all call history."""
        fake = FakeModelClient()
        await fake.start()
        await fake.stop()
        await fake.open_connection()
        await fake.prompt("s", "u", [])
        
        fake.reset_tracking()
        
        assert len(fake.prompt_calls) == 0
        assert fake.start_calls == 0
        assert fake.stop_calls == 0
        assert fake.open_connection_calls == 0


class TestFactories:
    """Tests for factory helper functions."""
    
    def test_make_simulation_infractions_defaults_to_empty(self):
        """make_simulation_infractions should create empty lists by default."""
        infractions = make_simulation_infractions()
        
        assert infractions.collisions_layout == []
        assert infractions.collisions_vehicle == []
        assert infractions.red_light == []
        assert infractions.total_count == 0
    
    def test_make_simulation_infractions_accepts_overrides(self):
        """make_simulation_infractions should accept specific values."""
        infractions = make_simulation_infractions(
            collisions_vehicle=["Collision 1", "Collision 2"],
            red_light=["Red light at frame 50"],
        )
        
        assert len(infractions.collisions_vehicle) == 2
        assert len(infractions.red_light) == 1
        assert infractions.total_count == 3
    
    def test_make_simulation_scores_defaults_to_perfect(self):
        """make_simulation_scores should create perfect scores by default."""
        scores = make_simulation_scores()
        
        assert scores.score_route == 100.0
        assert scores.score_penalty == 1.0
        assert scores.score_composed == 100.0
    
    def test_make_simulation_scores_accepts_overrides(self):
        """make_simulation_scores should accept specific values."""
        scores = make_simulation_scores(
            score_route=75.0,
            score_penalty=0.8,
            score_composed=60.0,
        )
        
        assert scores.score_route == 75.0
        assert scores.score_penalty == 0.8
        assert scores.score_composed == 60.0
    
    def test_make_simulation_meta_has_defaults(self):
        """make_simulation_meta should create reasonable defaults."""
        meta = make_simulation_meta()
        
        assert meta.route_length == 1000.0
        assert meta.duration_game == 60.0
        assert meta.duration_system == 5.0
    
    def test_make_simulation_result_creates_perfect_by_default(self):
        """make_simulation_result should create a PERFECT result by default."""
        result = make_simulation_result()
        
        assert result.scenario_id == 0
        assert result.scenario_name == "scenario_0"
        assert result.status == SimulationStatus.PERFECT
        assert result.was_route_completed is True
        assert result.infractions.total_count == 0
    
    def test_make_simulation_result_accepts_overrides(self):
        """make_simulation_result should accept all overrides."""
        custom_infractions = make_simulation_infractions(red_light=["violation"])
        
        result = make_simulation_result(
            scenario_id=42,
            scenario_name="custom_scenario",
            status=SimulationStatus.COMPLETED,
            infractions=custom_infractions,
        )
        
        assert result.scenario_id == 42
        assert result.scenario_name == "custom_scenario"
        assert result.status == SimulationStatus.COMPLETED
        assert result.infractions.total_count == 1
    
    def test_make_failed_simulation_result(self):
        """make_failed_simulation_result should create a failed result."""
        result = make_failed_simulation_result(
            scenario_id=5,
            failure_reason="Vehicle blocked by obstacle",
            score_route=30.0,
        )
        
        assert result.status == SimulationStatus.FAILED
        assert result.status.is_failure is True
        assert result.scores.score_route == 30.0
        assert "Vehicle blocked by obstacle" in result.infractions.vehicle_blocked


class TestConftest:
    """Tests for fixtures defined in conftest.py."""
    
    def test_fake_simulator_fixture(self, fake_simulator):
        """fake_simulator fixture should provide working FakeSimulator."""
        assert isinstance(fake_simulator, FakeSimulator)
        assert fake_simulator.base_dir is not None
    
    def test_fake_model_client_fixture(self, fake_model_client):
        """fake_model_client fixture should provide working FakeModelClient."""
        assert isinstance(fake_model_client, FakeModelClient)
        assert fake_model_client.model_type == ModelType.VLM
    
    def test_temp_base_dir_fixture(self, temp_base_dir):
        """temp_base_dir fixture should create expected structure."""
        assert (temp_base_dir / "cache").exists()
        assert (temp_base_dir / "logs").exists()
        assert (temp_base_dir / "output").exists()
    
    def test_temp_scenario_dir_fixture(self, temp_scenario_dir):
        """temp_scenario_dir fixture should create expected structure."""
        assert temp_scenario_dir.name == "run_000"
        assert (temp_scenario_dir / "images").exists()
        assert (temp_scenario_dir / "metadata").exists()
    
    def test_sample_perfect_result_fixture(self, sample_perfect_result):
        """sample_perfect_result fixture should provide PERFECT result."""
        assert sample_perfect_result.status == SimulationStatus.PERFECT
    
    def test_sample_failed_result_fixture(self, sample_failed_result):
        """sample_failed_result fixture should provide FAILED result."""
        assert sample_failed_result.status == SimulationStatus.FAILED
        assert sample_failed_result.infractions.total_count > 0
