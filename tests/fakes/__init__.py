"""
Test fakes/stubs for expensive or external dependencies.

This module provides lightweight replacements for components that are:
- Slow (e.g., CARLA simulator runs)
- Expensive (e.g., LLM/VLM API calls)
- Require external setup (e.g., CARLA installation, model servers)

Usage:
    from tests.fakes import FakeSimulator, FakeModelClient
    from tests.fakes import make_simulation_result, make_simulation_infractions
    
    # Create fake simulator with preset results
    simulator = FakeSimulator()
    simulator.preset_results[(0, 1, 0)] = make_simulation_result(
        status=SimulationStatus.FAILED
    )
    
    # Create fake model client with preset responses
    model = FakeModelClient()
    model.preset_responses = ["drive forward", "turn left"]
"""
from tests.fakes.simulator import FakeSimulator
from tests.fakes.model_client import FakeModelClient, ModelType
from tests.fakes.factories import (
    make_simulation_result,
    make_simulation_infractions,
    make_simulation_scores,
    make_simulation_meta,
    make_failed_simulation_result,
    # Job results factories
    make_res_json,
    make_experiment_info,
    make_custom_prompts,
)

__all__ = [
    # Fake implementations
    "FakeSimulator",
    "FakeModelClient",
    "ModelType",
    # Factory helpers
    "make_simulation_result",
    "make_simulation_infractions",
    "make_simulation_scores",
    "make_simulation_meta",
    "make_failed_simulation_result",
    # Job results factories
    "make_res_json",
    "make_experiment_info",
    "make_custom_prompts",
]
