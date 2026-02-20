"""
Factory helpers for creating test data objects.

Provides convenience functions for creating simulation results and other
data objects with sensible defaults that can be easily overridden.
"""
from typing import List, Optional

from untitled.simulator.simulation_result import (
    SimulationResult,
    SimulationStatus,
    SimulationInfractions,
    SimulationScores,
    SimulationMeta,
)


def make_simulation_infractions(
    collisions_layout: Optional[List[str]] = None,
    collisions_pedestrian: Optional[List[str]] = None,
    collisions_vehicle: Optional[List[str]] = None,
    red_light: Optional[List[str]] = None,
    stop_infraction: Optional[List[str]] = None,
    outside_route_lanes: Optional[List[str]] = None,
    min_speed_infractions: Optional[List[str]] = None,
    yield_emergency_vehicle_infractions: Optional[List[str]] = None,
    scenario_timeouts: Optional[List[str]] = None,
    route_dev: Optional[List[str]] = None,
    vehicle_blocked: Optional[List[str]] = None,
    route_timeout: Optional[List[str]] = None,
) -> SimulationInfractions:
    """
    Create a SimulationInfractions with sensible defaults (all empty lists).
    
    Args:
        All arguments are optional lists of infraction descriptions.
        If not provided, defaults to empty list.
        
    Returns:
        SimulationInfractions instance.
    """
    return SimulationInfractions(
        collisions_layout=collisions_layout or [],
        collisions_pedestrian=collisions_pedestrian or [],
        collisions_vehicle=collisions_vehicle or [],
        red_light=red_light or [],
        stop_infraction=stop_infraction or [],
        outside_route_lanes=outside_route_lanes or [],
        min_speed_infractions=min_speed_infractions or [],
        yield_emergency_vehicle_infractions=yield_emergency_vehicle_infractions or [],
        scenario_timeouts=scenario_timeouts or [],
        route_dev=route_dev or [],
        vehicle_blocked=vehicle_blocked or [],
        route_timeout=route_timeout or [],
    )


def make_simulation_scores(
    score_route: float = 100.0,
    score_penalty: float = 1.0,
    score_composed: float = 100.0,
) -> SimulationScores:
    """
    Create a SimulationScores with sensible defaults (perfect scores).
    
    Args:
        score_route: Route completion score (default 100.0 = fully completed).
        score_penalty: Penalty multiplier (default 1.0 = no penalties).
        score_composed: Composed score (default 100.0).
        
    Returns:
        SimulationScores instance.
    """
    return SimulationScores(
        score_route=score_route,
        score_penalty=score_penalty,
        score_composed=score_composed,
    )


def make_simulation_meta(
    route_length: float = 1000.0,
    duration_game: float = 60.0,
    duration_system: float = 5.0,
) -> SimulationMeta:
    """
    Create a SimulationMeta with sensible defaults.
    
    Args:
        route_length: Route length in meters.
        duration_game: Game/simulation duration in seconds.
        duration_system: Wall-clock duration in seconds.
        
    Returns:
        SimulationMeta instance.
    """
    return SimulationMeta(
        route_length=route_length,
        duration_game=duration_game,
        duration_system=duration_system,
    )


def make_simulation_result(
    scenario_id: int = 0,
    scenario_name: Optional[str] = None,
    status: SimulationStatus = SimulationStatus.PERFECT,
    infractions: Optional[SimulationInfractions] = None,
    scores: Optional[SimulationScores] = None,
    meta: Optional[SimulationMeta] = None,
) -> SimulationResult:
    """
    Create a SimulationResult with sensible defaults (perfect run).
    
    Args:
        scenario_id: Scenario identifier.
        scenario_name: Scenario name (defaults to "scenario_{id}").
        status: Simulation status (default PERFECT).
        infractions: Infractions (default empty).
        scores: Scores (default perfect).
        meta: Metadata (default reasonable values).
        
    Returns:
        SimulationResult instance.
        
    Examples:
        # Default perfect result
        result = make_simulation_result()
        
        # Failed result with collision
        result = make_simulation_result(
            status=SimulationStatus.FAILED,
            infractions=make_simulation_infractions(
                collisions_vehicle=["Collision at frame 100"]
            ),
            scores=make_simulation_scores(score_route=50.0, score_composed=25.0),
        )
    """
    return SimulationResult(
        scenario_id=scenario_id,
        scenario_name=scenario_name or f"scenario_{scenario_id}",
        status=status,
        infractions=infractions or make_simulation_infractions(),
        scores=scores or make_simulation_scores(),
        meta=meta or make_simulation_meta(),
    )


def make_failed_simulation_result(
    scenario_id: int = 0,
    scenario_name: Optional[str] = None,
    status: SimulationStatus = SimulationStatus.FAILED,
    failure_reason: str = "Generic failure",
    score_route: float = 50.0,
) -> SimulationResult:
    """
    Convenience function to create a failed SimulationResult.
    
    Args:
        scenario_id: Scenario identifier.
        scenario_name: Scenario name (defaults to "scenario_{id}").
        status: Failure status (default FAILED).
        failure_reason: Description of failure (added to vehicle_blocked).
        score_route: Route completion percentage before failure.
        
    Returns:
        SimulationResult instance with failure status.
    """
    return make_simulation_result(
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        status=status,
        infractions=make_simulation_infractions(vehicle_blocked=[failure_reason]),
        scores=make_simulation_scores(
            score_route=score_route,
            score_penalty=0.5,
            score_composed=score_route * 0.5,
        ),
    )
