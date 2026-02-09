from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional
import json

from untitled.core.base import LOG


class SimulationStatus(Enum):
    """Status of a simulation run."""
    COMPLETED = "Completed"
    FAILED = "Failed"
    CRASHED = "Crashed"
    TIMEOUT = "Timeout"
    UNKNOWN = "Unknown"

    @classmethod
    def from_string(cls, status_str: str) -> "SimulationStatus":
        """Convert a status string to SimulationStatus enum."""
        # Take first word only, lowercase for comparison
        status_keyword = status_str.split()[0].lower()
        for status in cls:
            if status.value.lower() == status_keyword:
                return status
        LOG.warning(f"Unknown simulation status: '{status_str}'. Defaulting to UNKNOWN.")
        return cls.UNKNOWN


@dataclass
class SimulationInfractions:
    """Infractions recorded during simulation."""
    collisions_layout: List[str]
    collisions_pedestrian: List[str]
    collisions_vehicle: List[str]
    red_light: List[str]
    stop_infraction: List[str]
    outside_route_lanes: List[str]
    min_speed_infractions: List[str]
    yield_emergency_vehicle_infractions: List[str]
    scenario_timeouts: List[str]
    route_dev: List[str]
    vehicle_blocked: List[str]
    route_timeout: List[str]

    @classmethod
    def from_dict(cls, data: Dict) -> "SimulationInfractions":
        """Create SimulationInfractions from a dictionary."""
        return cls(
            collisions_layout=data.get("collisions_layout", []),
            collisions_pedestrian=data.get("collisions_pedestrian", []),
            collisions_vehicle=data.get("collisions_vehicle", []),
            red_light=data.get("red_light", []),
            stop_infraction=data.get("stop_infraction", []),
            outside_route_lanes=data.get("outside_route_lanes", []),
            min_speed_infractions=data.get("min_speed_infractions", []),
            yield_emergency_vehicle_infractions=data.get("yield_emergency_vehicle_infractions", []),
            scenario_timeouts=data.get("scenario_timeouts", []),
            route_dev=data.get("route_dev", []),
            vehicle_blocked=data.get("vehicle_blocked", []),
            route_timeout=data.get("route_timeout", []),
        )

    @property
    def total_count(self) -> int:
        """Return the total number of infractions."""
        return (
            len(self.collisions_layout) +
            len(self.collisions_pedestrian) +
            len(self.collisions_vehicle) +
            len(self.red_light) +
            len(self.stop_infraction) +
            len(self.outside_route_lanes) +
            len(self.min_speed_infractions) +
            len(self.yield_emergency_vehicle_infractions) +
            len(self.scenario_timeouts) +
            len(self.route_dev) +
            len(self.vehicle_blocked) +
            len(self.route_timeout)
        )


@dataclass
class SimulationScores:
    """Scores from simulation."""
    score_route: float
    score_penalty: float
    score_composed: float

    @classmethod
    def from_dict(cls, data: Dict) -> "SimulationScores":
        """Create SimulationScores from a dictionary."""
        return cls(
            score_route=data.get("score_route", 0.0),
            score_penalty=data.get("score_penalty", 0.0),
            score_composed=data.get("score_composed", 0.0),
        )


@dataclass
class SimulationMeta:
    """Metadata about the simulation run."""
    route_length: float
    duration_game: float
    duration_system: float

    @classmethod
    def from_dict(cls, data: Dict) -> "SimulationMeta":
        """Create SimulationMeta from a dictionary."""
        return cls(
            route_length=data.get("route_length", 0.0),
            duration_game=data.get("duration_game", 0.0),
            duration_system=data.get("duration_system", 0.0),
        )


@dataclass
class SimulationResult:
    """Result of a simulation run."""
    scenario_id: int
    scenario_name: str
    status: SimulationStatus
    infractions: SimulationInfractions
    scores: SimulationScores
    meta: SimulationMeta

    @property
    def was_route_completed(self) -> bool:
        """Return True if the route was fully completed (score_route == 100.0)."""
        return self.scores.score_route >= 100.0

    @property
    def failure_text(self) -> str:
        """
        Return a concatenated string of all infractions except min_speed_infractions.
        
        min_speed_infractions are excluded as they are too frequent and noisy.
        """
        infraction_texts = []
        
        if self.infractions.collisions_layout:
            infraction_texts.append("Collisions with layout:\n" + "\n".join(f"  - {inf}" for inf in self.infractions.collisions_layout))
        
        if self.infractions.collisions_pedestrian:
            infraction_texts.append("Collisions with pedestrians:\n" + "\n".join(f"  - {inf}" for inf in self.infractions.collisions_pedestrian))
        
        if self.infractions.collisions_vehicle:
            infraction_texts.append("Collisions with vehicles:\n" + "\n".join(f"  - {inf}" for inf in self.infractions.collisions_vehicle))
        
        if self.infractions.red_light:
            infraction_texts.append("Red light infractions:\n" + "\n".join(f"  - {inf}" for inf in self.infractions.red_light))
        
        if self.infractions.stop_infraction:
            infraction_texts.append("Stop sign infractions:\n" + "\n".join(f"  - {inf}" for inf in self.infractions.stop_infraction))
        
        if self.infractions.outside_route_lanes:
            infraction_texts.append("Outside route lanes:\n" + "\n".join(f"  - {inf}" for inf in self.infractions.outside_route_lanes))
        
        if self.infractions.yield_emergency_vehicle_infractions:
            infraction_texts.append("Yield to emergency vehicles:\n" + "\n".join(f"  - {inf}" for inf in self.infractions.yield_emergency_vehicle_infractions))
        
        if self.infractions.scenario_timeouts:
            infraction_texts.append("Scenario timeouts:\n" + "\n".join(f"  - {inf}" for inf in self.infractions.scenario_timeouts))
        
        if self.infractions.route_dev:
            infraction_texts.append("Route deviations:\n" + "\n".join(f"  - {inf}" for inf in self.infractions.route_dev))
        
        if self.infractions.vehicle_blocked:
            infraction_texts.append("Vehicle blocked:\n" + "\n".join(f"  - {inf}" for inf in self.infractions.vehicle_blocked))
        
        if self.infractions.route_timeout:
            infraction_texts.append("Route timeouts:\n" + "\n".join(f"  - {inf}" for inf in self.infractions.route_timeout))
        
        if not infraction_texts:
            return "No infractions recorded."
        
        return "\n\n".join(infraction_texts)

    @classmethod
    def load_from_json_file(cls, json_path: Path, scenario_id: int) -> Optional["SimulationResult"]:
        """
        Load simulation result from a res.json file.
        
        Args:
            json_path: Path to the res.json file
            scenario_id: The scenario ID (from Simulator.current_scenario_id)
            
        Returns:
            SimulationResult object, or None if the file cannot be read or parsed
        """
        if not json_path.exists():
            LOG.error(f"Result file not found: {json_path}")
            return None

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            LOG.error(f"Failed to read result file {json_path}: {e}")
            return None

        # Extract the first record from the checkpoint
        checkpoint = data.get("_checkpoint", {})
        records = checkpoint.get("records", [])
        
        if not records:
            LOG.warning(f"No records found in result file: {json_path}")
            return None

        record = records[0]

        return cls(
            scenario_id=scenario_id,
            scenario_name=record.get("scenario_name", ""),
            status=SimulationStatus.from_string(record.get("status", "Unknown")),
            infractions=SimulationInfractions.from_dict(record.get("infractions", {})),
            scores=SimulationScores.from_dict(record.get("scores", {})),
            meta=SimulationMeta.from_dict(record.get("meta", {})),
        )
