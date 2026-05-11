"""
Fake Simulator for testing.

Provides a duck-typed replacement for the real Simulator that:
- Returns preset results instead of running actual simulations
- Tracks all method calls for test verification
- Requires no CARLA installation or environment
"""
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any

from untitled.generator.instruction import Instruction
from untitled.simulator.simulation_result import (
    SimulationResult,
    SimulationStatus,
    SimulationInfractions,
    SimulationScores,
    SimulationMeta,
)


def _default_simulation_result(scenario_id: int = 0) -> SimulationResult:
    """Create a default successful simulation result."""
    return SimulationResult(
        scenario_id=scenario_id,
        scenario_name=f"scenario_{scenario_id}",
        status=SimulationStatus.PERFECT,
        infractions=SimulationInfractions(
            collisions_layout=[],
            collisions_pedestrian=[],
            collisions_vehicle=[],
            red_light=[],
            stop_infraction=[],
            outside_route_lanes=[],
            min_speed_infractions=[],
            yield_emergency_vehicle_infractions=[],
            scenario_timeouts=[],
            route_dev=[],
            vehicle_blocked=[],
            route_timeout=[],
        ),
        scores=SimulationScores(
            score_route=100.0,
            score_penalty=1.0,
            score_composed=100.0,
        ),
        meta=SimulationMeta(
            route_length=1000.0,
            duration_game=60.0,
            duration_system=5.0,
        ),
    )


class FakeSimulator:
    """
    A fake Simulator that returns preset results without running actual simulations.
    
    Usage:
        fake = FakeSimulator()
        fake.preset_run_id = 0
        fake.preset_results[(job_id, scenario_id, run_id)] = my_result
        
        # Use fake in place of real Simulator
        run_id = fake.run_scenario(instruction)
        result = fake.get_result(job_id, scenario_id, run_id)
        
        # Verify calls
        assert len(fake.run_scenario_calls) == 1
        assert fake.run_scenario_calls[0]["instruction"] == instruction
        
    With directory creation (for filesystem tests):
        fake = FakeSimulator(base_dir=tmp_path, create_directories=True)
        fake.run_scenario()  # Creates directories in tmp_path
    """
    
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        create_directories: bool = False,
    ):
        """
        Initialize the fake simulator.
        
        Args:
            base_dir: Base directory for simulated output paths (optional).
            create_directories: If True, actually create directories on run_scenario().
        """
        # Preset values for controlled test behavior
        self.preset_run_id: Optional[int] = None  # If set, overrides auto-increment
        self.preset_results: Dict[Tuple[int, int, int], SimulationResult] = {}
        
        # Configuration
        self.create_directories = create_directories
        
        # Call tracking for verification
        self.run_scenario_calls: List[Dict[str, Any]] = []
        self.get_result_calls: List[Dict[str, Any]] = []
        self.set_job_context_calls: List[Dict[str, Any]] = []
        self.set_scenario_calls: List[int] = []
        self.free_calls: int = 0
        
        # State (mirroring real Simulator interface)
        self.base_dir: Path = base_dir or Path("/fake/base/dir")
        self.base_job_subdir: str = ""
        self._job_id: Optional[int] = None
        self._scenario_id: Optional[int] = None
        self._run_counter: int = 0
        
        # Track run IDs per (job_id, scenario_id, subdir) for auto-incrementing
        self._run_ids: Dict[Tuple[int, int, str], int] = {}
    
    def run_scenario(self, instruction: Optional[Instruction] = None, run_id: Optional[int] = None) -> int:
        """
        Fake running a scenario. Returns run_id and tracks the call.
        
        If preset_run_id is set, always returns that value.
        Otherwise, auto-increments run_id per (job_id, scenario_id) pair.
        
        If create_directories is True, creates the run directory.
        
        Args:
            instruction: Optional instruction (recorded but not used).
            
        Returns:
            The run_id for this scenario execution.
        """
        # Determine run_id
        if run_id is not None:
            run_id = run_id
        elif self.preset_run_id is not None:
            run_id = self.preset_run_id
        else:
            key = (self._job_id, self._scenario_id, self.base_job_subdir)
            run_id = self._run_ids.get(key, 0)
            self._run_ids[key] = run_id + 1

        if run_id is not None:
            key = (self._job_id, self._scenario_id, self.base_job_subdir)
            self._run_ids[key] = max(self._run_ids.get(key, 0), run_id + 1)
        
        self.run_scenario_calls.append({
            "instruction": instruction,
            "job_id": self._job_id,
            "scenario_id": self._scenario_id,
            "run_id": run_id,
        })
        self._run_counter += 1
        
        # Create directory if configured
        if self.create_directories and self._job_id is not None and self._scenario_id is not None:
            run_dir = Path(self._calculate_run_base_dir(
                self.base_dir,
                self._job_id,
                self._scenario_id,
                run_id,
                has_job_subdir=True,
            ))
            run_dir.mkdir(parents=True, exist_ok=True)
        
        return run_id
    
    def get_result(
        self,
        job_id: int,
        scenario_id: int,
        run_id: int,
        run_base_dir: Optional[Path] = None,
    ) -> Optional[SimulationResult]:
        """
        Get a preset simulation result, or a default successful result.
        
        Args:
            job_id: Job identifier.
            scenario_id: Scenario identifier.
            run_id: Run identifier.
            run_base_dir: Optional base directory (recorded but not used).
            
        Returns:
            Preset SimulationResult if configured, otherwise a default PERFECT result.
        """
        self.get_result_calls.append({
            "job_id": job_id,
            "scenario_id": scenario_id,
            "run_id": run_id,
            "run_base_dir": run_base_dir,
        })
        
        key = (job_id, scenario_id, run_id)
        if key in self.preset_results:
            return self.preset_results[key]
        
        return _default_simulation_result(scenario_id)
    
    def set_job_context(self, job_id: int, scenario_id: int) -> None:
        """Set the job context (tracks call and updates internal state)."""
        self.set_job_context_calls.append({
            "job_id": job_id,
            "scenario_id": scenario_id,
        })
        self._job_id = job_id
        self._scenario_id = scenario_id
    
    def set_scenario(self, scenario_id: int) -> None:
        """Set the current scenario (tracks call)."""
        self.set_scenario_calls.append(scenario_id)
        self._scenario_id = scenario_id
    
    def set_base_job_subdir(self, subdir: str) -> None:
        """Set the base job subdirectory."""
        self.base_job_subdir = subdir
    
    def get_run_dir(self, scenario_id: int, run_id: int, has_job_subdir: bool = True) -> Path:
        """
        Get the directory for a specific scenario and run.
        
        Args:
            scenario_id: The scenario ID number
            run_id: The run ID number
            has_job_subdir: Whether to include the job subdirectory (defaults to True)

        Returns:
            Path to the run directory
        """
        return Path(self._calculate_run_base_dir(
            self.base_dir,
            self._job_id,
            scenario_id,
            run_id,
            has_job_subdir,
        ))
    
    def _calculate_run_base_dir(
        self,
        base_dir: Path,
        job_id: int,
        scenario_id: int,
        run_id: int,
        has_job_subdir: bool = True,
    ) -> str:
        """
        Calculate the run base directory path (mirrors real Simulator logic).
        """
        path = Path(base_dir) / f"job_{job_id:03d}"
        if has_job_subdir and self.base_job_subdir:
            path = path / self.base_job_subdir
        path = path / f"sce_{scenario_id:03d}" / f"run_{run_id:03d}"
        return str(path)
    
    def free(self) -> None:
        """Free resources (tracks call count)."""
        self.free_calls += 1
    
    def reset_tracking(self) -> None:
        """Reset all call tracking (useful between test cases)."""
        self.run_scenario_calls.clear()
        self.get_result_calls.clear()
        self.set_job_context_calls.clear()
        self.set_scenario_calls.clear()
        self.free_calls = 0
        self._run_counter = 0
        self._run_ids.clear()
