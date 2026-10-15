from enum import Enum
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from untitled.core.base import LOG


class ExperimentResult(Enum):
    """Types of experiments based on their effect compared to ground truth."""
    NONE = 0
    UNCHANGED = 1
    ADVERSARIAL = 2
    CORRECTIVE = 3
    ERROR = 4


@dataclass
class RunResult:
    """Result of a single simulation run."""
    scenario_id: int
    run_id: int
    score_penalty: float
    score_route: float
    score_composed: float
    status: str
    infractions_count: int
    collisions_count: int
    ground_truth_index: Optional[int] = None    # can be None if no ground truth available for this experiment, or if it is itself a ground truth run
    prompt_text: Optional[str] = None   # ground truths do not have prompt text, only experiments
    t_start: Optional[float] = None
    t_end: Optional[float] = None

    @property
    def is_error(self) -> bool:
        """
        Return True when the simulator recorded a failed run.
        If agent got blocked, or deviated from the route, it is not considered an error.
        """
        status = (self.status or "").strip()
        if not status:
            return False

        # Check if status is "Failed"
        is_failed = "failed" in status.lower()
        # Ignore some failures
        is_route_deviation = "deviated from the route" in status.lower()
        is_blocked = "agent got blocked" in status.lower()

        return is_failed and not is_route_deviation #and not is_blocked

    @property
    def is_agent_blocked(self) -> bool:
        """Return True if the agent got blocked during the run."""
        status = (self.status or "").strip()
        if not status:
            return False
        return "agent got blocked" in status.lower()
    
    @property
    def is_agent_timed_out(self) -> bool:
        """Return True if the agent timed out during the run."""
        status = (self.status or "").strip()
        if not status:
            return False
        return "agent timed out" in status.lower()


@dataclass
class ScenarioResults:
    """Results for comparing experiments to a ground truth run of a single scenario."""
    scenario_id: int
    ground_truths: Dict[int, RunResult] = field(default_factory=dict)
    experiments: List[RunResult] = field(default_factory=list)
    
    def classify_run(self, exp: RunResult) -> ExperimentResult:
        """
        Classify if an experiment is adversarial, corrective, unchanged, or error.
        
        Returns:
            ExperimentResult.ADVERSARIAL if experiment penalty < ground truth penalty (worse)
            ExperimentResult.CORRECTIVE if experiment penalty > ground truth penalty (better)
            ExperimentResult.UNCHANGED if penalties are equal
            ExperimentResult.ERROR if the simulator recorded a failed run
            ExperimentResult.NONE if comparison not possible
        """
        if exp is None:
            return ExperimentResult.NONE

        if exp.is_error and not exp.is_agent_timed_out:
            return ExperimentResult.ERROR

        if exp.ground_truth_index is None:
            return ExperimentResult.NONE
        
        ground_truth = self.ground_truths.get(exp.ground_truth_index)
        if ground_truth is None:
            return ExperimentResult.NONE
        
        # Blocked counts as an infraction
        #if exp.is_agent_blocked:
        #    return ExperimentResult.ADVERSARIAL

        # Check penalty score
        if ground_truth.score_penalty > exp.score_penalty:
            return ExperimentResult.ADVERSARIAL
        elif ground_truth.score_penalty < exp.score_penalty:
            return ExperimentResult.CORRECTIVE

        # If penalties are equal, but there was a timeout, consider an error
        if exp.is_agent_timed_out:
            return ExperimentResult.ERROR
        
        return ExperimentResult.UNCHANGED

    def get_summary(self) -> Dict[str, int]:
        """
        Get summary counts of adversarial/corrective/unchanged/error runs.
        """
        summary = {
            "total_experiments": len(self.experiments),
            "adversarial": 0,
            "corrective": 0,
            "unchanged": 0,
            "errors": 0,
        }

        for exp in self.experiments:
            classification = self.classify_run(exp)
            if classification == ExperimentResult.ADVERSARIAL:
                summary["adversarial"] += 1
            elif classification == ExperimentResult.CORRECTIVE:
                summary["corrective"] += 1
            elif classification == ExperimentResult.UNCHANGED:
                summary["unchanged"] += 1
            elif classification == ExperimentResult.ERROR:
                summary["errors"] += 1

        return summary


class JobResults:
    """Parser for job results from saved simulation data."""
    
    def __init__(self, job_dir: Path, ground_truth_cache_dir: Path = ""):
        """
        Initialize job results parser.
        
        Args:
            job_dir: Path to job directory with experiments (e.g., output/26-01-01/job_000/)
            ground_truth_cache_dir: Path to cache directory with ground truths (e.g., cache/)
        """
        self._job_dir: Path = job_dir
        self._ground_truth_cache_dir: Path = ground_truth_cache_dir
        self._ground_truth_dir: Path = job_dir / "ground_truths"    # exists if ground truths were saved in job_dir, otherwise we will look in cache
        self.scenarios: Dict[int, ScenarioResults] = {}
        
        self._load_results()
    
    def _load_results(self) -> None:
        """Load all results from the job directory and cache."""
        if not self._job_dir.exists():
            raise FileNotFoundError(f"Job directory not found: {self._job_dir}")
        
        # Find all scenario directories in job_dir (sce_XXX)
        for sce_dir in sorted(self._job_dir.glob("sce_*")):
            if not sce_dir.is_dir():
                continue
            
            # Parse scenario ID from directory name
            try:
                scenario_id = int(sce_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            
            # Ground truths for this scenario: { ground_truth_index -> run_result } 
            ground_truths: Dict[int, RunResult] = {}
            
            experiments: List[RunResult] = []
            
            # Find all run directories in job_dir (run_XXX) - all are experiments
            for run_dir in sorted(sce_dir.glob("run_*")):
                if not run_dir.is_dir():
                    continue
                
                try:
                    run_id = int(run_dir.name.split("_")[1])
                except (IndexError, ValueError):
                    continue

                # Load ground truth index from experiment_info.json if available, otherwise default to index 0
                # experiment_info.json holds information about which ground truth run to compare against
                ground_truth_index = 0
                exp_info_file = run_dir / "experiment_info.json"
                if exp_info_file.exists():
                    try:
                        with open(exp_info_file, 'r', encoding='utf-8') as f:
                            exp_info = json.load(f)
                            ground_truth_index = exp_info.get("ground_truth_index", 0)
                    except (json.JSONDecodeError, IOError):
                        pass
                
                # Load ground truth based on ground_truth_index if not already loaded for this scenario
                ground_truth = None
                if ground_truth_index in ground_truths:
                    # Ground truth already loaded for this index
                    ground_truth = ground_truths[ground_truth_index]
                else:
                    # Try loading from job_dir/ground_truths first
                    gt_run_dir = self._ground_truth_dir / f"sce_{scenario_id:03d}" / f"run_{ground_truth_index:03d}"
                    if gt_run_dir.exists():
                        gt_result = self._load_run_result(scenario_id, ground_truth_index, gt_run_dir)
                        if gt_result:
                            ground_truths[ground_truth_index] = gt_result
                            ground_truth = gt_result
                        else:
                            ground_truth = None
                    else:
                        # Fallback to cache directory
                        gt_cache_dir = self._ground_truth_cache_dir / f"sce_{scenario_id:03d}" / f"run_{ground_truth_index:03d}"
                        if gt_cache_dir.exists():
                            gt_result = self._load_run_result(scenario_id, ground_truth_index, gt_cache_dir)
                            if gt_result:
                                ground_truths[ground_truth_index] = gt_result
                                ground_truth = gt_result
                            else:
                                ground_truth = None
                        else:
                            ground_truth = None
                    if ground_truth is None:
                        LOG.warning(f"Ground truth not found for scenario {scenario_id}, ground_truth_index {ground_truth_index}")
                # Save ground truth
                if ground_truth is not None:
                    ground_truths[ground_truth_index] = ground_truth
                
                # Load experiment result
                experiment_ground_truth_index = ground_truth_index if ground_truth is not None else None
                result = self._load_run_result(scenario_id, run_id, run_dir, ground_truth_index=experiment_ground_truth_index)
                if result:
                    experiments.append(result)
            
            # Save ground truths and experiments for this scenario
            if ground_truths or experiments:
                self.scenarios[scenario_id] = ScenarioResults(
                    scenario_id=scenario_id,
                    ground_truths=ground_truths,
                    experiments=experiments
                )

    
    def _load_run_result(self, scenario_id: int, run_id: int, run_dir: Path, ground_truth_index: Optional[int] = None) -> Optional[RunResult]:
        """Load result from a single run directory."""
        res_file = run_dir / "res.json"
        prompts_file = run_dir / "custom_prompts.json"
        
        if not res_file.exists():
            return None
        
        try:
            with open(res_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
        
        # Extract scores from checkpoint records
        checkpoint = data.get("_checkpoint", {})
        records = checkpoint.get("records", [])
        
        if not records:
            return None
        
        record = records[0]
        scores = record.get("scores", {})
        infractions = record.get("infractions", {})
        
        # Count total infractions (excluding min_speed)
        infractions_count = sum(
            len(v) if isinstance(v, list) else 0
            for k, v in infractions.items()
            if k != "min_speed_infractions"
        )
        
        # Count collisions
        collisions_count = (
            len(infractions.get("collisions_vehicle", [])) +
            len(infractions.get("collisions_pedestrian", [])) +
            len(infractions.get("collisions_layout", []))
        )
        
        # Load prompt text and timing metadata
        prompt_text = None
        t_start = None
        t_end = None
        if prompts_file.exists():
            try:
                with open(prompts_file, 'r', encoding='utf-8') as f:
                    prompts = json.load(f)
                if prompts and isinstance(prompts, list) and len(prompts) > 0:
                    prompt = prompts[0]
                    prompt_text = prompt.get("text", "")
                    injection_time = prompt.get("injection_time")
                    injection_duration = prompt.get("injection_duration")
                    if injection_time is not None:
                        t_start = float(injection_time)
                    if injection_duration is not None:
                        t_end = t_start + float(injection_duration) if t_start is not None else None
            except (json.JSONDecodeError, IOError, TypeError, ValueError):
                pass
        
        return RunResult(
            scenario_id=scenario_id,
            run_id=run_id,
            ground_truth_index=ground_truth_index,
            score_penalty=scores.get("score_penalty", 1.0),
            score_route=scores.get("score_route", 0.0),
            score_composed=scores.get("score_composed", 0.0),
            status=record.get("status", "Unknown"),
            infractions_count=infractions_count,
            collisions_count=collisions_count,
            prompt_text=prompt_text,
            t_start=t_start,
            t_end=t_end,
        )
    
    def get_scenario(self, scenario_id: int) -> Optional[ScenarioResults]:
        """Get results for a specific scenario."""
        return self.scenarios.get(scenario_id)
    
    def get_all_scenarios(self) -> List[ScenarioResults]:
        """Get all scenario results sorted by scenario ID."""
        return [self.scenarios[sid] for sid in sorted(self.scenarios.keys())]
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary counts of adversarial/corrective/unchanged/error runs."""
        adversarial = 0
        corrective = 0
        unchanged = 0
        errors = 0
        total_experiments = 0
        
        for scenario in self.scenarios.values():
            summary = scenario.get_summary()
            adversarial += summary["adversarial"]
            corrective += summary["corrective"]
            unchanged += summary["unchanged"]
            errors += summary["errors"]
            total_experiments += summary["total_experiments"]
        
        return {
            'total_experiments': total_experiments,
            'adversarial': adversarial,
            'corrective': corrective,
            'unchanged': unchanged,
            'errors': errors,
        }

