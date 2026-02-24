"""
Unit tests for untitled.data.job_results module.

Tests the JobResults parser which reads simulation results from disk.
Uses tmp_path fixtures with real files rather than mocking the filesystem.
"""
import json
import pytest
from pathlib import Path

from untitled.data.job_results import (
    ExperimentResult,
    RunResult,
    ScenarioResults,
    JobResults,
)
from tests.fakes import make_res_json, make_experiment_info, make_custom_prompts


# =============================================================================
# ScenarioResults.classify_run Tests
# =============================================================================

class TestScenarioResultsClassifyRun:
    """Tests for the classify_run method of ScenarioResults."""
    
    def test_returns_adversarial_when_experiment_penalty_lower_than_ground_truth(self):
        """Lower penalty = worse performance = adversarial."""
        gt = RunResult(scenario_id=1, run_id=0, score_penalty=1.0, score_route=100.0, score_composed=100.0, status="Perfect", infractions_count=0, collisions_count=0)
        exp = RunResult(scenario_id=1, run_id=1, ground_truth_index=0, score_penalty=0.5, score_route=100.0, score_composed=50.0, status="Failed", infractions_count=1, collisions_count=1)
        
        scenario = ScenarioResults(scenario_id=1, ground_truths={0: gt}, experiments=[exp])
        
        assert scenario.classify_run(exp) == ExperimentResult.ADVERSARIAL
    
    def test_returns_corrective_when_experiment_penalty_higher_than_ground_truth(self):
        """Higher penalty = better performance = corrective."""
        gt = RunResult(scenario_id=1, run_id=0, score_penalty=0.5, score_route=100.0, score_composed=50.0, status="Failed", infractions_count=1, collisions_count=1)
        exp = RunResult(scenario_id=1, run_id=1, ground_truth_index=0, score_penalty=1.0, score_route=100.0, score_composed=100.0, status="Perfect", infractions_count=0, collisions_count=0)
        
        scenario = ScenarioResults(scenario_id=1, ground_truths={0: gt}, experiments=[exp])
        
        assert scenario.classify_run(exp) == ExperimentResult.CORRECTIVE
    
    def test_returns_unchanged_when_penalties_equal(self):
        """Equal penalties = unchanged."""
        gt = RunResult(scenario_id=1, run_id=0, score_penalty=0.8, score_route=100.0, score_composed=80.0, status="Completed", infractions_count=1, collisions_count=0)
        exp = RunResult(scenario_id=1, run_id=1, ground_truth_index=0, score_penalty=0.8, score_route=100.0, score_composed=80.0, status="Completed", infractions_count=1, collisions_count=0)
        
        scenario = ScenarioResults(scenario_id=1, ground_truths={0: gt}, experiments=[exp])
        
        assert scenario.classify_run(exp) == ExperimentResult.UNCHANGED
    
    def test_returns_none_when_no_ground_truth(self):
        """Cannot classify without ground truth."""
        exp = RunResult(scenario_id=1, run_id=1, ground_truth_index=0, score_penalty=0.5, score_route=100.0, score_composed=50.0, status="Failed", infractions_count=1, collisions_count=1)
        
        scenario = ScenarioResults(scenario_id=1, ground_truths={}, experiments=[exp])
        
        assert scenario.classify_run(exp) == ExperimentResult.NONE
    
    def test_returns_none_when_experiment_is_none(self):
        """Cannot classify None experiment."""
        gt = RunResult(scenario_id=1, run_id=0, score_penalty=1.0, score_route=100.0, score_composed=100.0, status="Perfect", infractions_count=0, collisions_count=0)
        
        scenario = ScenarioResults(scenario_id=1, ground_truths={0: gt}, experiments=[])
        
        assert scenario.classify_run(None) == ExperimentResult.NONE


# =============================================================================
# JobResults._load_run_result Tests
# =============================================================================

class TestJobResultsLoadRunResult:
    """Tests for loading individual run results from disk."""
    
    def test_loads_valid_res_json(self, tmp_path: Path):
        """Successfully loads a valid res.json file."""
        run_dir = tmp_path / "sce_001" / "run_000"
        run_dir.mkdir(parents=True)
        
        res_data = make_res_json(
            status="Perfect",
            score_route=100.0,
            score_penalty=1.0,
            score_composed=100.0,
        )
        (run_dir / "res.json").write_text(json.dumps(res_data))
        
        # Create minimal job structure
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        results = JobResults.__new__(JobResults)
        results._job_dir = job_dir
        results._ground_truth_cache_dir = Path("")
        results._ground_truth_dir = job_dir / "ground_truths"
        results.scenarios = {}
        
        result = results._load_run_result(1, 0, run_dir)
        
        assert result is not None
        assert result.scenario_id == 1
        assert result.run_id == 0
        assert result.score_penalty == 1.0
        assert result.score_route == 100.0
        assert result.score_composed == 100.0
        assert result.status == "Perfect"
        assert result.infractions_count == 0
        assert result.collisions_count == 0
    
    def test_returns_none_when_res_json_missing(self, tmp_path: Path):
        """Returns None when res.json doesn't exist."""
        run_dir = tmp_path / "sce_001" / "run_000"
        run_dir.mkdir(parents=True)
        # No res.json created
        
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        results = JobResults.__new__(JobResults)
        results._job_dir = job_dir
        results._ground_truth_cache_dir = Path("")
        results._ground_truth_dir = job_dir / "ground_truths"
        results.scenarios = {}
        
        result = results._load_run_result(1, 0, run_dir)
        
        assert result is None
    
    def test_returns_none_when_records_empty(self, tmp_path: Path):
        """Returns None when checkpoint has no records."""
        run_dir = tmp_path / "sce_001" / "run_000"
        run_dir.mkdir(parents=True)
        
        res_data = {"_checkpoint": {"records": []}}
        (run_dir / "res.json").write_text(json.dumps(res_data))
        
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        results = JobResults.__new__(JobResults)
        results._job_dir = job_dir
        results._ground_truth_cache_dir = Path("")
        results._ground_truth_dir = job_dir / "ground_truths"
        results.scenarios = {}
        
        result = results._load_run_result(1, 0, run_dir)
        
        assert result is None
    
    def test_returns_none_when_json_malformed(self, tmp_path: Path):
        """Returns None when res.json contains invalid JSON."""
        run_dir = tmp_path / "sce_001" / "run_000"
        run_dir.mkdir(parents=True)
        
        (run_dir / "res.json").write_text("{ invalid json }")
        
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        results = JobResults.__new__(JobResults)
        results._job_dir = job_dir
        results._ground_truth_cache_dir = Path("")
        results._ground_truth_dir = job_dir / "ground_truths"
        results.scenarios = {}
        
        result = results._load_run_result(1, 0, run_dir)
        
        assert result is None
    
    def test_counts_infractions_excluding_min_speed(self, tmp_path: Path):
        """Infraction count excludes min_speed_infractions."""
        run_dir = tmp_path / "sce_001" / "run_000"
        run_dir.mkdir(parents=True)
        
        res_data = make_res_json(
            status="Failed",
            score_penalty=0.5,
            infractions={
                "collisions_vehicle": ["collision 1"],
                "red_light": ["red light 1", "red light 2"],
                "min_speed_infractions": ["too slow 1", "too slow 2", "too slow 3"],
            }
        )
        (run_dir / "res.json").write_text(json.dumps(res_data))
        
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        results = JobResults.__new__(JobResults)
        results._job_dir = job_dir
        results._ground_truth_cache_dir = Path("")
        results._ground_truth_dir = job_dir / "ground_truths"
        results.scenarios = {}
        
        result = results._load_run_result(1, 0, run_dir)
        
        assert result is not None
        # 1 collision + 2 red lights = 3 (min_speed excluded)
        assert result.infractions_count == 3
    
    def test_counts_all_collision_types(self, tmp_path: Path):
        """Collision count includes vehicle, pedestrian, and layout collisions."""
        run_dir = tmp_path / "sce_001" / "run_000"
        run_dir.mkdir(parents=True)
        
        res_data = make_res_json(
            status="Failed",
            score_penalty=0.25,
            infractions={
                "collisions_vehicle": ["vehicle 1", "vehicle 2"],
                "collisions_pedestrian": ["pedestrian 1"],
                "collisions_layout": ["wall 1", "wall 2", "wall 3"],
            }
        )
        (run_dir / "res.json").write_text(json.dumps(res_data))
        
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        results = JobResults.__new__(JobResults)
        results._job_dir = job_dir
        results._ground_truth_cache_dir = Path("")
        results._ground_truth_dir = job_dir / "ground_truths"
        results.scenarios = {}
        
        result = results._load_run_result(1, 0, run_dir)
        
        assert result is not None
        # 2 vehicle + 1 pedestrian + 3 layout = 6
        assert result.collisions_count == 6
    
    def test_loads_prompt_text_from_custom_prompts(self, tmp_path: Path):
        """Loads prompt text from custom_prompts.json."""
        run_dir = tmp_path / "sce_001" / "run_000"
        run_dir.mkdir(parents=True)
        
        (run_dir / "res.json").write_text(json.dumps(make_res_json()))
        (run_dir / "custom_prompts.json").write_text(
            json.dumps(make_custom_prompts(["Slow down at intersection"]))
        )
        
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        results = JobResults.__new__(JobResults)
        results._job_dir = job_dir
        results._ground_truth_cache_dir = Path("")
        results._ground_truth_dir = job_dir / "ground_truths"
        results.scenarios = {}
        
        result = results._load_run_result(1, 0, run_dir, ground_truth_index=0)
        
        assert result is not None
        assert result.prompt_text == "Slow down at intersection"
    
    def test_prompt_text_none_when_custom_prompts_missing(self, tmp_path: Path):
        """prompt_text is None when custom_prompts.json doesn't exist."""
        run_dir = tmp_path / "sce_001" / "run_000"
        run_dir.mkdir(parents=True)
        
        (run_dir / "res.json").write_text(json.dumps(make_res_json()))
        # No custom_prompts.json
        
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        results = JobResults.__new__(JobResults)
        results._job_dir = job_dir
        results._ground_truth_cache_dir = Path("")
        results._ground_truth_dir = job_dir / "ground_truths"
        results.scenarios = {}
        
        result = results._load_run_result(1, 0, run_dir)
        
        assert result is not None
        assert result.prompt_text is None


# =============================================================================
# JobResults._load_results Integration Tests
# =============================================================================

class TestJobResultsLoadResults:
    """Tests for loading all results from a job directory."""
    
    def test_loads_experiments_from_job_dir(self, tmp_path: Path):
        """Loads experiment runs from job directory."""
        job_dir = tmp_path / "job_001"
        run_dir = job_dir / "sce_005" / "run_000"
        run_dir.mkdir(parents=True)
        
        (run_dir / "res.json").write_text(json.dumps(make_res_json()))
        
        # Ground truth in cache
        cache_dir = tmp_path / "cache"
        gt_dir = cache_dir / "sce_005" / "run_000"
        gt_dir.mkdir(parents=True)
        (gt_dir / "res.json").write_text(json.dumps(make_res_json()))
        
        results = JobResults(job_dir, ground_truth_cache_dir=cache_dir)
        
        assert 5 in results.scenarios
        assert len(results.scenarios[5].experiments) == 1
        assert results.scenarios[5].experiments[0].scenario_id == 5
        assert results.scenarios[5].experiments[0].run_id == 0
    
    def test_loads_ground_truth_from_job_dir_ground_truths_first(self, tmp_path: Path):
        """Prefers ground truths from job_dir/ground_truths/ over cache."""
        job_dir = tmp_path / "job_001"
        
        # Experiment
        exp_dir = job_dir / "sce_010" / "run_000"
        exp_dir.mkdir(parents=True)
        (exp_dir / "res.json").write_text(json.dumps(make_res_json(score_penalty=0.5)))
        
        # Ground truth in job_dir/ground_truths (should be used)
        gt_dir = job_dir / "ground_truths" / "sce_010" / "run_000"
        gt_dir.mkdir(parents=True)
        (gt_dir / "res.json").write_text(json.dumps(make_res_json(score_penalty=0.9)))
        
        # Ground truth in cache (should NOT be used)
        cache_dir = tmp_path / "cache"
        cache_gt_dir = cache_dir / "sce_010" / "run_000"
        cache_gt_dir.mkdir(parents=True)
        (cache_gt_dir / "res.json").write_text(json.dumps(make_res_json(score_penalty=0.1)))
        
        results = JobResults(job_dir, ground_truth_cache_dir=cache_dir)
        
        assert 10 in results.scenarios
        # Should use ground truth from job_dir (penalty=0.9), not cache (penalty=0.1)
        assert results.scenarios[10].ground_truths[0].score_penalty == 0.9
    
    def test_falls_back_to_cache_when_no_job_ground_truth(self, tmp_path: Path):
        """Falls back to cache directory when job_dir/ground_truths missing."""
        job_dir = tmp_path / "job_001"
        
        # Experiment
        exp_dir = job_dir / "sce_015" / "run_000"
        exp_dir.mkdir(parents=True)
        (exp_dir / "res.json").write_text(json.dumps(make_res_json()))
        
        # No ground_truths dir in job_dir
        
        # Ground truth in cache
        cache_dir = tmp_path / "cache"
        gt_dir = cache_dir / "sce_015" / "run_000"
        gt_dir.mkdir(parents=True)
        (gt_dir / "res.json").write_text(json.dumps(make_res_json(score_penalty=0.75)))
        
        results = JobResults(job_dir, ground_truth_cache_dir=cache_dir)
        
        assert 15 in results.scenarios
        assert results.scenarios[15].ground_truths[0].score_penalty == 0.75
    
    def test_uses_experiment_info_for_ground_truth_index(self, tmp_path: Path):
        """Reads ground_truth_index from experiment_info.json."""
        job_dir = tmp_path / "job_001"
        
        # Experiment pointing to ground truth index 2
        exp_dir = job_dir / "sce_020" / "run_000"
        exp_dir.mkdir(parents=True)
        (exp_dir / "res.json").write_text(json.dumps(make_res_json()))
        (exp_dir / "experiment_info.json").write_text(
            json.dumps(make_experiment_info(ground_truth_index=2))
        )
        
        # Ground truth at index 2 in cache
        cache_dir = tmp_path / "cache"
        gt_dir = cache_dir / "sce_020" / "run_002"
        gt_dir.mkdir(parents=True)
        (gt_dir / "res.json").write_text(json.dumps(make_res_json(score_penalty=0.6)))
        
        results = JobResults(job_dir, ground_truth_cache_dir=cache_dir)
        
        assert 20 in results.scenarios
        # Should have loaded ground truth at index 2
        assert 2 in results.scenarios[20].ground_truths
        assert results.scenarios[20].ground_truths[2].score_penalty == 0.6
        # Experiment should reference ground truth index 2
        assert results.scenarios[20].experiments[0].ground_truth_index == 2
    
    def test_defaults_ground_truth_index_to_zero_when_no_experiment_info(self, tmp_path: Path):
        """Defaults to ground_truth_index=0 when experiment_info.json missing."""
        job_dir = tmp_path / "job_001"
        
        # Experiment without experiment_info.json
        exp_dir = job_dir / "sce_025" / "run_000"
        exp_dir.mkdir(parents=True)
        (exp_dir / "res.json").write_text(json.dumps(make_res_json()))
        # No experiment_info.json
        
        # Ground truth at index 0 in cache
        cache_dir = tmp_path / "cache"
        gt_dir = cache_dir / "sce_025" / "run_000"
        gt_dir.mkdir(parents=True)
        (gt_dir / "res.json").write_text(json.dumps(make_res_json()))
        
        results = JobResults(job_dir, ground_truth_cache_dir=cache_dir)
        
        assert 25 in results.scenarios
        assert 0 in results.scenarios[25].ground_truths
    
    def test_loads_multiple_scenarios(self, tmp_path: Path):
        """Loads experiments from multiple scenarios."""
        job_dir = tmp_path / "job_001"
        cache_dir = tmp_path / "cache"
        
        for sce_id in [5, 10, 15]:
            # Experiment
            exp_dir = job_dir / f"sce_{sce_id:03d}" / "run_000"
            exp_dir.mkdir(parents=True)
            (exp_dir / "res.json").write_text(json.dumps(make_res_json()))
            
            # Ground truth
            gt_dir = cache_dir / f"sce_{sce_id:03d}" / "run_000"
            gt_dir.mkdir(parents=True)
            (gt_dir / "res.json").write_text(json.dumps(make_res_json()))
        
        results = JobResults(job_dir, ground_truth_cache_dir=cache_dir)
        
        assert len(results.scenarios) == 3
        assert set(results.scenarios.keys()) == {5, 10, 15}
    
    def test_raises_error_when_job_dir_not_found(self, tmp_path: Path):
        """Raises FileNotFoundError when job directory doesn't exist."""
        nonexistent = tmp_path / "nonexistent_job"
        
        with pytest.raises(FileNotFoundError):
            JobResults(nonexistent)


# =============================================================================
# JobResults Public API Tests
# =============================================================================

class TestJobResultsAPI:
    """Tests for JobResults public methods."""
    
    def test_get_scenario_returns_scenario_by_id(self, job_dir: Path, cache_dir_with_ground_truth: Path):
        """get_scenario returns ScenarioResults for the given ID."""
        results = JobResults(job_dir, ground_truth_cache_dir=cache_dir_with_ground_truth)
        
        scenario = results.get_scenario(5)
        
        assert scenario is not None
        assert scenario.scenario_id == 5
    
    def test_get_scenario_returns_none_for_unknown_id(self, job_dir: Path, cache_dir_with_ground_truth: Path):
        """get_scenario returns None for non-existent scenario ID."""
        results = JobResults(job_dir, ground_truth_cache_dir=cache_dir_with_ground_truth)
        
        scenario = results.get_scenario(999)
        
        assert scenario is None
    
    def test_get_all_scenarios_returns_sorted_list(self, tmp_path: Path):
        """get_all_scenarios returns scenarios sorted by ID."""
        job_dir = tmp_path / "job_001"
        cache_dir = tmp_path / "cache"
        
        # Create scenarios in non-sorted order
        for sce_id in [20, 5, 15, 10]:
            exp_dir = job_dir / f"sce_{sce_id:03d}" / "run_000"
            exp_dir.mkdir(parents=True)
            (exp_dir / "res.json").write_text(json.dumps(make_res_json()))
            
            gt_dir = cache_dir / f"sce_{sce_id:03d}" / "run_000"
            gt_dir.mkdir(parents=True)
            (gt_dir / "res.json").write_text(json.dumps(make_res_json()))
        
        results = JobResults(job_dir, ground_truth_cache_dir=cache_dir)
        
        all_scenarios = results.get_all_scenarios()
        
        assert len(all_scenarios) == 4
        assert [s.scenario_id for s in all_scenarios] == [5, 10, 15, 20]
