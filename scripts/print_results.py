from untitled.data.job_results import ExperimentResult, JobResults

import argparse
import sys
from pathlib import Path


# ANSI color codes
class Colors:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'


def print_results(job_dir: Path, ground_truth_cache_dir: Path) -> None:
    """Print detailed results for a job."""
    
    print("=" * 80)
    print(f"JOB RESULTS: {job_dir}")
    print(f"GROUND TRUTH: {job_dir}/ground_truths (fallback cache: {ground_truth_cache_dir})")
    print("=" * 80)
    
    # Load groundtruth and experiment results
    try:
        results = JobResults(job_dir, ground_truth_cache_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    if not results.scenarios:
        print("No scenarios found.")
        return
    
    # Print per-scenario results
    for scenario in results.get_all_scenarios():
        print()
        print(f"Scenario {scenario.scenario_id:03d}")
        print("-" * 60)
        
        # Ground truths
        for gt_index, gt in scenario.ground_truths.items():
            if gt:
                print(f"  [GT run_{gt_index:03d}] penalty={gt.score_penalty:.3f}, "
                      f"route={gt.score_route:.0f}%, status={gt.status}")
            else:
                print(f"  [GT run_{gt_index:03d}] No result in cache")
        
        # Experiments (all runs in job_dir)
        for exp in sorted(scenario.experiments, key=lambda x: x.run_id):
            classification = scenario.classify_run(exp)
            marker = ""
            if classification == ExperimentResult.ADVERSARIAL:
                marker = f" {Colors.RED}[ADVERSARIAL]{Colors.RESET}"
            elif classification == ExperimentResult.CORRECTIVE:
                marker = f" {Colors.GREEN}[CORRECTIVE]{Colors.RESET}"
            elif classification == ExperimentResult.UNCHANGED:
                marker = f" {Colors.YELLOW}[UNCHANGED]{Colors.RESET}"
            
            print(f"  run_{exp.run_id:03d}: penalty={exp.score_penalty:.3f}, "
                  f"route={exp.score_route:.0f}%, infractions={exp.infractions_count}, "
                  f"collisions={exp.collisions_count}{marker}")
            if exp.prompt_text:
                print(f"       prompt: \"{exp.prompt_text[:60]}{'...' if len(exp.prompt_text) > 60 else ''}\"")
    
    # Summary
    summary = results.get_summary()
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total scenarios: {len(results.scenarios)}")
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Adversarial (worse than GT): {summary['adversarial']}")
    print(f"Corrective (better than GT): {summary['corrective']}")
    print(f"Unchanged: {summary['unchanged']}")
    
    if summary['total_experiments'] > 0:
        adv_rate = (summary['adversarial'] / summary['total_experiments']) * 100
        corr_rate = (summary['corrective'] / summary['total_experiments']) * 100
        print(f"Adversarial rate: {adv_rate:.1f}%")
        print(f"Corrective rate: {corr_rate:.1f}%")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Print job results summary")
    parser.add_argument("job_dir", type=str, help="Path to job directory (e.g., data/saved/02-10)")
    parser.add_argument("--gt-dir", type=str, default="cache", help="Path to ground truth cache directory (default: cache)")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    job_dir = Path(args.job_dir)
    if not job_dir.is_absolute():
        job_dir = project_root / args.job_dir
    
    gt_dir = Path(args.gt_dir)
    if not gt_dir.is_absolute():
        gt_dir = project_root / args.gt_dir
    
    print_results(job_dir, gt_dir)


if __name__ == "__main__":
    main()

