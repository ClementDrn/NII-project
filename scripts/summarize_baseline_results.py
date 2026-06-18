from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from untitled.data.job_results import ExperimentResult, JobResults
from untitled.generator.instruction import InstructionCategory


CATEGORY_COLUMNS = [
    "INCREASE_RELATIVE_SPEED",
    "DECREASE_RELATIVE_SPEED",
    "INCREASE_ABSOLUTE_SPEED",
    "DECREASE_ABSOLUTE_SPEED",
    "INCREASE_RELATIVE_DISTANCE",
    "DECREASE_RELATIVE_DISTANCE",
    "INCREASE_ABSOLUTE_DISTANCE",
    "DECREASE_ABSOLUTE_DISTANCE",
    "OVERTAKE_LEFT",
    "OVERTAKE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "GO_STRAIGHT",
    "LANE_CHANGE_LEFT",
    "LANE_CHANGE_RIGHT",
    "PULL_OVER",
]


def _parse_category_from_file_stem(stem: str) -> Optional[Tuple[InstructionCategory, int]]:
    # Without variables
    direct_mappings: Dict[str, Tuple[InstructionCategory, int]] = {
        "left_lc": (InstructionCategory.LANE_CHANGE_LEFT, 0),
        "right_lc": (InstructionCategory.LANE_CHANGE_RIGHT, 0),
        "left_overtake": (InstructionCategory.OVERTAKE_LEFT, 0),
        "right_overtake": (InstructionCategory.OVERTAKE_RIGHT, 0),
        "turn_left": (InstructionCategory.TURN_LEFT, 0),
        "turn_right": (InstructionCategory.TURN_RIGHT, 0),
        "go_straight": (InstructionCategory.GO_STRAIGHT, 0),
        "pull_over": (InstructionCategory.PULL_OVER, 0),
    }
    if stem in direct_mappings:
        return direct_mappings[stem]

    # With variables (e.g., inc_abs_speed30)
    pattern_mappings: Iterable[Tuple[str, InstructionCategory]] = [
        (r"^inc_abs_speed(\d+)$", InstructionCategory.INCREASE_ABSOLUTE_SPEED),
        (r"^dec_abs_speed(\d+)$", InstructionCategory.DECREASE_ABSOLUTE_SPEED),
        (r"^inc_rel_speed(\d+)$", InstructionCategory.INCREASE_RELATIVE_SPEED),
        (r"^dec_rel_speed(\d+)$", InstructionCategory.DECREASE_RELATIVE_SPEED),
        (r"^inc_abs_dis(\d+)$", InstructionCategory.INCREASE_ABSOLUTE_DISTANCE),
        (r"^dec_abs_dis(\d+)$", InstructionCategory.DECREASE_ABSOLUTE_DISTANCE),
        (r"^inc_rel_dis(\d+)$", InstructionCategory.INCREASE_RELATIVE_DISTANCE),
        (r"^dec_rel_dis(\d+)$", InstructionCategory.DECREASE_RELATIVE_DISTANCE),
    ]

    import re

    for pattern, category in pattern_mappings:
        match = re.match(pattern, stem)
        if match:
            return (category, int(match.group(1)))

    return None


def _load_commands(json_path: Path) -> Iterable[str]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    commands = data.get("commands", [])
    if not isinstance(commands, list):
        return []
    return [cmd.strip() for cmd in commands if isinstance(cmd, str) and cmd.strip()]


def build_instruction_text_map(dataset_dir: Path) -> Tuple[Dict[str, InstructionCategory], int]:
    text_to_category: Dict[str, InstructionCategory] = {}
    conflicts = 0

    for json_path in sorted(dataset_dir.glob("*.json")):
        parsed = _parse_category_from_file_stem(json_path.stem)
        if parsed is None:
            continue
        category, _variable = parsed

        for command in _load_commands(json_path):
            existing = text_to_category.get(command)
            if existing is None:
                text_to_category[command] = category
            elif existing != category:
                conflicts += 1

    return text_to_category, conflicts


def summarize_baseline(job_dir: Path, ground_truth_cache_dir: Path, dataset_dir: Path, out_path: Path) -> None:
    text_to_category, conflicts = build_instruction_text_map(dataset_dir)
    if conflicts > 0:
        print(f"Warning: {conflicts} command texts map to multiple categories.", file=sys.stderr)

    try:
        results = JobResults(job_dir, ground_truth_cache_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return

    header = ["Scenario", "#Runs", "#Adversarial"] + CATEGORY_COLUMNS

    total_runs = 0
    total_adversarial = 0
    total_by_category = {name: 0 for name in CATEGORY_COLUMNS}
    unknown_category_count = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for scenario in results.get_all_scenarios():
            row_counts = {name: 0 for name in CATEGORY_COLUMNS}
            adversarial_count = 0
            run_count = len(scenario.experiments)

            for exp in scenario.experiments:
                classification = scenario.classify_run(exp)
                if classification != ExperimentResult.ADVERSARIAL:
                    continue

                adversarial_count += 1
                if exp.prompt_text:
                    category = text_to_category.get(exp.prompt_text.strip())
                    if category is not None:
                        category_name = category.name
                        if category_name in row_counts:
                            row_counts[category_name] += 1
                    else:
                        unknown_category_count += 1
                else:
                    unknown_category_count += 1

            row = [
                str(scenario.scenario_id),
                str(run_count),
                str(adversarial_count),
            ] + [str(row_counts[name]) for name in CATEGORY_COLUMNS]
            writer.writerow(row)

            total_runs += run_count
            total_adversarial += adversarial_count
            for name in CATEGORY_COLUMNS:
                total_by_category[name] += row_counts[name]

        total_row = ["TOTAL", str(total_runs), str(total_adversarial)] + [
            str(total_by_category[name]) for name in CATEGORY_COLUMNS
        ]
        writer.writerow(total_row)

    if unknown_category_count > 0:
        print(
            f"Warning: {unknown_category_count} adversarial runs have no matching category text.",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize baseline adversarialness per scenario into a CSV file"
    )
    parser.add_argument("job_dir", type=str, help="Path to job directory (e.g., output/.../job_000)")
    parser.add_argument(
        "--gt-dir",
        type=str,
        default="cache",
        help="Path to ground truth cache directory (default: cache)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="vendor/data/LaMPilot/DbLv1",
        help="Path to instruction dataset directory (default: vendor/data/LaMPilot/DbLv1)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="baseline_summary.csv",
        help="Output CSV path (default: baseline_summary.csv)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    job_dir = Path(args.job_dir)
    if not job_dir.is_absolute():
        job_dir = project_root / job_dir

    gt_dir = Path(args.gt_dir)
    if not gt_dir.is_absolute():
        gt_dir = project_root / gt_dir

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = project_root / dataset_dir

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = project_root / out_path

    summarize_baseline(job_dir, gt_dir, dataset_dir, out_path)


if __name__ == "__main__":
    main()
