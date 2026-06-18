#!/usr/bin/env python3
"""
Check a rule against all metadata frames in a scenario or run directory.

Usage:
  python scripts/check_rules_on_scenario.py <scenario_dir> <rule>

Examples:
  python scripts/check_rules_on_scenario.py cache/sce_005 inc_abs_speed30
  python scripts/check_rules_on_scenario.py cache/sce_005/run_000 TURN_LEFT
  python scripts/check_rules_on_scenario.py cache/sce_005 TURN_RIGHT --variable 0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

from untitled.generator.instruction import InstructionCategory
from untitled.generator.rules import get_rule_for_category
from untitled.simulator.simulation_data import SimulationFrameData, load_metadata


def _parse_category_from_spec(spec: str) -> Optional[Tuple[InstructionCategory, Optional[float]]]:
	direct_mappings = {
		"left_lc": (InstructionCategory.LANE_CHANGE_LEFT, None),
		"right_lc": (InstructionCategory.LANE_CHANGE_RIGHT, None),
		"left_overtake": (InstructionCategory.OVERTAKE_LEFT, None),
		"right_overtake": (InstructionCategory.OVERTAKE_RIGHT, None),
		"turn_left": (InstructionCategory.TURN_LEFT, None),
		"turn_right": (InstructionCategory.TURN_RIGHT, None),
		"go_straight": (InstructionCategory.GO_STRAIGHT, None),
		"pull_over": (InstructionCategory.PULL_OVER, None),
	}

	normalized = spec.strip().lower()
	if normalized in direct_mappings:
		return direct_mappings[normalized]

	name_candidate = normalized.replace("-", "_").upper()
	if name_candidate in InstructionCategory.__members__:
		return InstructionCategory[name_candidate], None

	pattern_mappings: Iterable[Tuple[str, InstructionCategory]] = [
		(r"^inc_abs_speed(\d+(?:\.\d+)?)$", InstructionCategory.INCREASE_ABSOLUTE_SPEED),
		(r"^dec_abs_speed(\d+(?:\.\d+)?)$", InstructionCategory.DECREASE_ABSOLUTE_SPEED),
		(r"^inc_rel_speed(\d+(?:\.\d+)?)$", InstructionCategory.INCREASE_RELATIVE_SPEED),
		(r"^dec_rel_speed(\d+(?:\.\d+)?)$", InstructionCategory.DECREASE_RELATIVE_SPEED),
		(r"^inc_abs_dis(\d+(?:\.\d+)?)$", InstructionCategory.INCREASE_ABSOLUTE_DISTANCE),
		(r"^dec_abs_dis(\d+(?:\.\d+)?)$", InstructionCategory.DECREASE_ABSOLUTE_DISTANCE),
		(r"^inc_rel_dis(\d+(?:\.\d+)?)$", InstructionCategory.INCREASE_RELATIVE_DISTANCE),
		(r"^dec_rel_dis(\d+(?:\.\d+)?)$", InstructionCategory.DECREASE_RELATIVE_DISTANCE),
	]

	for pattern, category in pattern_mappings:
		match = re.match(pattern, normalized)
		if match:
			return category, float(match.group(1))

	return None


def _parse_rule_spec(rule_spec: str, variable: Optional[float]) -> Tuple[InstructionCategory, float]:
	parsed = _parse_category_from_spec(rule_spec)
	if parsed is None:
		raise ValueError(f"Unsupported rule spec: '{rule_spec}'")

	category, parsed_variable = parsed
	if parsed_variable is not None and variable is not None:
		raise ValueError("Variable is already provided by rule spec; remove --variable")

	resolved_variable = variable if variable is not None else parsed_variable

	requires_variable = {
		InstructionCategory.INCREASE_ABSOLUTE_SPEED,
		InstructionCategory.DECREASE_ABSOLUTE_SPEED,
		InstructionCategory.INCREASE_ABSOLUTE_DISTANCE,
		InstructionCategory.DECREASE_ABSOLUTE_DISTANCE,
	}
	if category in requires_variable and (resolved_variable is None or resolved_variable <= 0):
		raise ValueError(f"Rule '{category.name}' requires --variable > 0")

	if resolved_variable is None:
		resolved_variable = 0.0

	return category, float(resolved_variable)


def _parse_run_id(run_dir: Path) -> Optional[int]:
	try:
		return int(run_dir.name.split("_")[1])
	except (IndexError, ValueError):
		return None


def _resolve_run_dirs(scenario_dir: Path) -> list[Path]:
	metadata_dir = scenario_dir / "metadata"
	if metadata_dir.exists() and metadata_dir.is_dir():
		return [scenario_dir]

	run_dirs = [path for path in scenario_dir.glob("run_*") if path.is_dir()]
	return sorted(run_dirs, key=lambda path: _parse_run_id(path) or -1)


def _sorted_metadata_files(metadata_dir: Path) -> list[Path]:
	def sort_key(path: Path) -> tuple[int, str]:
		stem = path.stem
		if stem.isdigit():
			return (0, f"{int(stem):09d}")
		return (1, stem)

	return sorted(metadata_dir.glob("*.json"), key=sort_key)


def _check_rule_on_run(run_dir: Path, category: InstructionCategory, variable: float) -> list[tuple[str, int, bool]]:
	metadata_dir = run_dir / "metadata"
	if not metadata_dir.exists():
		raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

	rule = get_rule_for_category(category)
	if rule is None:
		raise ValueError(f"No rule registered for category: {category.name}")

	results: list[tuple[str, int, bool]] = []
	for metadata_path in _sorted_metadata_files(metadata_dir):
		metadata = load_metadata(metadata_path)
		frame = SimulationFrameData(image_name=f"{metadata_path.stem}.png", metadata=metadata)
		is_valid = bool(rule.is_valid_for_this_frame(frame, variable))
		results.append((metadata_path.name, metadata.step, is_valid))

	return results


def main() -> int:
	parser = argparse.ArgumentParser(description="Check a rule against all metadata frames.")
	parser.add_argument(
		"scenario_dir",
		type=Path,
		help="Scenario directory (sce_XXX) or run directory (run_XXX)",
	)
	parser.add_argument("rule", type=str, help="Rule spec or InstructionCategory name")
	parser.add_argument(
		"--variable",
		type=float,
		default=None,
		help="Numeric variable for absolute speed/distance rules",
	)
	args = parser.parse_args()

	project_root = Path(__file__).parent.parent
	scenario_dir = args.scenario_dir
	if not scenario_dir.is_absolute():
		scenario_dir = project_root / scenario_dir

	if not scenario_dir.exists():
		print(f"Error: scenario_dir does not exist: {scenario_dir}", file=sys.stderr)
		return 1

	try:
		category, variable = _parse_rule_spec(args.rule, args.variable)
	except ValueError as exc:
		print(f"Error: {exc}", file=sys.stderr)
		return 1

	run_dirs = _resolve_run_dirs(scenario_dir)
	if not run_dirs:
		print(f"Error: No run directories found under {scenario_dir}", file=sys.stderr)
		return 1

	for run_dir in run_dirs:
		run_label = run_dir.name if run_dir != scenario_dir else run_dir.name
		print(f"# {run_label} ({category.name}, variable={variable})")

		try:
			results = _check_rule_on_run(run_dir, category, variable)
		except (FileNotFoundError, ValueError) as exc:
			print(f"Error: {exc}", file=sys.stderr)
			return 1

		if not results:
			print("(no metadata frames found)")
			continue

		for frame_name, step, is_valid in results:
			print(f"{frame_name}\tstep={step}\t{str(is_valid).lower()}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
