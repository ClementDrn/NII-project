#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import json
import re
from typing import Any, Dict, Iterable, List, Optional


SCENARIO_STAGE_PATTERN = re.compile(r"Scenario\s+(\d+)\s+\|\s+VLM Instruction Generation Stage")
SCENARIO_GENERATING_PATTERN = re.compile(r"Generating instructions for scenario\s+(\d+)\.")
INSTRUCTION_PATTERN = re.compile(
	r"- Instruction\s+\d+:\s+Window=\[(?P<start>[0-9]+(?:\.[0-9]+)?)s,\s+(?P<end>[0-9]+(?:\.[0-9]+)?)s\],\s+Text='(?P<text>.*)'$"
)


def _parse_scenario_id(line: str) -> Optional[int]:
	match = SCENARIO_STAGE_PATTERN.search(line)
	if match:
		return int(match.group(1))

	match = SCENARIO_GENERATING_PATTERN.search(line)
	if match:
		return int(match.group(1))

	return None


def extract_instructions_from_log(log_path: Path) -> List[Dict[str, Any]]:
	"""Extract flattened instruction records from a pipeline log."""
	records: List[Dict[str, Any]] = []
	current_scenario_id: Optional[int] = None

	with open(log_path, "r", encoding="utf-8") as handle:
		for line in handle:
			scenario_id = _parse_scenario_id(line)
			if scenario_id is not None:
				current_scenario_id = scenario_id
				continue

			match = INSTRUCTION_PATTERN.search(line.rstrip("\n"))
			if match is None or current_scenario_id is None:
				continue

			records.append(
				{
					"scenario_id": current_scenario_id,
					"injection_window": {
						"start_timestamp": float(match.group("start")),
						"end_timestamp": float(match.group("end")),
					},
					"instruction": match.group("text"),
				}
			)

	return records


def write_instructions_json(records: Iterable[Dict[str, Any]], out_path: Path) -> None:
	"""Write extracted instruction records as JSON."""
	with open(out_path, "w", encoding="utf-8") as handle:
		json.dump(list(records), handle, indent=2, ensure_ascii=False)
		handle.write("\n")



def main() -> int:
	parser = argparse.ArgumentParser(description="Extract simulation instructions from a log file.")
	parser.add_argument("log_path", type=Path, help="Path to the pipeline log file")
	parser.add_argument(
		"--out",
		type=Path,
		default=Path("instructions.json"),
		help="Output JSON file path (default: instructions.json)",
	)
	args = parser.parse_args()

	project_root = Path(__file__).parent.parent
	log_path = args.log_path
	if not log_path.is_absolute():
		log_path = project_root / log_path

	out_path = args.out
	if not out_path.is_absolute():
		out_path = project_root / out_path

	records = extract_instructions_from_log(log_path)
	write_instructions_json(records, out_path)
	print(f"Wrote {len(records)} instruction(s) to {out_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())