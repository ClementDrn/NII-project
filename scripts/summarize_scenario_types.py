from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET


def summarize_scenario_types(directory: Path) -> dict[str, list[int]]:
    """Collect scenario types from all XML files in a directory."""
    summary: dict[str, list[int]] = defaultdict(list)

    for file_path in sorted(directory.glob("*.xml")):
        try:
            root = ET.parse(file_path).getroot()
        except ET.ParseError as exc:
            print(f"Warning: could not parse {file_path.name}: {exc}")
            continue

        # Capture all scenario 'type' attributes in this file.
        for scenario in root.findall(".//scenario"):
            scenario_type = scenario.get("type")
            if scenario_type:
                # ID is in filename: bench2drive_10.xml -> 10
                scenario_id: int = int(file_path.stem.split("_")[-1])
                summary[scenario_type].append(scenario_id)

    return summary


def print_summary(summary: dict[str, list[int]]) -> None:
    print("All types:")
    if not summary:
        print("- (no scenario types found)")
        return

    for scenario_type in sorted(summary):
        files = sorted(set(summary[scenario_type]))
        count = len(files)
        # - Category (count): IDA, IDB, IDC, etc.
        print(f"- {scenario_type} ({count}): {', '.join(str(f) for f in files)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read XML files in a directory and summarize scenario types."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory containing XML files (default: current directory)",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists() or not directory.is_dir():
        raise SystemExit(f"Error: '{directory}' is not a valid directory")

    summary = summarize_scenario_types(directory)
    print_summary(summary)


if __name__ == "__main__":
    main()
