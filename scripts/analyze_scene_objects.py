"""Append scene object mention counts to an instruction CSV."""

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from untitled.data.scene_object import SceneObjectType, parse_scene_objects


# Keep the CSV columns in sync with the types supported by the scene parser.
OBJECT_FIELDNAMES = [object_type.value for object_type in SceneObjectType]


def count_scene_objects(instruction: str) -> dict[str, int]:
    """
    Counts mentions of each scene object type in an instruction.
    Repeated mentions count separately; absent types have a count of zero.
    """
    counts = {field_name: 0 for field_name in OBJECT_FIELDNAMES}
    # Empty instructions have no objects and do not need NLP processing.
    if not instruction.strip():
        return counts

    for scene_object in parse_scene_objects(instruction):
        counts[scene_object.type.value] += 1
    return counts


def analyze_scene_objects(input_path: Path, output_path: Path) -> int:
    """
    Copies the input CSV and appends one count column per scene object type.
    Preserves all input columns and rows. Returns the number of rows written.
    """
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output CSV paths must be different")

    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or "instruction" not in reader.fieldnames:
            raise ValueError("Input CSV must contain an 'instruction' column")

        # Preserve identifiers such as model, method, repetition, scenario, run,
        # and redacted (or redacted?), along with any other input columns.
        # Recalculate existing object columns without duplicating their headers.
        fieldnames = list(reader.fieldnames)
        fieldnames.extend(field for field in OBJECT_FIELDNAMES if field not in fieldnames)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        row_count = 0
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                instruction = row.get("instruction") or ""
                row.update(count_scene_objects(instruction))
                writer.writerow(row)
                row_count += 1

    return row_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="Input CSV containing an instruction column")
    parser.add_argument("output_csv", type=Path, help="Output CSV with scene object mention counts")
    args = parser.parse_args()

    row_count = analyze_scene_objects(args.input_csv, args.output_csv)
    print(f"Wrote {row_count} instructions with scene object counts to {args.output_csv}")


if __name__ == "__main__":
    main()
