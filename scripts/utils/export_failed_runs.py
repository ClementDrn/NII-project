#!/usr/bin/env python3

"""Export error-tagged runs from one job directory to a single failed-runs JSON file.

This script is intended to be used per job, for example:
    python scripts/export_failed_runs.py /home/cdarne/digit/output/simlingo/bench2drive/1/08-30-claude-oneshot-remaining/job_000 \
        -o claude-oneshot-failedruns-n1.json
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a single job directory and the output JSON path."""
    parser = argparse.ArgumentParser(
        description="Export all runs marked [ERROR] from one job into a failedruns JSON file.",
    )
    parser.add_argument(
        "job_dir",
        type=Path,
        help="A single job directory such as /path/to/job_000.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=Path,
        required=True,
        help="Output JSON file path for this job's failed runs.",
    )
    return parser.parse_args()


def extract_error_records_from_print_output(output: str):
    """Parse the summary output from scripts/print_results.py and recover [ERROR] runs.

    The summary looks roughly like:
        Scenario 040
        run_002: ... [ERROR]
        prompt: "..." t_start=50.00s t_end=53.50s
    """
    records = []
    current_scenario = None
    pending_error = False

    for line in output.splitlines():
        # Each block starts with Scenario NNN.
        scenario_match = re.match(r"^Scenario\s+(\d+)", line.strip())
        if scenario_match:
            current_scenario = int(scenario_match.group(1))
            continue

        # Once an [ERROR] tag appears, the next prompt line contains the instruction and
        # injection timing for that failed run.
        if "[ERROR]" in line:
            pending_error = True
            continue

        if not pending_error:
            continue

        prompt_match = re.search(
            r'prompt:\s*"(.+?)"\s*(?:t_start=(\d+(?:\.\d+)?)s\s+t_end=(\d+(?:\.\d+)?)s)?',
            line,
        )
        if not prompt_match:
            continue

        instruction = prompt_match.group(1)
        t_start = float(prompt_match.group(2)) if prompt_match.group(2) else None
        t_end = float(prompt_match.group(3)) if prompt_match.group(3) else None

        if current_scenario is None or t_start is None or t_end is None:
            pending_error = False
            continue

        records.append(
            {
                "scenario_id": current_scenario,
                "injection_window": {
                    "start_timestamp": t_start,
                    "end_timestamp": t_end,
                },
                "instruction": instruction,
            }
        )
        pending_error = False

    return records


def extract_error_records_from_job(job_dir: Path):
    """Collect failing runs for one job.

    We prefer the summary output from scripts/print_results.py because it already contains the
    [ERROR] tagging and the prompt timing for each failed experiment. If that parsing fails to
    recover anything useful, we fall back to reading res.json + custom_prompts.json directly.
    """
    cmd = [sys.executable, "scripts/print_results.py", str(job_dir)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    records = extract_error_records_from_print_output(output)

    if records:
        return records

    # Fallback for jobs whose print output is missing or formatted differently.
    fallback = []
    for sce_dir in sorted(job_dir.glob("sce_*")):
        if not sce_dir.is_dir():
            continue
        try:
            scenario_id = int(sce_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        for run_dir in sorted(sce_dir.glob("run_*")):
            res_file = run_dir / "res.json"
            prompts_file = run_dir / "custom_prompts.json"
            if not res_file.exists() or not prompts_file.exists():
                continue

            try:
                res_data = json.loads(res_file.read_text())
                prompts_data = json.loads(prompts_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            record = res_data.get("_checkpoint", {}).get("records", [{}])[0]
            status = str(record.get("status", "")).lower()
            if "failed" not in status:
                continue
            if not prompts_data:
                continue

            item = prompts_data[0]
            instruction = item.get("text", "")
            t_start = item.get("injection_time")
            t_end = item.get("injection_duration")
            if t_start is None or t_end is None:
                continue

            fallback.append(
                {
                    "scenario_id": scenario_id,
                    "injection_window": {
                        "start_timestamp": float(t_start),
                        "end_timestamp": float(t_start) + float(t_end),
                    },
                    "instruction": instruction,
                }
            )

    return fallback


def write_failed_runs(job_dir: Path, output_file: Path):
    """Export one job's failed runs to a single JSON file in the baseline format."""
    records = extract_error_records_from_job(job_dir)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"{job_dir.name}: wrote {len(records)} error runs to {output_file}")


def main():
    """Export error-tagged runs for a single job directory."""
    args = parse_args()
    job_dir = args.job_dir.resolve()
    output_file = args.output_file.resolve()

    if not job_dir.exists() or not job_dir.is_dir():
        raise FileNotFoundError(f"Job directory not found: {job_dir}")

    write_failed_runs(job_dir, output_file)


if __name__ == "__main__":
    main()
