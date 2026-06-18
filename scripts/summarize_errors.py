from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ERROR_PATTERN = re.compile(r"\[ERROR\]\s*(.*)")

# Normalize common volatile tokens to aggregate similar errors.
NORMALIZE_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"0x[0-9a-fA-F]+"), "<hex>"),
    (re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"), "<uuid>"),
    (re.compile(r"\b\d+\.\d+\b"), "<num>"),
    (re.compile(r"\b\d+\b"), "<num>"),
]


@dataclass
class ErrorGroup:
    normalized: str
    count: int
    example: str
    files: List[str]


def _iter_log_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(path.rglob("*.log"))
        else:
            yield path


def _normalize_message(message: str, enable: bool) -> str:
    if not enable:
        return message

    normalized = message
    for pattern, replacement in NORMALIZE_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def _extract_error_message(line: str) -> str | None:
    match = ERROR_PATTERN.search(line)
    if not match:
        return None
    return match.group(1).strip()


def summarize_errors(paths: Iterable[Path], normalize: bool) -> List[ErrorGroup]:
    counts: Counter[str] = Counter()
    examples: Dict[str, str] = {}
    files_by_key: Dict[str, List[str]] = defaultdict(list)

    for log_path in _iter_log_files(paths):
        if not log_path.exists():
            continue
        try:
            content = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for line in content.splitlines():
            message = _extract_error_message(line)
            if message is None:
                continue
            key = _normalize_message(message, normalize)
            counts[key] += 1
            examples.setdefault(key, message)
            if not files_by_key[key] or files_by_key[key][-1] != str(log_path):
                files_by_key[key].append(str(log_path))

    groups = []
    for key, count in counts.most_common():
        groups.append(
            ErrorGroup(
                normalized=key,
                count=count,
                example=examples.get(key, ""),
                files=files_by_key.get(key, []),
            )
        )
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize [ERROR] lines and aggregate similar messages"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=str,
        help="Log files or directories to scan (directories are searched for *.log)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable normalization of numbers/UUIDs/hex values",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
        help="Max number of example files to show per error (default: 5)",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths]
    groups = summarize_errors(paths, normalize=not args.no_normalize)

    if not groups:
        print("No [ERROR] lines found.")
        return

    for group in groups:
        print(f"{group.count:>5}  {group.normalized}")
        if group.example and group.example != group.normalized:
            print(f"       example: {group.example}")
        if group.files:
            sample = group.files[: args.max_files]
            print(f"       files: {', '.join(sample)}")
            if len(group.files) > args.max_files:
                print(f"       files: (+{len(group.files) - args.max_files} more)")


if __name__ == "__main__":
    main()
