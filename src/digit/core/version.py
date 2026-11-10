"""Supported configuration formats, independent of the software package version."""

import re

CONFIG_VERSION = "2.0"
JOBS_VERSION = "2.0"
HOSTS_VERSION = "1.0"


def _parse_format_version(value: str) -> tuple[int, int]:
    # Released formats are major.minor. Read the historical -dev spelling too.
    match = (
        re.fullmatch(r"([0-9]+)(?:\.([0-9]+))?(?:-dev)?", value)
        if isinstance(value, str)
        else None
    )
    if match is None:
        raise ValueError(f"Invalid configuration version format: {value!r}")
    return int(match.group(1)), int(match.group(2) or 0)


def are_versions_compatible(imported_version: str, supported_version: str) -> bool:
    """Accept the same major format with a minor no newer than supported.

    A missing minor means zero. Historical ``-dev`` suffixes remain readable;
    package versions such as ``1.0.0`` are not configuration format versions.
    Malformed versions raise ValueError.
    """
    imported_major, imported_minor = _parse_format_version(imported_version)
    supported_major, supported_minor = _parse_format_version(supported_version)
    return imported_major == supported_major and imported_minor <= supported_minor
