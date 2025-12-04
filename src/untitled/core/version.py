# This file contains the supported versions for various components of the project.
from untitled.core.base import LOG


# Version for configuration files
CONFIG_VERSION = "0.0"
JOBS_VERSION = "0.0"


def are_versions_compatible(imported_version: str, supported_version: str) -> bool:
    """
    Check if the file version is compatible with the supported version.
    For compatibility, the major version numbers must match, and the minor version must be less than or equal to the supported minor version.

    Parameters:
    - imported_version (str): The version string from the imported file (e.g., "1.2").
    - supported_version (str): The supported version string (e.g., "2.0").
    """
    # Split imported version into major and minor numbers
    split_imported_version = imported_version.split('.')
    if len(split_imported_version) < 1 or len(split_imported_version) > 2:
        LOG.error(f"Invalid imported version format: {imported_version}")
        return False
    imported_major = int(split_imported_version[0])
    imported_minor = int(split_imported_version[1]) if len(split_imported_version) > 1 else 0

    # Split supported version into major and minor numbers
    split_supported_version = supported_version.split('.')
    if len(split_supported_version) < 1 or len(split_supported_version) > 2:
        LOG.error(f"Invalid supported version format: {supported_version}")
        return False
    supported_major = int(split_supported_version[0])
    supported_minor = int(split_supported_version[1]) if len(split_supported_version) > 1 else 0

    # Majors must match and imported minor must be less than or equal to supported minor
    return (imported_major == supported_major) and (imported_minor <= supported_minor)
