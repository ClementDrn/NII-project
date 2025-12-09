# This file contains the supported versions for various components of the project.

# Version for configuration files
CONFIG_VERSION = "2.0-dev"
JOBS_VERSION = "1.1-dev"


def are_versions_compatible(imported_version: str, supported_version: str) -> bool:
    """
    Check if the file version is compatible with the supported version.
    For compatibility, the major version numbers must match, and the minor version must be less than or equal to the supported minor version.

    Parameters:
    - imported_version (str): The version string from the imported file (e.g., "1.2").
    - supported_version (str): The supported version string (e.g., "2.0").
    
    Returns:
    - bool: True if versions are compatible, False otherwise.
    
    Raises:
    - ValueError: If version format is invalid.
    """
    # Remove any suffixes like "-dev" from the version strings
    imported_version = imported_version.split('-')[0]
    supported_version = supported_version.split('-')[0]

    # Split imported version into major and minor numbers
    split_imported_version = imported_version.split('.')
    if len(split_imported_version) < 1 or len(split_imported_version) > 2:
        raise ValueError(f"Invalid imported version format: {imported_version}")
    imported_major = int(split_imported_version[0])
    imported_minor = int(split_imported_version[1]) if len(split_imported_version) > 1 else 0

    # Split supported version into major and minor numbers
    split_supported_version = supported_version.split('.')
    if len(split_supported_version) < 1 or len(split_supported_version) > 2:
        raise ValueError(f"Invalid supported version format: {supported_version}")
    supported_major = int(split_supported_version[0])
    supported_minor = int(split_supported_version[1]) if len(split_supported_version) > 1 else 0

    # Majors must match and imported minor must be less than or equal to supported minor
    return (imported_major == supported_major) and (imported_minor <= supported_minor)
