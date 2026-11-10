import pytest

from digit.core.version import are_versions_compatible


@pytest.mark.parametrize(
    ("imported", "supported", "compatible"),
    [
        ("2.0", "2.0", True),
        ("2.0-dev", "2.0", True),
        ("2", "2.0", True),
        ("2.0", "2.1", True),
        ("2.1", "2.0", False),
        ("1.0", "2.0", False),
    ],
)
def test_format_compatibility(imported, supported, compatible):
    assert are_versions_compatible(imported, supported) is compatible


@pytest.mark.parametrize("value", ["", "1.0.0", "-1", "2.foo", "2.0-unknown", None])
def test_invalid_format_versions_raise(value):
    with pytest.raises(ValueError, match="Invalid configuration version"):
        are_versions_compatible(value, "2.0")
