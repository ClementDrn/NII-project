import pytest
from untitled.generator.experiment_nature import ExperimentNature


class TestExperimentNature:
    """Test suite for ExperimentNature enum."""
    
    def test_adversarial_value(self):
        """Test ADVERSARIAL enum value."""
        assert ExperimentNature.ADVERSARIAL.value == "adversarial"
    
    def test_corrective_value(self):
        """Test CORRECTIVE enum value."""
        assert ExperimentNature.CORRECTIVE.value == "corrective"
    
    def test_from_string_adversarial(self):
        """Test from_string with 'adversarial'."""
        assert ExperimentNature.from_string("adversarial") == ExperimentNature.ADVERSARIAL
    
    def test_from_string_corrective(self):
        """Test from_string with 'corrective'."""
        assert ExperimentNature.from_string("corrective") == ExperimentNature.CORRECTIVE
    
    def test_from_string_case_insensitive(self):
        """Test from_string is case-insensitive."""
        assert ExperimentNature.from_string("ADVERSARIAL") == ExperimentNature.ADVERSARIAL
        assert ExperimentNature.from_string("Corrective") == ExperimentNature.CORRECTIVE
        assert ExperimentNature.from_string("AdVeRsArIaL") == ExperimentNature.ADVERSARIAL
    
    def test_from_string_with_whitespace(self):
        """Test from_string handles whitespace."""
        assert ExperimentNature.from_string("  adversarial  ") == ExperimentNature.ADVERSARIAL
        assert ExperimentNature.from_string("\tcorrective\n") == ExperimentNature.CORRECTIVE
    
    def test_from_string_invalid(self):
        """Test from_string raises ValueError for invalid values."""
        with pytest.raises(ValueError) as exc_info:
            ExperimentNature.from_string("invalid")
        
        assert "Unknown experiment nature" in str(exc_info.value)
        assert "'invalid'" in str(exc_info.value)
    
    def test_from_string_empty(self):
        """Test from_string raises ValueError for empty string."""
        with pytest.raises(ValueError):
            ExperimentNature.from_string("")
    
    def test_from_string_similar_but_wrong(self):
        """Test from_string raises ValueError for similar but wrong values."""
        with pytest.raises(ValueError):
            ExperimentNature.from_string("adversary")
        
        with pytest.raises(ValueError):
            ExperimentNature.from_string("correction")
