from enum import Enum


class ExperimentNature(Enum):
    """
    Enum representing the nature of an experiment.
    
    - ADVERSARIAL: Instructions designed to cause failures (crash testing)
    - CORRECTIVE: Instructions designed to prevent failures (safety intervention)
    """
    ADVERSARIAL = "adversarial"
    CORRECTIVE = "corrective"
    
    @classmethod
    def from_string(cls, value: str) -> "ExperimentNature":
        """
        Create an ExperimentNature from a string value.
        
        Args:
            value: String representation ("adversarial" or "corrective")
            
        Returns:
            The corresponding ExperimentNature enum value
            
        Raises:
            ValueError: If the value is not recognized
        """
        value_lower = value.lower().strip()
        if value_lower == "adversarial":
            return cls.ADVERSARIAL
        elif value_lower == "corrective":
            return cls.CORRECTIVE
        else:
            raise ValueError(f"Unknown experiment nature: '{value}'. Expected 'adversarial' or 'corrective'.")
