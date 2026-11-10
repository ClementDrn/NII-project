"""Minimal on-disk result fixtures for the real JobResults parser."""
from typing import Any, Dict, List, Optional


def make_res_json(
    status: str = "Perfect",
    score_route: float = 100.0,
    score_penalty: float = 1.0,
    score_composed: float = 100.0,
    infractions: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Create a res.json structure as written by the simulator.
    
    Args:
        status: Record status (e.g., "Perfect", "Failed", "Timeout").
        score_route: Route completion percentage (0-100).
        score_penalty: Penalty multiplier (0.0-1.0, lower is worse).
        score_composed: Composed score (route * penalty).
        infractions: Dict of infraction lists. If None, defaults to empty.
        
    Returns:
        Dict matching the res.json schema.
        
    Example:
        make_res_json(
            status="Failed",
            score_penalty=0.5,
            infractions={"collisions_vehicle": ["Collision at frame 50"]}
        )
    """
    if infractions is None:
        infractions = {}
    
    return {
        "_checkpoint": {
            "records": [{
                "status": status,
                "scores": {
                    "score_route": score_route,
                    "score_penalty": score_penalty,
                    "score_composed": score_composed,
                },
                "infractions": infractions,
            }]
        }
    }


def make_experiment_info(ground_truth_index: int = 0) -> Dict[str, Any]:
    """
    Create an experiment_info.json structure.
    
    Args:
        ground_truth_index: Index of the ground truth run to compare against.
        
    Returns:
        Dict matching the experiment_info.json schema.
    """
    return {"ground_truth_index": ground_truth_index}


def make_custom_prompts(prompts: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Create a custom_prompts.json structure.
    
    Args:
        prompts: List of prompt texts. If None, defaults to one generic prompt.
        
    Returns:
        List of prompt objects matching custom_prompts.json schema.
        
    Example:
        make_custom_prompts(["Slow down gradually", "Turn left at intersection"])
    """
    if prompts is None:
        prompts = ["Test instruction"]
    
    return [{"text": text} for text in prompts]
