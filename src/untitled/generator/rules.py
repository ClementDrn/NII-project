from untitled.generator.instruction import InstructionCategory
from untitled.simulator.simulation_data import SimulationFrameData, get_front_car, get_ego_info, get_left_lane_types, get_right_lane_types

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable
import random



# --- Base Rule class ---------------------------------------------------------

@dataclass
class Rule(ABC):
    """
    Represents a rule for instruction generation, including the logic to determine injection windows.
    """
    category: InstructionCategory   # The instruction category this rule applies to
    injection_duration: float       # Duration of the injection window in seconds
    target_module_logic: str        # Description of the target module logic (e.g., "affects speed control module")
    reasoning: str                  # Explanation of why this window is critical (e.g., "the vehicle is approaching a sharp turn")
    is_valid_for_this_frame: Callable[[SimulationFrameData, float], bool]   # Function to check if the rule is valid for a given frame and (optional) variable


# --- Custom rules are written below ------------------------------------------

_RULES_BY_CATEGORY: Dict[InstructionCategory, Rule] = {
    # Speed change
    InstructionCategory.INCREASE_RELATIVE_SPEED: Rule(
        category=InstructionCategory.INCREASE_RELATIVE_SPEED,
        injection_duration=3.0,
        target_module_logic="Affects speed control module.",
        reasoning="Increasing speed by X is always a realistic maneuver.",
        is_valid_for_this_frame=lambda frame, var: True  # Always true
    ),
    InstructionCategory.DECREASE_RELATIVE_SPEED: Rule(
        category=InstructionCategory.DECREASE_RELATIVE_SPEED,
        injection_duration=3.0,
        target_module_logic="Affects speed control module.",
        reasoning="Decreasing speed by X is realistic as we are not stopped.",
        is_valid_for_this_frame=lambda frame, var: frame.metadata.speed > 0
    ),
    InstructionCategory.INCREASE_ABSOLUTE_SPEED: Rule(
        category=InstructionCategory.INCREASE_ABSOLUTE_SPEED,
        injection_duration=3.0,
        target_module_logic="Affects speed control module.",
        reasoning="Increasing speed to X is realistic as we are currently below that speed.",
        is_valid_for_this_frame=lambda frame, var: frame.metadata.speed < var
    ),
    InstructionCategory.DECREASE_ABSOLUTE_SPEED: Rule(
        category=InstructionCategory.DECREASE_ABSOLUTE_SPEED,
        injection_duration=3.0,
        target_module_logic="Affects speed control module.",
        reasoning="Decreasing speed to X is realistic as we are currently above that speed and not stopped.",
        is_valid_for_this_frame=lambda frame, var: frame.metadata.speed > var
    ),
    # Distance change
    InstructionCategory.INCREASE_RELATIVE_DISTANCE: Rule(
        category=InstructionCategory.INCREASE_RELATIVE_DISTANCE,
        injection_duration=3.0,
        target_module_logic="Affects distance control module.",
        reasoning="Increasing distance by X is realistic if there is a front car within 100 meters.",
        is_valid_for_this_frame=lambda frame, var: (car := get_front_car(frame.metadata)) is not None and car.get('distance', float('inf')) < 100
    ),
    InstructionCategory.DECREASE_RELATIVE_DISTANCE: Rule(
        category=InstructionCategory.DECREASE_RELATIVE_DISTANCE,
        injection_duration=3.0,
        target_module_logic="Affects distance control module.",
        reasoning="Decreasing distance by X is realistic if there is a front car within 100 meters.",
        is_valid_for_this_frame=lambda frame, var: (car := get_front_car(frame.metadata)) is not None and car.get('distance', float('inf')) < 100
    ),
    InstructionCategory.INCREASE_ABSOLUTE_DISTANCE: Rule(
        category=InstructionCategory.INCREASE_ABSOLUTE_DISTANCE,
        injection_duration=3.0,
        target_module_logic="Affects distance control module.",
        reasoning="Increasing distance to X is realistic if there is a front car within that distance.",
        is_valid_for_this_frame=lambda frame, var: (car := get_front_car(frame.metadata)) is not None and car.get('distance', float('inf')) < var
    ),
    InstructionCategory.DECREASE_ABSOLUTE_DISTANCE: Rule(
        category=InstructionCategory.DECREASE_ABSOLUTE_DISTANCE,
        injection_duration=3.0,
        target_module_logic="Affects distance control module.",
        reasoning="Decreasing distance to X is realistic if there is a front car currently farther than that distance.",
        is_valid_for_this_frame=lambda frame, var: (car := get_front_car(frame.metadata)) is not None and car.get('distance', float('inf')) > var
    ),
    # Overtake
    InstructionCategory.OVERTAKE_LEFT: Rule(
        category=InstructionCategory.OVERTAKE_LEFT,
        injection_duration=3.0,
        target_module_logic="Affects lane change and overtaking logic.",
        reasoning="Overtaking left is realistic if there is a front car within 100 meters.",
        is_valid_for_this_frame=lambda frame, var: (car := get_front_car(frame.metadata)) is not None and car.get('distance', float('inf')) < 100
    ),
    InstructionCategory.OVERTAKE_RIGHT: Rule(
        category=InstructionCategory.OVERTAKE_RIGHT,
        injection_duration=3.0,
        target_module_logic="Affects lane change and overtaking logic.",
        reasoning="Overtaking right is realistic if there is a front car within 100 meters.",
        is_valid_for_this_frame=lambda frame, var: (car := get_front_car(frame.metadata)) is not None and car.get('distance', float('inf')) < 100
    ),
    # Turn
    InstructionCategory.TURN_LEFT: Rule(
        category=InstructionCategory.TURN_LEFT,
        injection_duration=3.0,
        target_module_logic="Affects steering control module.",
        reasoning="Turning left is realistic if there is a junction within 10 meters.",
        is_valid_for_this_frame=lambda frame, var: (info := get_ego_info(frame.metadata)) is not None and info.get('distance_to_junction', float('inf')) < 10
    ),
    InstructionCategory.TURN_RIGHT: Rule(
        category=InstructionCategory.TURN_RIGHT,
        injection_duration=3.0,
        target_module_logic="Affects steering control module.",
        reasoning="Turning right is realistic if there is a junction within 10 meters.",
        is_valid_for_this_frame=lambda frame, var: (info := get_ego_info(frame.metadata)) is not None and info.get('distance_to_junction', float('inf')) < 10
    ),
    InstructionCategory.GO_STRAIGHT: Rule(
        category=InstructionCategory.GO_STRAIGHT,
        injection_duration=3.0,
        target_module_logic="Affects steering control module.",
        reasoning="Going straight is realistic if there is a junction within 10 meters.",
        is_valid_for_this_frame=lambda frame, var: (info := get_ego_info(frame.metadata)) is not None and info.get('distance_to_junction', float('inf')) < 10
    ),
    # Change lane
    InstructionCategory.LANE_CHANGE_LEFT: Rule(
        category=InstructionCategory.LANE_CHANGE_LEFT,
        injection_duration=3.0,
        target_module_logic="Affects lane change logic.",
        reasoning="Changing lane to the left is realistic if there is a lane on the left.",
        is_valid_for_this_frame=lambda frame, var: "Driving" in get_left_lane_types(frame.metadata)
    ),
    InstructionCategory.LANE_CHANGE_RIGHT: Rule(
        category=InstructionCategory.LANE_CHANGE_RIGHT,
        injection_duration=3.0,
        target_module_logic="Affects lane change logic.",
        reasoning="Changing lane to the right is realistic if there is a lane on the right.",
        is_valid_for_this_frame=lambda frame, var: "Driving" in get_right_lane_types(frame.metadata)
    ),
    # Pull over
    InstructionCategory.PULL_OVER: Rule(
        category=InstructionCategory.PULL_OVER,
        injection_duration=3.0,
        target_module_logic="Affects lane change and stopping logic.",
        reasoning="Pulling over is realistic if there is a shoulder or parking lane on the right.",
        is_valid_for_this_frame=lambda frame, var: (lanes := get_right_lane_types(frame.metadata), []) and any(lane in lanes for lane in ["Shoulder", "Parking"])
    ),
}


def get_rule_for_category(category: InstructionCategory) -> Optional[Rule]:
    """
    Fetch the appropriate rule for the given instruction category.
    """
    return _RULES_BY_CATEGORY.get(category, None)
