from untitled.data.scene_object import (
    SceneObjectModifier,
    SceneObjectModifierType,
    SceneObjectType,
    PositionModifier,
    parse_scene_objects,
)

from math import isfinite


# Scene object types that can be checked against the collected actors.
_ACTOR_OBJECT_TYPES = {
    SceneObjectType.CAR,
    SceneObjectType.MOTORBIKE,
    SceneObjectType.BICYCLE,
    SceneObjectType.PEDESTRIAN,
    SceneObjectType.TRAFFIC_LIGHT,
    SceneObjectType.TRAFFIC_SIGN,
}

# CARLA lane names and the corresponding scene object types.
_LANE_OBJECT_TYPES = {
    "driving": SceneObjectType.DRIVING_LANE,
    "parking": SceneObjectType.PARKING_LANE,
    "shoulder": SceneObjectType.SHOULDER,
    "sidewalk": SceneObjectType.SIDEWALK,
    "biking": SceneObjectType.BIKE_LANE,
}

# ego_info also reports lane presence through fields like shoulder_left.
_LANE_PRESENCE_FIELDS = {
    "parking": SceneObjectType.PARKING_LANE,
    "shoulder": SceneObjectType.SHOULDER,
    "sidewalk": SceneObjectType.SIDEWALK,
    "bike_lane": SceneObjectType.BIKE_LANE,
}


def _get_actor_object_type(actor: dict) -> SceneObjectType | None:
    """
    Determines the scene object type of an actor in the metadata.
    Returns None for unsupported classes, including the ego vehicle.
    """
    # WARN: Not all classes are collected in the metdata nor are supported for hallucination checks.
    class_name = actor.get("class")
    if class_name in {"car", "static_car"}:
        # NOTE: data_agent.py collects bicycles and motorbikes under the car class too.
        # This is quite misleading and should be fixed in the future. 
        base_type = str(actor.get("base_type", "")).lower()
        if base_type in {"bicycle", "bike"}:
            return SceneObjectType.BICYCLE
        if base_type in {"motorcycle", "motorbike", "scooter"}:
            return SceneObjectType.MOTORBIKE
        return SceneObjectType.CAR
    elif class_name == "walker":
        return SceneObjectType.PEDESTRIAN
    elif class_name == "traffic_light":
        return SceneObjectType.TRAFFIC_LIGHT
    elif class_name == "stop_sign":
        return SceneObjectType.TRAFFIC_SIGN
    
    # Return None if the actor class is unsupported for hallucination checks
    return None


def _get_lane_candidates(scene: list) -> tuple[dict, set]:
    """
    Collects lane candidates from ego_info, grouped by scene object type.
    Also returns the types whose absence can be checked from the metadata.

    Each candidate records its side relative to the ego vehicle. The ego lane
    has no side, since it is neither on the left nor on the right.
    """
    lane_candidates = {object_type: [] for object_type in _LANE_OBJECT_TYPES.values()}
    verifiable_lane_types = set()

    for ego_info in scene:
        if ego_info.get("class") != "ego_info":
            continue

        # 1. Check whether both side inventories are complete.
        # Empty lists mean no lanes. Missing fields or incomplete entries do
        # not give enough information to conclude that a lane type is absent.
        side_inventories_complete = True
        for field_name in ["left_lanes", "right_lanes"]:
            lanes = ego_info.get(field_name)
            if not isinstance(lanes, list):
                side_inventories_complete = False
                break
            for lane in lanes:
                if not isinstance(lane, dict):
                    side_inventories_complete = False
                    break
                lane_type = lane.get("type", lane.get("type:"))
                if not isinstance(lane_type, str):
                    side_inventories_complete = False
                    break

        if side_inventories_complete:
            verifiable_lane_types.update(_LANE_OBJECT_TYPES.values())

        # 2. Collect the ego lane and the lanes on each side.
        lane_fields = [
            ("ego_lane", None),
            ("left_lanes", PositionModifier.LEFT),
            ("right_lanes", PositionModifier.RIGHT),
        ]
        for field_name, side in lane_fields:
            lanes = ego_info.get(field_name)
            if field_name == "ego_lane" and isinstance(lanes, dict):
                lanes = [lanes]
            if not isinstance(lanes, list):
                continue

            for lane in lanes:
                if not isinstance(lane, dict):
                    continue
                # Raw metadata contains a typo: "type:" instead of "type".
                lane_type = str(lane.get("type", lane.get("type:", ""))).lower()
                object_type = _LANE_OBJECT_TYPES.get(lane_type)
                if object_type is not None:
                    lane_candidates[object_type].append({"side": side})

        # 3. Add presence reported by side flags.
        # A false flag does not remove a listed lane (e.g. a narrow shoulder).
        for field_prefix, object_type in _LANE_PRESENCE_FIELDS.items():
            left_present = ego_info.get(f"{field_prefix}_left")
            right_present = ego_info.get(f"{field_prefix}_right")
            if isinstance(left_present, bool) and isinstance(right_present, bool):
                verifiable_lane_types.add(object_type)

            for side_name, side in [("left", PositionModifier.LEFT), ("right", PositionModifier.RIGHT)]:
                lane_present = ego_info.get(f"{field_prefix}_{side_name}")
                if lane_present is True:
                    lane_candidates[object_type].append({"side": side})

    return lane_candidates, verifiable_lane_types


def _count_modifier_hallucination(
    modifier: SceneObjectModifier,
    candidate: dict,
    object_type: SceneObjectType,
    ego_info: dict,
) -> int:
    """
    Returns 1 if the candidate contradicts a color or position modifier.
    Returns 0 if the modifier matches or cannot be checked from the metadata.
    """
    # Traffic light instructions concern the signal affecting the ego vehicle.
    # Other actors use their own color.
    if modifier.type == SceneObjectModifierType.COLOR:
        if object_type == SceneObjectType.TRAFFIC_LIGHT:
            color_name = ego_info.get("traffic_light_state")
        else:
            color_name = candidate.get("color_name")
        if not isinstance(color_name, str) or not color_name.strip():
            return 0
        if color_name.strip().lower() != modifier.value.value:  # type: ignore
            return 1
        return 0

    # Unknown modifiers are counted, but their fidelity cannot be checked.
    if modifier.type != SceneObjectModifierType.POSITION:
        return 0

    # Lane metadata provides left/right presence, but no front/back position.
    if object_type in _LANE_OBJECT_TYPES.values():
        if modifier.value not in {PositionModifier.LEFT, PositionModifier.RIGHT}:
            return 0
        if candidate["side"] != modifier.value:
            return 1
        return 0

    # Actor positions use ego-relative coordinates: x is forward, y is right.
    if modifier.value in {PositionModifier.FRONT, PositionModifier.BACK}:
        coordinate_index = 0
    else:
        coordinate_index = 1

    position = candidate.get("position")
    if not isinstance(position, (list, tuple)) or len(position) <= coordinate_index:
        return 0
    coordinate = position[coordinate_index]
    if isinstance(coordinate, bool) or not isinstance(coordinate, (float, int)):
        return 0
    if not isfinite(coordinate):
        return 0

    # Zero is neither front/back nor left/right.
    if modifier.value in {PositionModifier.FRONT, PositionModifier.RIGHT}:
        position_matches = coordinate > 0
    else:
        position_matches = coordinate < 0
    if position_matches:
        return 0
    return 1


def check_hallucinations_in_instruction(instruction: str, metadata: dict) -> dict:
    """
    Parse scene objects and their modifiers in the instruction and check for hallucinations against the metadata.

    Counts every scene object mention and its modifiers. Checks hallucinations
    only for supported object types and color/position modifiers with metadata.
    Repeated mentions can match the same metadata object.

    Args:
        instruction: The instruction string to analyze.
        metadata: Raw metadata (scene_objects) or cleaned metadata (scene).

    Return:
        A dictionary containing the object count, modifier count, object
        hallucination count, and modifier hallucination count.
    """
    # 1. Count all detected objects and modifiers, including unknown modifiers.
    scene_objects = parse_scene_objects(instruction)
    counts = {
        "object_count": len(scene_objects),
        "modifier_count": sum(len(scene_object.modifiers) for scene_object in scene_objects),
        "object_hallucination_count": 0,
        "modifier_hallucination_count": 0,
    }

    # 2. Load the scene inventory. Raw metadata takes precedence when supplied.
    if "scene_objects" in metadata:
        scene_metadata = metadata.get("scene_objects")
    else:
        scene_metadata = metadata.get("scene")
    if not isinstance(scene_metadata, list):
        return counts
    scene_metadata = [actor for actor in scene_metadata if isinstance(actor, dict)]
    lane_candidates, verifiable_lane_types = _get_lane_candidates(scene_metadata)

    # ego_info reports the traffic light state relevant to the ego vehicle.
    # Nearby lights may control other lanes, so their states are not used.
    ego_info = {}
    for actor in scene_metadata:
        if actor.get("class") == "ego_info":
            ego_info = actor
            break

    # 3. Find candidates of the same type for each mention.
    for scene_object in scene_objects:
        if scene_object.type in _ACTOR_OBJECT_TYPES:
            candidates = [
                actor for actor in scene_metadata
                if _get_actor_object_type(actor) == scene_object.type
            ]
        elif scene_object.type in verifiable_lane_types or lane_candidates.get(scene_object.type):
            candidates = lane_candidates[scene_object.type]
        else:
            # Unsupported types (including junctions) and missing lane evidence.
            continue

        if not candidates:
            counts["object_hallucination_count"] += 1
            # Do not also count its modifiers as hallucinations.
            continue

        # 4. Check all modifiers against one candidate at a time.
        # For "red car ahead", a red car behind and a blue car ahead cannot
        # jointly satisfy the description. Keep the smallest mismatch count.
        candidate_hallucination_counts = []
        for candidate in candidates:
            modifier_hallucination_count = 0
            for modifier in scene_object.modifiers:
                modifier_hallucination_count += _count_modifier_hallucination(
                    modifier, candidate, scene_object.type, ego_info,
                )
            candidate_hallucination_counts.append(modifier_hallucination_count)

        counts["modifier_hallucination_count"] += min(candidate_hallucination_counts)

    return counts
