from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import json

from untitled.core.base import LOG


MAX_OBJECTS_PER_TYPE = 5


@dataclass
class SimulationMetadata:
    step: int
    control: Dict[str, float]
    speed: float
    position: List[float]
    rotation: List[float]
    forward_vector: List[float]
    right_vector: List[float]
    velocity: List[float]
    angular_velocity: List[float]
    acceleration: List[float]
    scene: List[Dict]  # List of scene objects

@dataclass
class SimulationFrameData:
    image_name: str  # Just the filename, to be combined with image_dir
    metadata: SimulationMetadata
    timestamp: float = 0.0

@dataclass
class SimulationData:
    image_dir: Path
    frame_count: int
    frames: List[SimulationFrameData]


def _clean_vehicle_metadata(obj: dict) -> dict:
    """
    Clean and extract relevant vehicle metadata fields.
    """
    allowed_fields = [
        "class",
        # "color_rgb",
        "color_name",
        "road_id",
        "lane_id",
        "lane_type_str",
        "is_in_junction",
        # "junction_id",
        "same_road_as_ego",
        "same_direction_as_ego",
        "lane_relative_to_ego",
        "light_state",
        "traffic_light_state",
        "is_at_traffic_light",
        "base_type",
        # "number_of_wheels",
        # "extent",
        "position",
        "yaw",
        # "num_points",
        "distance",
        "speed",
        "brake",
        "steer",
        "throttle",
        "id",
        # "role_name",
        # "type_id",
        # "matrix",
    ]
    return {k: v for k, v in obj.items() if k in allowed_fields}

def _clean_pedestrian_metadata(obj: dict) -> dict:
    """
    Clean and extract relevant pedestrian metadata fields.
    """
    allowed_fields = [
        "class",
        "role_name",
        # "gender",
        # "age",
        # "extent",
        "position",
        # "yaw",
        # "num_points",
        "distance",
        "speed",
        "id",
        "lane_type",
        "same_road_as_ego",
        "same_direction_as_ego",
        "lane_relative_to_ego",
        # "matrix"
    ]
    return {k: v for k, v in obj.items() if k in allowed_fields}

def _clean_traffic_light_metadata(obj: dict) -> dict:
    """
    Clean and extract relevant traffic light metadata fields.
    """
    allowed_fields = [
        "class",
        "state",
        "position",
        # "yaw",
        "distance",
        "id",
        "type_id",
        # "matrix",
    ]
    return {k: v for k, v in obj.items() if k in allowed_fields}

def _clean_stop_sign_metadata(obj: dict) -> dict:
    """
    Clean and extract relevant stop sign metadata fields.
    """
    allowed_fields = [
        "class",
        "position",
        # "yaw",
        "distance",
        "id",
        "type_id",
        # "matrix",
    ]
    return {k: v for k, v in obj.items() if k in allowed_fields}

def _clean_static_object_metadata(obj: dict) -> dict:
    """
    Clean and extract relevant static object metadata fields.
    """
    allowed_fields = None
    class_name = obj.get("class", "")
    if class_name == "static_car":
        allowed_fields = [
            "class",
            "id",       # FIXME: inexistant?
            # "extent",
            "position",
            # "yaw",
            "distance",
            # "speed",
            # "num_points",
            # "road_id",
            # "junction_id",
            # "lane_id",
            "on_lane_type",
            "same_road_as_ego",
            "same_direction_as_ego",
            "lane_relative_to_ego",
            # "matrix",
            # "mesh_path",
        ]
    elif class_name == "static_trafficwarning":
        allowed_fields = [
            "class",
            "id",
            # "extent",
            "position",
            # "yaw",
            # "num_points",
            "distance",
            # "mesh_path",
        ]
        # Fix class name (for better understanding by LLMs)
        obj["class"] = "static_traffic_warning"
    else:       # static
        allowed_fields = [
            "class",
            "id",
            "type_id",
            # "extent",
            "position",
            # "yaw",
            # "num_points",
            "distance",
            # "mesh_path",
        ]
    return {k: v for k, v in obj.items() if k in allowed_fields}

def _clean_landmark_metadata(obj: dict) -> dict:
    """
    Clean and extract relevant landmark metadata fields.
    """
    return None     # ignored

def _clean_weather_metadata(obj: dict) -> dict:
    """
    Clean and extract relevant weather metadata fields.
    """
    return None     # ignored

def _clean_ego_info_metadata(obj: dict) -> dict:
    """
    Clean and extract relevant ego vehicle info metadata fields.
    """
    allowed_fields = [
        "class",
        "traffic_light_state",
        "distance_to_junction",
        "ego_lane_number",
        "road_id",
        "lane_id",
        "is_in_junction",
        "is_intersection",
        # "junction_id",
        "next_road_junction",
        # "next_junction_id",
        # "next_road_ids",
        # "next_next_road_ids_ego",
        "ego_lane",
        "left_lanes",
        "right_lanes",
        "num_lanes_same_direction",
        "num_lanes_opposite_direction",
        "lane_change",
        "lane_change_str",
        # "lane_type",
        # "lane_type_str",
        # "left_lane_marking_color",
        "left_lane_marking_color_str",
        # "left_lane_marking_type",
        "left_lane_marking_type_str",
        # "right_lane_marking_color",
        "right_lane_marking_color_str",
        # "right_lane_marking_type",
        "right_lane_marking_type_str",
        "shoulder_left",
        "shoulder_right",
        "parking_left",
        "parking_right",
        "sidewalk_left",
        "sidewalk_right",
        "bike_lane_left",
        "bike_lane_right",
    ]
    return {k: v for k, v in obj.items() if k in allowed_fields}

def _clean_scene_metadata(scene_objects: list) -> dict:
    """
    Clean and extract relevant scene metadata fields.
    """ 
    # 1. Filter fields of scene objects
    objects_per_class: Dict[str, List[dict]] = {
        "ego_car": [],
        "car": [],
        "walker": [],
        "traffic_light": [],
        "stop_sign": [],
        "static": [],
        "static_car": [],
        "static_trafficwarning": [],
        "landmark": [],
        "weather": [],
        "ego_info": []
    }
    ego_car: dict = None
    ego_info: dict = None
    for obj in scene_objects:
        # We will filter based on the class
        class_name = obj.get('class', '')
        object_metadata: dict = None

        # Ego car
        if class_name == "ego_car":
            ego_car = _clean_vehicle_metadata(obj)
            # ego_car's fields will be merged to ego_info when available
            if ego_info is not None:
                # Merge fields
                ego_info.update(ego_car)
                object_metadata = ego_info
        # Other moving vehicles
        elif class_name == "car":
            object_metadata = _clean_vehicle_metadata(obj)
        # Pedestrians
        elif class_name =="walker":
            object_metadata = _clean_pedestrian_metadata(obj)
        # Traffic lights
        elif class_name == "traffic_light":
            object_metadata = _clean_traffic_light_metadata(obj)
        # Stop signs
        elif class_name == "stop_sign":
            object_metadata = _clean_stop_sign_metadata(obj)
        # Static objects
        elif class_name in ["static", "static_car", "static_trafficwarning"]:
            object_metadata = _clean_static_object_metadata(obj)
        # Landmarks
        elif class_name == "landmark":
            object_metadata = _clean_landmark_metadata(obj)
        # Weather
        elif class_name == "weather":
            object_metadata = _clean_weather_metadata(obj)
        # Ego vehicle info
        elif class_name == "ego_info":
            ego_info = _clean_ego_info_metadata(obj)
            # ego_car's fields will be merged to ego_info when available 
            if ego_car is not None:
                # Merge fields
                ego_info.update(ego_car)
                object_metadata = ego_info
        # Unknown classes
        else:
            LOG.warning(f"Unknown scene object class '{class_name}' in metadata. Ignoring.")
        
        # Add cleaned metadata if available
        if object_metadata is not None:
            objects_per_class[class_name].append(object_metadata)

    # 2. Remove static objects that are behind the ego vehicle
    # NOTE: Even if static_trafficwarning was renamed to static_traffic_warning, it is still the prior in objects_per_class
    static_classes = ["static", "static_car", "static_trafficwarning"]
    for class_name in static_classes:
        objects_per_class[class_name] = [
            obj for obj in objects_per_class[class_name]
            if obj.get("position", [0])[0] >= 0
        ]

    # 3. Limit the number of objects per type to avoid excessive metadata size
    # Only keep the closest N objects per type
    MAX_OBJECTS_PER_TYPE = 5
    limited_scene_metadata: List[dict] = []
    for class_name, objects in objects_per_class.items():
        # Skip ego_car and ego_info as they should always be kept
        if class_name in ["ego_car", "ego_info"]:
            limited_scene_metadata.extend(objects)
            continue

        # If object count does not exceed the limit, keep all
        if len(objects) <= MAX_OBJECTS_PER_TYPE:
            limited_scene_metadata.extend(objects)
            continue

        # Sort objects by distance to ego vehicle
        objects_sorted = sorted(objects, key=lambda obj: obj.get('distance', float('inf')))
        limited_scene_metadata.extend(objects_sorted[:MAX_OBJECTS_PER_TYPE])

    return limited_scene_metadata
    

def load_metadata(metadata_path: Path) -> SimulationMetadata:
    """
    Load simulation metadata from a JSON file. Only the relevant fields are extracted.
    """
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

        # Collect scene objects and clean data to only retain relevant fields.
        scene_objects = data.get('scene_objects', [])
        scene_metadata = _clean_scene_metadata(scene_objects)

        return SimulationMetadata(
            step=data.get('step', 0),
            control=data.get('control', {}),
            speed=data.get('speed', None),
            position=data.get('location', []),
            rotation=data.get('rotation', []),
            forward_vector=data.get('forward_vector', []),
            right_vector=data.get('right_vector', []),
            velocity=data.get('velocity', []),
            angular_velocity=data.get('angular_velocity', []),
            acceleration=data.get('acceleration', []),
            scene=scene_metadata
        )   

def get_front_car(metadata: SimulationMetadata) -> Optional[dict]:
    """
    Get the closest car in front of the ego vehicle from the scene metadata.
    """
    cars = [obj for obj in metadata.scene if obj.get('class') == 'car']
    if not cars:
        return None
    
    # Filter by same lane/road/direction
    car_candidates = []
    for car in cars:
        try:
            if bool(car.get('same_road_as_ego')) and bool(car.get('same_direction_as_ego')) and int(car.get('lane_relative_to_ego', 0)) == 0:
                car_candidates.append(car)
        except (ValueError, TypeError):
            LOG.warning(f"Car object with id {car.get('id')} has invalid metadata fields. Skipping for front car detection.")
            continue

    if not car_candidates:
        return None

    # Assuming the position's x-coordinate indicates the distance in front of the ego vehicle
    # Get closest car among candidates
    front_car = min(car_candidates, key=lambda car: car.get('position', [float('inf')])[0])
    
    return front_car

def get_ego_info(metadata: SimulationMetadata) -> Optional[dict]:
    """
    Get the ego vehicle info from the scene metadata.
    """
    for obj in metadata.scene:
        if obj.get('class') == 'ego_info':
            return obj
    return None

def get_left_lane_types(metadata: SimulationMetadata) -> List[str]:
    """
    Get the types of the lanes to the left of the ego vehicle from the scene metadata.
    """
    ego_info = get_ego_info(metadata)
    if ego_info is not None:
        left_lanes = ego_info.get('left_lanes', [])
        if left_lanes:
            return [lane.get('type', None) for lane in left_lanes]
    return []

def get_right_lane_types(metadata: SimulationMetadata) -> List[str]:
    """
    Get the types of the lanes to the right of the ego vehicle from the scene metadata.
    """
    ego_info = get_ego_info(metadata)
    if ego_info is not None:
        right_lanes = ego_info.get('right_lanes', [])
        if right_lanes:
            return [lane.get('type', None) for lane in right_lanes]
    return []
