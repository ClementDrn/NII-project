from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import json


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


def load_metadata(metadata_path: Path) -> SimulationMetadata:
    """
    Load simulation metadata from a JSON file.
    """
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

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
        )   
