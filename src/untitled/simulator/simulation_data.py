from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict


@dataclass
class SimulationMetadata:
    speed: float
    # TODO: Add more metadata fields as needed

@dataclass
class SimulationFrameData:
    image_name: str  # Just the filename, to be combined with image_dir
    metadata: SimulationMetadata

@dataclass
class SimulationData:
    image_dir: Path
    frame_count: int
    frames: List[SimulationFrameData]
