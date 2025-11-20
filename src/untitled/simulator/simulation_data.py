from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict


@dataclass
class SimulationMetadata:
    speed: float
    # TODO: Add more metadata fields as needed

@dataclass
class SimulationData:
    image_dir: Path
    frame_count: int
    images: List[str]
    metadata: List[SimulationMetadata]
