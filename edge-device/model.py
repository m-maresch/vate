from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Dict, Tuple


@dataclass
class Image:
    id: int
    file_name: str


@dataclass
class Frame:
    id: int
    video: str
    data: Any
    edge_data: Any
    cloud_data: Any


@dataclass
class Dimensions:
    edge_processing_width: int
    edge_processing_height: int
    cloud_processing_width: int
    cloud_processing_height: int


class DetectionType(Enum):
    EDGE = 1
    CLOUD = 2


@dataclass
class Detection:
    category: str
    score: int
    bbox: List[float]


@dataclass
class DetectionView:
    frame_id: int
    x: int
    y: int
    w: int
    h: int
    score: int
    category: int
    type: DetectionType
    tracked: bool


@dataclass
class AnnotationView:
    x: int
    y: int
    w: int
    h: int
    category: int


@dataclass
class TrackerRecord:
    raw_tracker: Any
    det_bbox: List[float]
    det_score: int
    det_category: str
    det_type: DetectionType


DetectionsWithTypes = List[Tuple[Detection, DetectionType]]
AnnotationsByImage = Dict[int, List[AnnotationView]]
