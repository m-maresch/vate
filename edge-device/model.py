from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Dict


@dataclass
class Image:
    id: int
    file_name: str


@dataclass
class Frame:
    id: int
    video: str
    data: Any
    resized_data: Any


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
    det_score: int
    det_category: str
    det_type: DetectionType


ImageList = List[Image]
AnnotationsByImage = Dict[int, List[AnnotationView]]
