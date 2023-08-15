from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel
from typing import List


class DetectionType(Enum):
    EDGE = 1
    CLOUD = 2


@dataclass
class RawDetection:
    class_name: str
    score: float
    bbox: List[float]
    last_type: DetectionType


class Detection(BaseModel):
    category: str
    score: int
    bbox: List[int]


class DetectionResponse(BaseModel):
    frame_id: int
    type: str
    detections: List[Detection] = []
