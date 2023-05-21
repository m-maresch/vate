from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel


class DetectionType(Enum):
    EDGE = 1
    CLOUD = 2


@dataclass
class RawDetection:
    class_name: str
    score: float
    bbox: list[float]
    last_type: DetectionType


class Detection(BaseModel):
    category: str
    score: int
    bbox: list[int]


class DetectionResponse(BaseModel):
    frame_id: int
    type: str
    detections: list[Detection] = []
