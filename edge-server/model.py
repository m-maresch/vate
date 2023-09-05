from dataclasses import dataclass
from pydantic import BaseModel
from typing import List


@dataclass
class RawDetection:
    class_name: str
    score: float
    bbox: List[float]


class Detection(BaseModel):
    category: str
    score: int
    bbox: List[float]


class DetectionResponse(BaseModel):
    detections: List[Detection] = []
