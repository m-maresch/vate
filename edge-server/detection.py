from fastapi import BackgroundTasks
from typing import List, Tuple
import random

from edge_prediction import get_edge_predictions
from fusion import fuse_edge_cloud_detections
from model import RawDetection, DetectionType
from cloud_prediction import get_cloud_predictions


class ObjectDetector:
    last_detections: List[RawDetection]
    last_cloud_detections: List[RawDetection]
    in_progress: bool

    def __init__(self):
        self.last_detections = []
        self.last_cloud_detections = []
        self.in_progress = False

    def detect_objects(self, frame: bytes,
                       background_tasks: BackgroundTasks) -> Tuple[DetectionType, List[RawDetection]]:
        if random.randint(0, 100) < 20 and not self.in_progress:
            background_tasks.add_task(self._cloud_detect_objects, frame)

        edge_detections = get_edge_predictions(frame)
        if self.last_cloud_detections:
            cloud_detections = self.last_cloud_detections.copy()
            self.last_cloud_detections.clear()
            detections = fuse_edge_cloud_detections(edge_detections, cloud_detections, DetectionType.CLOUD)
            self._record(detections)
            return DetectionType.CLOUD, detections
        else:
            detections = fuse_edge_cloud_detections(self.last_detections, edge_detections, DetectionType.EDGE)
            self._record(detections)
            return DetectionType.EDGE, detections

    def _cloud_detect_objects(self, frame: bytes):
        self.in_progress = True
        print("Cloud detection start")
        detections = get_cloud_predictions(frame)
        self.last_cloud_detections.clear()
        self.last_cloud_detections.extend(detections)
        print("Cloud detection end")
        self.in_progress = False

    def _record(self, detections: List[RawDetection]):
        self.last_detections.clear()
        self.last_detections.extend(detections)
