from fastapi import BackgroundTasks
import random

from fusion import fuse_edge_cloud_detections
from model import RawDetection, DetectionType
from prediction import get_predictions

EDGE_DETECTION_MODEL_URL: str = "http://127.0.0.1:9090/predictions/mobilenetv2_ssd_visdrone"
CLOUD_DETECTION_MODEL_URL: str = "http://127.0.0.1:9093/predictions/faster_rcnn_visdrone"
EDGE_DETECTION_TIMEOUT: int = 10
CLOUD_DETECTION_TIMEOUT: int = 30


class ObjectDetector:
    last_detections: list[RawDetection]
    last_cloud_detections: list[RawDetection]
    in_progress: bool

    def __init__(self):
        self.last_detections = []
        self.last_cloud_detections = []
        self.in_progress = False

    def detect_objects(self, frame: bytes,
                       background_tasks: BackgroundTasks) -> tuple[DetectionType, list[RawDetection]]:
        if random.randint(0, 100) < 20 and not self.in_progress:
            background_tasks.add_task(self._cloud_detect_objects, frame)

        edge_detections = self._edge_detect_objects(frame)
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

    def _edge_detect_objects(self, frame: bytes) -> list[RawDetection]:
        return get_predictions(EDGE_DETECTION_MODEL_URL, frame, EDGE_DETECTION_TIMEOUT, DetectionType.EDGE)

    def _cloud_detect_objects(self, frame: bytes):
        self.in_progress = True
        print("Cloud detection start")
        detections = get_predictions(CLOUD_DETECTION_MODEL_URL, frame, CLOUD_DETECTION_TIMEOUT, DetectionType.CLOUD)
        self.last_cloud_detections.clear()
        self.last_cloud_detections.extend(detections)
        print("Cloud detection end")
        self.in_progress = False

    def _record(self, detections: list[RawDetection]):
        self.last_detections.clear()
        self.last_detections.extend(detections)
