import asyncio
from typing import List, Tuple, Dict

from fusion import fuse_edge_cloud_detections
from model import RawDetection, DetectionType
from prediction import Predictor


class ObjectDetector:
    edge_predictor: Predictor
    cloud_predictor: Predictor

    last_detections: Dict[str, List[RawDetection]]
    last_cloud_detections: Dict[str, List[RawDetection]]
    in_progress: bool

    def __init__(self, edge_predictor: Predictor, cloud_predictor: Predictor):
        self.edge_predictor = edge_predictor
        self.cloud_predictor = cloud_predictor
        self.last_detections = dict()
        self.last_cloud_detections = dict()
        self.in_progress = False

    async def detect_objects(self, video: str, frame) -> Tuple[DetectionType, List[RawDetection]]:
        if not self.in_progress:
            asyncio.ensure_future(self._cloud_detect_objects(video, frame))

        edge_detections = self.edge_predictor.get_predictions(frame)
        if self.last_cloud_detections.get(video, []):
            cloud_detections = self.last_cloud_detections[video].copy()
            self.last_cloud_detections[video] = []
            detections = fuse_edge_cloud_detections(edge_detections, cloud_detections, DetectionType.CLOUD)
            self._record(video, detections)
            return DetectionType.CLOUD, detections
        else:
            detections = fuse_edge_cloud_detections(self.last_detections.get(video, []), edge_detections,
                                                    DetectionType.EDGE)
            self._record(video, detections)
            return DetectionType.EDGE, detections

    async def _cloud_detect_objects(self, video: str, frame):
        self.in_progress = True
        print("Cloud detection start")
        detections = self.cloud_predictor.get_predictions(frame)
        self.last_cloud_detections[video] = detections
        print("Cloud detection end")
        self.in_progress = False

    def _record(self, video: str, detections: List[RawDetection]):
        self.last_detections[video] = detections
