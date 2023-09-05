import asyncio
from typing import List, Tuple, Union

from cloud_server import CloudServer
from edge_server import EdgeServer
from fusion import fuse_edge_cloud_detections
from model import Detection, DetectionType, Frame
from track import MultiObjectTracker


class EdgeCloudObjectDetector:
    edge_server: EdgeServer
    cloud_server: CloudServer
    object_tracker: MultiObjectTracker

    prev_frames: List[Frame]
    last_detections: List[Detection]
    cloud_detection_done: bool
    last_cloud_detections: List[Detection]

    def __init__(self, edge_server: EdgeServer, cloud_server: CloudServer, object_tracker: MultiObjectTracker):
        self.edge_server = edge_server
        self.cloud_server = cloud_server
        self.object_tracker = object_tracker
        self.prev_frames = []
        self.last_detections = []
        self.cloud_detection_done = True
        self.last_cloud_detections = []

    def detect_objects(self, frame: Frame, wait: bool) -> Tuple[Union[DetectionType, None], List[Detection]]:
        edge_detections = self.edge_server.detect_objects(frame, wait)

        if not edge_detections:
            return None, []

        if self.cloud_detection_done:
            self.object_tracker.reset_objects()
            for detection in self.last_cloud_detections:
                self.object_tracker.add_object(frame, detection, DetectionType.CLOUD)

            asyncio.get_event_loop().run_in_executor(None, self._cloud_detect_objects, frame)

            for prev_frame in self.prev_frames:
                self.object_tracker.track_objects(prev_frame)
            self.prev_frames.clear()

            cloud_tracking_result = self.object_tracker.track_objects(frame)
            current_cloud_detections = [detection for detection, _ in cloud_tracking_result]

            if current_cloud_detections:
                detections = fuse_edge_cloud_detections(edge_detections, current_cloud_detections, DetectionType.CLOUD)
                self.last_detections = detections
                return DetectionType.CLOUD, detections
            else:
                detections = fuse_edge_cloud_detections(self.last_detections, edge_detections, DetectionType.EDGE)
                self.last_detections = detections
                return DetectionType.EDGE, detections
        else:
            detections = fuse_edge_cloud_detections(self.last_detections, edge_detections, DetectionType.EDGE)
            self.last_detections = detections
            return DetectionType.EDGE, detections

    def record(self, frame: Frame):
        self.prev_frames.append(frame)

    def _cloud_detect_objects(self, frame: Frame):
        self.cloud_detection_done = False
        self.last_cloud_detections = []

        print("Cloud detection start")
        detections = self.cloud_server.detect_objects(frame)
        print("Cloud detection end")

        self.last_cloud_detections = detections
        self.cloud_detection_done = True
