from concurrent.futures import Executor, Future, ThreadPoolExecutor
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
    max_fps: int
    executor: Executor

    prev_frames: List[Frame]
    last_detections: List[Detection]
    cloud_detection: Future[List[Detection]]

    def __init__(self, edge_server: EdgeServer, cloud_server: CloudServer, object_tracker: MultiObjectTracker,
                 max_fps: int):
        self.edge_server = edge_server
        self.cloud_server = cloud_server
        self.object_tracker = object_tracker
        self.max_fps = max_fps
        self.executor = ThreadPoolExecutor()
        self.prev_frames = []
        self.last_detections = []
        self.cloud_detection = Future()
        self.cloud_detection.set_result([])

    def detect_objects(self, frame: Frame, wait: bool) -> Tuple[Union[DetectionType, None], List[Detection]]:
        edge_detections = self.edge_server.detect_objects(frame, wait)

        if not edge_detections:
            return None, []

        if self.cloud_detection.done():
            cloud_detections = self.cloud_detection.result()
            self.cloud_detection = self.executor.submit(self._cloud_detect_objects, frame)

            self.object_tracker.reset_objects()
            for detection in cloud_detections:
                self.object_tracker.add_object(frame, detection, DetectionType.CLOUD)

            cloud_tracking_result, _ = self.object_tracker.track_objects_until_current(self.prev_frames, frame)
            current_cloud_detections = [detection for detection, _ in cloud_tracking_result]
            self.prev_frames.clear()

            if current_cloud_detections:
                detections = fuse_edge_cloud_detections(edge_detections, current_cloud_detections, DetectionType.CLOUD)
                self.last_detections = detections
                return DetectionType.CLOUD, detections
            else:
                detections = fuse_edge_cloud_detections(self.last_detections, edge_detections, DetectionType.EDGE)
                self.last_detections = detections
                return DetectionType.EDGE, detections
        else:
            detections = fuse_edge_cloud_detections([], edge_detections, DetectionType.EDGE)
            self.last_detections = detections
            return DetectionType.EDGE, detections

    def record(self, frame: Frame):
        self.prev_frames.append(frame)

        if len(self.prev_frames) > self.max_fps:
            self.prev_frames = self.prev_frames[-self.max_fps:]

    def reset(self):
        self.object_tracker.reset_objects()
        self.prev_frames.clear()
        self.last_detections.clear()

    def _cloud_detect_objects(self, frame: Frame) -> list[Detection]:
        print("Cloud detection start")
        cloud_detections = self.cloud_server.detect_objects(frame)
        print(f"Cloud detection end")

        return cloud_detections
