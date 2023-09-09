import requests

from concurrent.futures import Executor, Future, ThreadPoolExecutor
from multiprocessing import Process, Queue
from queue import Empty, Full
from typing import List, Tuple, Union

from cloud_server import CloudServer
from edge_server import EdgeServer
from fusion import fuse_edge_cloud_detections
from model import Detection, DetectionType, Frame
from track import track_objects_until_current_worker


class EdgeCloudObjectDetector:
    edge_server: EdgeServer
    cloud_server: CloudServer
    cloud_detection_executor: Executor
    cloud_tracking_input_queue: Queue
    cloud_tracking_output_queue: Queue
    cloud_tracking_process: Union[Process, None]
    cloud_tracking_min_score: int
    max_fps: int

    frames_until_current: List[Frame]
    last_detections: List[Detection]
    cloud_detection: Future[List[Detection]]

    def __init__(self, edge_server: EdgeServer, cloud_server: CloudServer, cloud_tracking_min_score: int, max_fps: int):
        self.edge_server = edge_server
        self.cloud_server = cloud_server
        self.cloud_detection_executor = ThreadPoolExecutor()
        self.cloud_tracking_input_queue = Queue(maxsize=1)
        self.cloud_tracking_output_queue = Queue(maxsize=1)
        self.cloud_tracking_process = None
        self.cloud_tracking_min_score = cloud_tracking_min_score
        self.max_fps = max_fps
        self.frames_until_current = []
        self.last_detections = []
        self.cloud_detection = Future()
        self.cloud_detection.set_result([])

    def start_cloud_tracking_process(self):
        self.cloud_tracking_process = Process(
            target=track_objects_until_current_worker,
            args=(self.cloud_tracking_input_queue, self.cloud_tracking_output_queue)
        )
        self.cloud_tracking_process.start()

    def stop_cloud_tracking_process(self):
        self.cloud_tracking_process.terminate()
        self.cloud_tracking_process.close()

    def detect_objects(self, frame: Frame, wait: bool) -> Tuple[Union[DetectionType, None], List[Detection]]:
        edge_detections = self.edge_server.detect_objects(frame, wait)

        if not edge_detections:
            return None, []

        if self.cloud_detection.done():
            cloud_detections = self.cloud_detection.result()
            self.cloud_detection = self.cloud_detection_executor.submit(self._cloud_detect_objects, frame)
            self._add_cloud_tracking_task(cloud_detections, frame)

        if not self.cloud_tracking_output_queue.empty():
            current_cloud_detections = self.cloud_tracking_output_queue.get_nowait()
            if current_cloud_detections:
                detections = fuse_edge_cloud_detections(edge_detections, current_cloud_detections,
                                                        DetectionType.CLOUD)
                self.last_detections = detections
                return DetectionType.CLOUD, detections
            else:
                detections = fuse_edge_cloud_detections([], edge_detections, DetectionType.EDGE)
                self.last_detections = detections
                return DetectionType.EDGE, detections
        else:
            detections = fuse_edge_cloud_detections([], edge_detections, DetectionType.EDGE)
            self.last_detections = detections
            return DetectionType.EDGE, detections

    def record(self, frame: Frame):
        self.frames_until_current.append(frame)

        if len(self.frames_until_current) > self.max_fps:
            self.frames_until_current = self.frames_until_current[-self.max_fps:]

    def reset(self):
        self.frames_until_current.clear()
        self.last_detections.clear()

    def _cloud_detect_objects(self, frame: Frame) -> list[Detection]:
        try:
            print("Cloud detection start")
            cloud_detections = self.cloud_server.detect_objects(frame)
            print(f"Cloud detection end")
            return cloud_detections
        except requests.exceptions.RequestException as e:
            print("Cloud detection failure:", e)
            return []

    def _add_cloud_tracking_task(self, cloud_detections: List[Detection], frame: Frame):
        if not self.frames_until_current:
            return

        try:
            while not self.cloud_tracking_input_queue.empty():
                self.cloud_tracking_input_queue.get_nowait()
        except Empty:
            pass

        try:
            self.cloud_tracking_input_queue.put_nowait(
                (cloud_detections, self.frames_until_current.copy(), frame, self.cloud_tracking_min_score)
            )
        except Full as e:
            print("Cloud tracking failed:", e)

        self.frames_until_current.clear()
