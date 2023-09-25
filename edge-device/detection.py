import requests
import zmq

from concurrent.futures import Executor, Future, ThreadPoolExecutor
from multiprocessing import Process, Event
from queue import Full
from typing import List, Union, Any

from bbox import scale
from cloud_server import CloudServer
from edge_server import EdgeServer
from fusion import fuse_edge_cloud_detections
from model import Detection, DetectionType, Frame, Dimensions, DetectionsWithTypes
from process import track_objects_until_current_worker


class EdgeCloudObjectDetector:
    edge_server: EdgeServer
    cloud_server: CloudServer
    executor: Executor
    dimensions: Dimensions
    cloud_tracking_process: Union[Process, None]
    cloud_tracking_context: Any
    cloud_tracking_socket: Any
    cloud_tracking_stop_event: Event
    cloud_tracking_min_score: int
    cloud_tracking_stride: int
    max_fps: int

    frames_until_current: List[Frame]
    cloud_detection: "Future[List[Detection]]"

    def __init__(self, edge_server: EdgeServer, cloud_server: CloudServer, dimensions: Dimensions,
                 cloud_tracking_min_score: int, cloud_tracking_stride: int, max_fps: int):
        self.edge_server = edge_server
        self.cloud_server = cloud_server
        self.executor = ThreadPoolExecutor()
        self.dimensions = dimensions
        self.cloud_tracking_process = None
        self.cloud_tracking_context = zmq.Context()
        self.cloud_tracking_socket = self.cloud_tracking_context.socket(zmq.DEALER)
        self.cloud_tracking_stop_event = Event()
        self.cloud_tracking_min_score = cloud_tracking_min_score
        self.cloud_tracking_stride = cloud_tracking_stride
        self.max_fps = max_fps
        self.frames_until_current = []
        self.cloud_detection = Future()
        self.cloud_detection.set_result([])

    def start_cloud_tracking(self):
        self.cloud_tracking_process = Process(
            target=track_objects_until_current_worker,
            args=(self.cloud_tracking_stop_event,)
        )
        self.cloud_tracking_process.start()

        self.cloud_tracking_socket.setsockopt_string(zmq.IDENTITY, "edge-device")
        self.cloud_tracking_socket.connect("ipc:///tmp/edge-device/0")

    def stop_cloud_tracking(self):
        self.cloud_tracking_stop_event.set()
        self.cloud_tracking_process.join()

    def get_edge_object_detections_sync(self,
                                        frame: Frame,
                                        current_detections: DetectionsWithTypes) -> Union[DetectionsWithTypes, None]:
        self.request_edge_object_detections_async(frame)
        return self.get_edge_object_detections_async(current_detections, timeout=60000)

    def request_edge_object_detections_async(self, frame: Frame):
        self.edge_server.send_frame(frame)

    def get_edge_object_detections_async(self,
                                         current_detections: DetectionsWithTypes,
                                         timeout: int = 3) -> Union[DetectionsWithTypes, None]:
        edge_detections = self.edge_server.receive_object_detections(timeout)

        if not edge_detections:
            return None

        current_detections = [detection for detection, det_type in current_detections
                              if det_type == DetectionType.CLOUD]
        return fuse_edge_cloud_detections(current_detections, edge_detections, DetectionType.EDGE)

    def get_cloud_object_detections_async(self,
                                          frame: Frame,
                                          current_detections: DetectionsWithTypes) -> Union[DetectionsWithTypes, None]:
        tracked_cloud_detections = []
        if self.cloud_tracking_socket.poll(1, zmq.POLLIN):
            tracked_cloud_detections = self.cloud_tracking_socket.recv_pyobj(zmq.NOBLOCK)

        if self.cloud_detection.done():
            cloud_detections = self.cloud_detection.result()
            scaled_cloud_detections = self._scale_cloud_to_edge(cloud_detections)
            self.cloud_detection = self.executor.submit(self._cloud_detect_objects, frame)
            self.executor.submit(self._add_cloud_tracking_task, scaled_cloud_detections, frame)

        if not tracked_cloud_detections:
            return None

        current_detections = [detection for detection, det_type in current_detections
                              if det_type == DetectionType.EDGE]
        return fuse_edge_cloud_detections(current_detections, tracked_cloud_detections, DetectionType.CLOUD)

    def record(self, frame: Frame):
        self.frames_until_current.append(frame)

        if len(self.frames_until_current) > self.max_fps:
            self.frames_until_current = self.frames_until_current[-self.max_fps:]

    def reset(self):
        self.frames_until_current.clear()

    def _scale_cloud_to_edge(self, detections: List[Detection]) -> List[Detection]:
        return [
            Detection(
                detection.category,
                detection.score,
                scale(
                    detection.bbox,
                    self.dimensions.edge_processing_width / self.dimensions.cloud_processing_width,
                    self.dimensions.edge_processing_height / self.dimensions.cloud_processing_height
                )
            )
            for detection in detections
        ]

    def _cloud_detect_objects(self, frame: Frame) -> List[Detection]:
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
            obj = dict(
                detections=cloud_detections,
                frames_until_current=self._filter_frames_until_current(),
                current_frame=frame,
                min_score=self.cloud_tracking_min_score
            )
            self.cloud_tracking_socket.send_pyobj(obj, protocol=-1)
        except Full as e:
            print("Cloud tracking failure:", e)

        self.frames_until_current.clear()

    def _filter_frames_until_current(self) -> List[Frame]:
        return [frame for idx, frame in enumerate(self.frames_until_current)
                if idx % self.cloud_tracking_stride == 0]
