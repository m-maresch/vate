import json
import time
import zmq.asyncio
from typing import List, Any

from bbox import xyxy2xywh
from detection import ObjectDetector
from model import Detection, DetectionResponse, RawDetection, DetectionType


class EdgeServer:
    context: Any
    socket: Any
    poll: Any

    object_detector: ObjectDetector

    def __init__(self, object_detector: ObjectDetector):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.poll = zmq.asyncio.Poller()
        self.object_detector = object_detector

    def listen(self, ipc: bool):
        if ipc:
            self.socket.bind("ipc:///tmp/edge-server/0")
        else:
            self.socket.bind("tcp://*:8000")

        self.poll.register(self.socket, zmq.POLLIN)

    async def handle_requests(self):
        while True:
            sockets = dict(await self.poll.poll(1000))
            if sockets:
                await self._handle_request()

    async def _handle_request(self):
        identity = await self.socket.recv_string()

        metadata = await self.socket.recv_string()
        metadata_json = json.loads(metadata)
        frame_id = metadata_json['frame_id']
        video = metadata_json['video']

        print(f"Identity: {identity}, frame id: {frame_id}, video: {video}")

        frame = await self.socket.recv(flags=0, copy=False, track=False)

        start_detect = time.time()
        (det_type, detections) = await self.object_detector.detect_objects(video, frame)
        detection_response = _to_response(frame_id, det_type, detections)
        print(f"Detect took: {time.time() - start_detect}")

        await self.socket.send_string(identity, zmq.SNDMORE)
        await self.socket.send_string(detection_response.json())


def _to_response(frame_id: int, det_type: DetectionType, detections: List[RawDetection]) -> DetectionResponse:
    return DetectionResponse(
        frame_id=frame_id,
        type=det_type.name,
        detections=[Detection(
            bbox=xyxy2xywh(detection.bbox),
            score=int(detection.score * 100),
            category=detection.class_name
        ) for detection in detections]
    )
