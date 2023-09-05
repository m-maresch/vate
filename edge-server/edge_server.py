import time
import zmq.asyncio
from typing import List, Any

from model import Detection, DetectionResponse, RawDetection
from prediction import Predictor


class EdgeServer:
    context: Any
    socket: Any
    poll: Any

    predictor: Predictor

    def __init__(self, predictor: Predictor):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.poll = zmq.asyncio.Poller()
        self.predictor = predictor

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

        print(f"Identity: {identity}")

        frame = await self.socket.recv(flags=0, copy=False, track=False)

        start_detect = time.time()
        detections = self.predictor.get_predictions(frame)
        detection_response = _to_response(detections)
        print(f"Detect took: {time.time() - start_detect}")

        await self.socket.send_string(identity, zmq.SNDMORE)
        await self.socket.send_string(detection_response.json())


def _to_response(detections: List[RawDetection]) -> DetectionResponse:
    return DetectionResponse(
        detections=[Detection(
            bbox=detection.bbox,
            score=int(detection.score * 100),
            category=detection.class_name
        ) for detection in detections]
    )
