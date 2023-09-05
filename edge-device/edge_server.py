import cv2 as cv
import json
import numpy as np
import random
import zmq
from typing import List, Any

from bbox import xyxy2xywh, scale
from model import Detection, Frame


class EdgeServer:
    context: Any
    socket: Any
    ipc: bool
    frame_processing_width: int
    frame_processing_height: int
    model_input_size: int

    in_progress: bool

    def __init__(self, ipc: bool, frame_processing_width: int, frame_processing_height: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.ipc = ipc
        self.frame_processing_width = frame_processing_width
        self.frame_processing_height = frame_processing_height
        self.model_input_size = 512
        self.in_progress = False

    def connect(self):
        self.socket.setsockopt_string(zmq.IDENTITY, str(random.randint(0, 8000)))
        if self.ipc:
            self.socket.connect("ipc:///tmp/edge-server/0")
        else:
            self.socket.connect("tcp://127.0.0.1:8000")

    def detect_objects(self, frame: Frame, wait: bool) -> List[Detection]:
        if not self.in_progress:
            resized_data = cv.resize(frame.resized_data, (self.model_input_size, self.model_input_size),
                                     interpolation=cv.INTER_LINEAR)

            encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 90]
            encoded = cv.imencode(".jpg", resized_data, encode_param)[1]

            print("Sending frame")

            self.in_progress = True
            if encoded.flags['C_CONTIGUOUS']:
                self.socket.send(encoded, 0, copy=False, track=False)
            else:
                encoded = np.ascontiguousarray(encoded)
                self.socket.send(encoded, 0, copy=False, track=False)

        timeout = 10
        if wait:
            timeout = 60000

        if not self.socket.poll(timeout, zmq.POLLIN):
            return []

        print("Receiving detections")
        self.in_progress = False
        response = self.socket.recv(zmq.NOBLOCK)

        body = json.loads(response)

        detections = body['detections']

        return [self._to_detection(detection) for detection in detections]

    def _to_detection(self, detection) -> Detection:
        bbox = xyxy2xywh(scale(
            detection['bbox'],
            self.frame_processing_width / self.model_input_size,
            self.frame_processing_height / self.model_input_size)
        )
        score = detection['score']
        category = detection['category']
        return Detection(category, score, bbox)
