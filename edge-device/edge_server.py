import cv2 as cv
import json
import numpy as np
import random
import zmq
from typing import Union, Tuple, List, Any

from model import Detection, DetectionType, Frame


class EdgeServer:
    context: Any
    socket: Any
    ipc: bool

    in_progress: bool

    def __init__(self, ipc: bool):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.ipc = ipc
        self.in_progress = False

    def connect(self):
        self.socket.setsockopt_string(zmq.IDENTITY, str(random.randint(0, 8000)))
        if self.ipc:
            self.socket.connect("ipc:///tmp/edge-server/0")
        else:
            self.socket.connect("tcp://127.0.0.1:8000")

    def detect_objects(self, frame: Frame, wait: bool) -> Tuple[Union[DetectionType, None], List[Detection]]:
        if not self.in_progress:
            metadata = {
                "frame_id": frame.id,
                "video": frame.video
            }
            self.socket.send_string(json.dumps(metadata), zmq.SNDMORE)

            encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 90]
            encoded = cv.imencode(".jpg", frame.resized_data, encode_param)[1]

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
            return None, []

        print("Receiving detections")
        self.in_progress = False
        response = self.socket.recv(zmq.NOBLOCK)

        body = json.loads(response)

        det_type = DetectionType[body['type']]
        detections = body['detections']

        return det_type, [_to_detection(detection) for detection in detections]


def _to_detection(detection) -> Detection:
    bbox = detection['bbox']
    score = detection['score']
    category = detection['category']
    return Detection(category, score, bbox)
