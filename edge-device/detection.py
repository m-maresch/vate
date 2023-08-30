import cv2 as cv
import json
import numpy as np
import random
import zmq
from typing import Union, Tuple, List

from model import Detection, DetectionType, Frame

context = zmq.Context()
socket = context.socket(zmq.DEALER)
socket.setsockopt_string(zmq.IDENTITY, str(random.randint(0, 8000)))
socket.connect("tcp://127.0.0.1:8000")

in_progress = False


def frame_change_detected(frame: Frame, prev_frame: Frame) -> bool:
    return np.bitwise_xor(frame.resized_data, prev_frame.resized_data).any()


def detect_objects(frame: Frame, wait: bool) -> Tuple[Union[DetectionType, None], List[Detection]]:
    global in_progress
    if not in_progress:
        metadata = {
            "frame_id": frame.id,
            "video": frame.video
        }
        socket.send_string(json.dumps(metadata), zmq.SNDMORE)

        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 90]
        encoded = cv.imencode(".jpg", frame.resized_data, encode_param)[1]

        print("Sending frame")

        in_progress = True
        if encoded.flags['C_CONTIGUOUS']:
            socket.send(encoded, 0, copy=False, track=False)
        else:
            encoded = np.ascontiguousarray(encoded)
            socket.send(encoded, 0, copy=False, track=False)

    timeout = 10
    if wait:
        timeout = 10000

    if not socket.poll(timeout, zmq.POLLIN):
        return None, []

    print("Receiving detections")
    in_progress = False
    response = socket.recv(zmq.NOBLOCK)

    body = json.loads(response)

    det_type = DetectionType[body['type']]
    detections = body['detections']

    return det_type, [_to_detection(detection) for detection in detections]


def _to_detection(detection) -> Detection:
    bbox = detection['bbox']
    score = detection['score']
    category = detection['category']
    return Detection(category, score, bbox)
