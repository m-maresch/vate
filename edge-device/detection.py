import cv2 as cv
import numpy as np
import requests
import time

from model import Detection, DetectionType, Frame


def frame_change_detected(frame: Frame, prev_frame: Frame) -> bool:
    return np.bitwise_xor(frame.resized_data, prev_frame.resized_data).any()


def detect_objects(frame: Frame) -> tuple[DetectionType, list[Detection]]:
    start = time.time()

    encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 90]
    encoded = cv.imencode(".jpg", frame.resized_data, encode_param)[1]
    file = {'file': (f'{frame.id}.jpg', encoded.tobytes(), 'image/jpeg')}
    data = {"id": f"{frame.id}"}
    response = requests.post("http://127.0.0.1:8000/detection", files=file, data=data, timeout=25)
    body = response.json()
    print(f"Got: {body}")
    det_type = DetectionType[body['type']]
    detections = body['detections']

    end = time.time()
    print(f"Took: {end - start}s")

    return det_type, [_to_detection(detection) for detection in detections]


def _to_detection(detection) -> Detection:
    bbox = detection['bbox']
    score = detection['score']
    category = detection['category']
    return Detection(category, score, bbox)
