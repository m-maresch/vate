import cv2 as cv
import numpy as np
import requests

from model import Detection, DetectionType, Frame


def frame_change_detected(frame: Frame, prev_frame: Frame) -> bool:
    frame_gray = cv.cvtColor(frame.data, cv.COLOR_BGR2GRAY)
    prev_frame_gray = cv.cvtColor(prev_frame.data, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, flow=None,
                                       pyr_scale=0.5, levels=3, winsize=15,
                                       iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(mag) >= 0.1


def detect_objects(frame: Frame) -> tuple[DetectionType, list[Detection]]:
    encoded = cv.imencode(".jpg", frame.data)[1]
    file = {'file': (f'{frame.id}.jpg', encoded.tobytes(), 'image/jpeg')}
    data = {"id": f"{frame.id}"}
    response = requests.post("http://127.0.0.1:8000/detection", files=file, data=data, timeout=25)
    body = response.json()
    print(f"Got: {body}")
    det_type = DetectionType[body['type']]
    detections = body['detections']
    return det_type, [_to_detection(detection) for detection in detections]


def _to_detection(detection) -> Detection:
    bbox = detection['bbox']
    score = detection['score']
    category = detection['category']
    return Detection(category, score, bbox)
