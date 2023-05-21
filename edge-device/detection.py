import cv2 as cv
import requests

from model import Detection, DetectionType, Frame


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
