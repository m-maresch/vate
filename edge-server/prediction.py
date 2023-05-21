import requests

from model import RawDetection, DetectionType


def get_predictions(url: str, frame: bytes, timeout: int, det_type: DetectionType) -> list[RawDetection]:
    response = requests.get(url, data=frame, timeout=timeout)
    body = response.json()
    print(f"Got: {body}")
    return [RawDetection(
        class_name=detection['class_name'],
        score=detection['score'],
        bbox=detection['bbox'],
        last_type=det_type
    ) for detection in body]
