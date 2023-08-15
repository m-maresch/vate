import requests
import time
from typing import List

from model import RawDetection, DetectionType

CLOUD_DETECTION_MODEL_URL: str = "http://127.0.0.1:9093/predictions/faster_rcnn_visdrone"
CLOUD_DETECTION_TIMEOUT: int = 30


def get_cloud_predictions(frame: bytes) -> List[RawDetection]:
    start = time.time()

    response = requests.get(CLOUD_DETECTION_MODEL_URL, data=frame, timeout=CLOUD_DETECTION_TIMEOUT)
    body = response.json()
    print(f"Got: {body}")

    end = time.time()
    print(f"Request took: {end - start}s")

    return [RawDetection(
        class_name=detection['class_name'],
        score=detection['score'],
        bbox=detection['bbox'],
        last_type=DetectionType.CLOUD
    ) for detection in body]
