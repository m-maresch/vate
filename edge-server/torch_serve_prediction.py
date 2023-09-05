import numpy as np
import requests
import time
from typing import List

from model import RawDetection


class TorchServePredictor:
    detection_model_url: str
    detection_timeout: int

    def __init__(self, detection_model_url: str, detection_timeout: int):
        self.detection_model_url = detection_model_url
        self.detection_timeout = detection_timeout

    def get_predictions(self, frame) -> List[RawDetection]:
        start = time.time()

        image = np.frombuffer(frame, dtype=np.uint8).tobytes()

        response = requests.get(self.detection_model_url, data=image, timeout=self.detection_timeout)
        body = response.json()
        print(f"Num detections: {len(body)}")

        end = time.time()
        print(f"Detection request took: {end - start}s")

        return [RawDetection(
            class_name=detection['class_name'],
            score=detection['score'],
            bbox=detection['bbox']
        ) for detection in body]
