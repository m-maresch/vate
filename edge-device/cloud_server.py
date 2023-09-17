import cv2 as cv
import requests
import time
from typing import List

from bbox import xyxy2xywh
from model import Detection, Frame


class CloudServer:
    detection_model_url: str
    detection_timeout: int

    def __init__(self, detection_model_url: str, detection_timeout: int):
        self.detection_model_url = detection_model_url
        self.detection_timeout = detection_timeout

    def detect_objects(self, frame: Frame) -> List[Detection]:
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 90]
        encoded = cv.imencode(".jpg", frame.cloud_data, encode_param)[1]

        start = time.time()

        response = requests.get(self.detection_model_url, data=encoded.tobytes(), timeout=self.detection_timeout)
        body = response.json()
        print(f"Cloud num detections: {len(body)}")

        end = time.time()
        print(f"Cloud detection request took: {end - start}s")

        return [Detection(
            category=detection['class_name'],
            score=int(detection['score'] * 100),
            bbox=xyxy2xywh(detection['bbox'])
        ) for detection in body]
