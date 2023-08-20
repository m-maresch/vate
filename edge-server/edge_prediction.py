import cv2 as cv
import numpy as np
import time
from typing import List

from jetson_inference import detectNet
from jetson_utils import (cudaFromNumpy, cudaAllocMapped, cudaConvertColor)

from model import RawDetection, DetectionType

CATEGORIES = [
    {"id": 0, "name": "pedestrian"},
    {"id": 1, "name": "people"},
    {"id": 2, "name": "bicycle"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "van"},
    {"id": 5, "name": "truck"},
    {"id": 6, "name": "tricycle"},
    {"id": 7, "name": "awning-tricycle"},
    {"id": 8, "name": "bus"},
    {"id": 9, "name": "motor"}
]

MODEL = detectNet(model="mb2-ssd-lite.onnx", labels="onnxlabels.txt", input_blob="input_0", output_cvg="scores",
                  output_bbox="boxes", threshold=0.35)
INPUT_WIDTH = 300
INPUT_HEIGHT = 300


def get_edge_predictions(frame) -> List[RawDetection]:
    detections = _run_inference(frame)

    return [RawDetection(
        class_name=_to_category_name(detection.ClassID - 1),
        score=detection.Confidence,
        bbox=[detection.Left, detection.Top, detection.Right, detection.Bottom],
        last_type=DetectionType.EDGE
    ) for detection in detections]


def _run_inference(image):
    start = time.time()

    image = np.frombuffer(image, dtype=np.uint8)
    image = cv.imdecode(image, cv.IMREAD_COLOR)

    original_height, original_width, _ = image.shape
    image = cv.resize(image, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv.INTER_LINEAR)

    bgr_image = cudaFromNumpy(image, isBGR=True)
    rgb_image = cudaAllocMapped(width=bgr_image.width, height=bgr_image.height, format='rgb8')
    cudaConvertColor(bgr_image, rgb_image)

    detections = MODEL.Detect(rgb_image, overlay="box,labels,conf")

    scale_width = original_width / INPUT_WIDTH
    scale_height = original_height / INPUT_HEIGHT
    for detection in detections:
        detection.Left *= scale_width
        detection.Right *= scale_width
        detection.Top *= scale_height
        detection.Bottom *= scale_height

    end = time.time()
    print(f"Inference took: {end - start}s")

    return detections


def _to_category_name(category_id: int) -> str:
    category = next(filter(lambda cat: cat["id"] == category_id, CATEGORIES))
    return category["name"]
