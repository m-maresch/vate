import cv2 as cv
import numpy as np
import time
from typing import List, Any

from jetson_inference import detectNet
from jetson_utils import (cudaFromNumpy, cudaAllocMapped, cudaConvertColor)

from prediction import to_category_name
from model import RawDetection, DetectionType


class JetsonPredictor:
    model: Any
    input_width: int
    input_height: int

    def __init__(self, model: str, labels: str, threshold: float, input_width: int, input_height: int):
        self.model = detectNet(model=model, labels=labels, input_blob="input_0",
                               output_cvg="scores", output_bbox="boxes", threshold=threshold)
        self.input_width = input_width
        self.input_height = input_height

    def get_predictions(self, frame) -> List[RawDetection]:
        detections = self._run_inference(frame)

        return [RawDetection(
            class_name=to_category_name(detection.ClassID - 1),
            score=detection.Confidence,
            bbox=[detection.Left, detection.Top, detection.Right, detection.Bottom],
            last_type=DetectionType.EDGE
        ) for detection in detections]

    def _run_inference(self, image):
        start = time.time()

        image = np.frombuffer(image, dtype=np.uint8)
        image = cv.imdecode(image, cv.IMREAD_COLOR)

        original_height, original_width, _ = image.shape
        image = cv.resize(image, (self.input_width, self.input_height), interpolation=cv.INTER_LINEAR)

        cv.imshow('det frame', image)
        if cv.waitKey(1) == ord('q'):
            pass

        bgr_image = cudaFromNumpy(image, isBGR=True)
        rgb_image = cudaAllocMapped(width=bgr_image.width, height=bgr_image.height, format='rgb8')
        cudaConvertColor(bgr_image, rgb_image)

        detections = self.model.Detect(rgb_image, overlay="box,labels,conf")

        print(f"Num detections: {len(detections)}")

        scale_width = original_width / self.input_width
        scale_height = original_height / self.input_height
        for detection in detections:
            detection.Left *= scale_width
            detection.Right *= scale_width
            detection.Top *= scale_height
            detection.Bottom *= scale_height

        end = time.time()
        print(f"Inference took: {end - start}s")

        return detections
