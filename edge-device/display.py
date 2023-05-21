import cv2 as cv

from category import to_category_name
from color import color_det, color_tracked, color_annotation
from model import DetectionView, AnnotationView

FONT = cv.FONT_HERSHEY_SIMPLEX


def display_detection(frame, detection: DetectionView):
    (x, y, w, h) = [detection.x, detection.y, detection.w, detection.h]
    if not detection.tracked:
        category_name = to_category_name(detection.category)
        color = color_det(detection.type)
        cv.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv.putText(frame, f"{category_name} {detection.score}%", (x, y - 10), FONT, 0.5, color, 2)
    else:
        color = color_tracked()
        cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def display_annotation(frame, annotation: AnnotationView):
    (x, y, w, h) = [annotation.x, annotation.y, annotation.w, annotation.h]
    category_name = to_category_name(annotation.category)
    color = color_annotation()
    cv.rectangle(frame, (x, y), (x + w, y + h), color, 1)
    cv.putText(frame, f"{category_name}", (x, y - 10), FONT, 0.5, color, 1)


def display_fps(frame, fps: int):
    cv.putText(frame, f"{fps} FPS", (50, 50), FONT, 0.8, (0, 255, 0), 2)
