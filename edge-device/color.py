from typing import Tuple

from model import DetectionType


def color_det(det_type: DetectionType) -> Tuple[int, int, int]:
    return (255, 0, 0) if det_type == DetectionType.EDGE \
        else (0, 0, 255) if det_type == DetectionType.CLOUD \
        else (0, 0, 0)


def color_tracked(det_type: DetectionType) -> Tuple[int, int, int]:
    return (255, 180, 0) if det_type == DetectionType.EDGE \
        else (0, 120, 255) if det_type == DetectionType.CLOUD \
        else (0, 0, 0)


def color_annotation() -> Tuple[int, int, int]:
    return 0, 255, 0
