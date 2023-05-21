from model import DetectionType


def color_det(det_type: DetectionType) -> tuple[int, int, int]:
    return (255, 0, 0) if det_type == DetectionType.EDGE \
        else (0, 0, 255) if det_type == DetectionType.CLOUD \
        else (0, 0, 0)


def color_tracked() -> tuple[int, int, int]:
    return 255, 180, 0


def color_annotation() -> tuple[int, int, int]:
    return 0, 255, 0
