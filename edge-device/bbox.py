from typing import List


def xyxy2xywh(xyxy_bbox: List[float]) -> List[float]:
    bbox = xyxy_bbox
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]


def scale(bbox: List[float], scale_width: float, scale_height: float) -> List[int]:
    return [
        int(bbox[0] * scale_width),
        int(bbox[1] * scale_height),
        int(bbox[2] * scale_width),
        int(bbox[3] * scale_height),
    ]
