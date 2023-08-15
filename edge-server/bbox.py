from typing import List


def xyxy2xywh(xyxy_bbox: List[float]) -> List[float]:
    bbox = xyxy_bbox
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
