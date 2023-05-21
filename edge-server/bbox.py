def xyxy2xywh(xyxy_bbox: list[float]) -> list[float]:
    bbox = xyxy_bbox
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
