from typing import List


def scale(bbox: List[float], scale_width: float, scale_height: float) -> List[int]:
    return [
        int(bbox[0] * scale_width),
        int(bbox[1] * scale_height),
        int(bbox[2] * scale_width),
        int(bbox[3] * scale_height),
    ]
