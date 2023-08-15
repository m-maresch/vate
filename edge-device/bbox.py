def scale(bbox: list[float], scale_width: float, scale_height: float) -> list[int]:
    return [
        int(bbox[0] * scale_width),
        int(bbox[1] * scale_height),
        int(bbox[2] * scale_width),
        int(bbox[3] * scale_height),
    ]
