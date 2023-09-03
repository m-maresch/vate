from typing import List, Protocol

from model import RawDetection

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


class Predictor(Protocol):
    def get_predictions(self, frame) -> List[RawDetection]:
        ...


def to_category_name(category_id: int) -> str:
    category = next(filter(lambda cat: cat["id"] == category_id, CATEGORIES))
    return category["name"]
