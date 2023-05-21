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


def to_category_id(category_name: str) -> int:
    category = next(filter(lambda cat: cat["name"] == category_name, CATEGORIES))
    return category["id"]


def to_category_name(category_id: int) -> str:
    category = next(filter(lambda cat: cat["id"] == category_id, CATEGORIES))
    return category["name"]
