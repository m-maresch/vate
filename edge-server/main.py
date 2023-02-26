from fastapi import FastAPI, UploadFile
from mmdet.apis import init_detector, inference_detector
import mmcv

config_file = '../detection-models/mobilenetv2_ssd.py'
checkpoint_file = '../detection-models/work_dirs/mobilenetv2_ssd/latest.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')
categories = [
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

app = FastAPI()


@app.get("/")
async def root():
    return {"status": "OK"}


@app.post("/detection")
async def detect(file: UploadFile):
    frame = await file.read()
    image_id = file.filename
    image = mmcv.imfrombytes(frame)
    detections = inference_detector(model, image)
    json = _detections2json(detections)
    return {
        "image_id": image_id,
        "detections": json
    }


def _detections2json(detections):
    json = []
    for label in range(len(detections)):
        bboxes = detections[label]
        for i in range(bboxes.shape[0]):
            json.append({
                "bbox": list(map(int, _xyxy2xywh(bboxes[i]))),
                "score": int(bboxes[i][4] * 100),
                "category": _category_name(label)
            })
    return json


def _xyxy2xywh(xyxy_bbox):
    bbox = xyxy_bbox.tolist()
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]


def _category_name(category_id):
    category = next(filter(lambda cat: cat["id"] == category_id, categories))
    return category["name"]
