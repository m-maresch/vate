from fastapi import FastAPI, UploadFile
from mmdet.apis import init_detector, inference_detector
import mmcv

config_file = '../detection-models/mobilenetv2_ssd.py'
checkpoint_file = '../detection-models/checkpoints/mobilenetv2_ssd_2000_epochs_0_frozen.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')

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
                "category": label
            })
    return json


def _xyxy2xywh(xyxy_bbox):
    bbox = xyxy_bbox.tolist()
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
