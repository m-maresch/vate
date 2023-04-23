from fastapi import FastAPI, UploadFile
import requests

app = FastAPI()


@app.get("/")
async def root():
    return {"status": "OK"}


@app.post("/detection")
async def detect(file: UploadFile):
    frame = await file.read()
    frame_id = file.filename
    detections = _edge_detect_objects(frame)
    json = _detections2json(detections)
    return {
        "frame_id": frame_id,
        "type": "edge",
        "detections": json
    }


def _edge_detect_objects(frame):
    response = requests.get("http://127.0.0.1:9090/predictions/mobilenetv2_ssd_visdrone", data=frame, timeout=5)
    body = response.json()
    print(f"Got: {body}")
    return body


def _detections2json(detections):
    json = []
    for detection in detections:
        class_name = detection['class_name']
        bbox = detection['bbox']
        score = detection['score']
        json.append({
            "bbox": list(map(int, _xyxy2xywh(bbox))),
            "score": int(score * 100),
            "category": class_name
        })
    return json


def _xyxy2xywh(xyxy_bbox):
    bbox = xyxy_bbox
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
