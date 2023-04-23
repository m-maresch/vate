from fastapi import FastAPI, UploadFile
import requests
import random

app = FastAPI()

edge_detection_model_url = "http://127.0.0.1:9090/predictions/mobilenetv2_ssd_visdrone"
cloud_detection_model_url = "http://127.0.0.1:9093/predictions/faster_rcnn_visdrone"


@app.get("/")
async def root():
    return {"status": "OK"}


@app.post("/detection")
async def detect(file: UploadFile):
    frame = await file.read()
    frame_id = file.filename
    (det_type, detections) = _detect_object(frame)
    json = _detections2json(detections)
    return {
        "frame_id": frame_id,
        "type": det_type,
        "detections": json
    }


def _detect_object(frame):
    if random.randint(0, 100) > 10:
        return "edge", _edge_detect_objects(frame)
    else:
        return "cloud", _cloud_detect_objects(frame)


def _edge_detect_objects(frame):
    return _get_predictions(edge_detection_model_url, frame, 5)


def _cloud_detect_objects(frame):
    return _get_predictions(cloud_detection_model_url, frame, 20)


def _get_predictions(url, frame, timeout):
    response = requests.get(url, data=frame, timeout=timeout)
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
