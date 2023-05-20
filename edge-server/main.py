from fastapi import FastAPI, BackgroundTasks, UploadFile
from pycocotools.coco import maskUtils as mask
from scipy.optimize import linear_sum_assignment
import numpy as np
import requests
import random

app = FastAPI()

edge_detection_model_url = "http://127.0.0.1:9090/predictions/mobilenetv2_ssd_visdrone"
cloud_detection_model_url = "http://127.0.0.1:9093/predictions/faster_rcnn_visdrone"

last_detections = []
last_cloud_detections = []
in_progress = False


@app.get("/")
async def root():
    return {"status": "OK"}


@app.post("/detection")
async def detect(file: UploadFile, background_tasks: BackgroundTasks):
    frame = await file.read()
    frame_id = file.filename
    (det_type, detections) = _detect_object(frame, background_tasks)
    last_detections.clear()
    last_detections.extend(detections)
    json = _detections2json(detections)
    return {
        "frame_id": frame_id,
        "type": det_type,
        "detections": json
    }


def _detect_object(frame, background_tasks):
    if random.randint(0, 100) < 20 and not in_progress:
        background_tasks.add_task(_cloud_detect_objects, frame)

    edge_detections = _edge_detect_objects(frame)
    if last_cloud_detections:
        cloud_detections = last_cloud_detections.copy()
        last_cloud_detections.clear()
        return "cloud", _fuse_edge_cloud_detections(edge_detections, cloud_detections, "cloud")
    return "edge", _fuse_edge_cloud_detections(last_detections, edge_detections, "edge")


def _edge_detect_objects(frame):
    return _get_predictions(edge_detection_model_url, frame, 10)


def _cloud_detect_objects(frame):
    global in_progress
    in_progress = True
    print("Cloud detection start")
    detections = _get_predictions(cloud_detection_model_url, frame, 30)
    last_cloud_detections.clear()
    last_cloud_detections.extend(detections)
    print("Cloud detection end")
    in_progress = False


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


def _fuse_edge_cloud_detections(current_detections, new_detections, new_type):
    if not current_detections:
        return new_detections

    M = np.zeros((len(current_detections), len(new_detections)))
    for i, current in enumerate(current_detections):
        for j, new in enumerate(new_detections):
            iou = mask.iou([current['bbox']], [new['bbox']], [0])
            if iou >= 0.5:
                M[i, j] = iou
            else:
                M[i, j] = 0

    current_ind, new_ind = linear_sum_assignment(M, maximize=True)
    result = []

    for i, j in zip(current_ind, new_ind):
        if M[i, j] != 0:
            o = {}
            if new_type == "cloud":
                o['class_name'] = new_detections[j]['class_name']
                o['bbox'] = current_detections[i]['bbox']
            if new_type == "edge":
                o['class_name'] = current_detections[i]['class_name']
                o['bbox'] = new_detections[j]['bbox']
            o['score'] = new_detections[j]['score']
            o['score'] = o['score'] * 0.99
            o['last_type'] = new_type
            result.append(o)
        else:
            new_detections[j]['last_type'] = new_type
            result.append(new_detections[j])

    return result
