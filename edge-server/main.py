from fastapi import FastAPI, BackgroundTasks, UploadFile, Depends
from typing import List

from bbox import xyxy2xywh
from detection import ObjectDetector
from model import Detection, DetectionResponse, RawDetection
from services import get_object_detector

app = FastAPI()


@app.get("/")
async def root():
    return {"status": "OK"}


@app.post("/detection")
async def detect(file: UploadFile, background_tasks: BackgroundTasks,
                 object_detector: ObjectDetector = Depends(get_object_detector)) -> DetectionResponse:
    frame = await file.read()
    frame_id = int(file.filename.split(".")[0])
    (det_type, detections) = object_detector.detect_objects(frame, background_tasks)
    return DetectionResponse(
        frame_id=frame_id,
        type=det_type.name,
        detections=_convert_detections(detections)
    )


def _convert_detections(detections: List[RawDetection]) -> List[Detection]:
    return [Detection(
        bbox=list(map(int, xyxy2xywh(detection.bbox))),
        score=int(detection.score * 100),
        category=detection.class_name
    ) for detection in detections]
