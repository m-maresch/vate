from typing import Union

import cv2 as cv
import time

from annotation import annotations_available, load_annotations
from category import to_category_id
from detection import detect_objects
from display import display_detection, display_annotation, display_fps
from evaluation import evaluate_detections
from frame import get_frames
from model import DetectionView, ImageList, AnnotationsByImage
from track import MultiObjectTracker


class EdgeDevice:
    video: Union[str, None]
    annotations_path: Union[str, None]
    detection_rate: int
    object_tracker: MultiObjectTracker

    frame_count: int
    prev_frame_at: float

    all_detections: list[DetectionView]
    detections_to_display: list[DetectionView]

    tracker_failures: int

    def __init__(self, video: Union[str, None], annotations_path: Union[str, None], detection_rate: int,
                 object_tracker: MultiObjectTracker):
        self.video = video
        self.annotations_path = annotations_path
        self.detection_rate = detection_rate
        self.object_tracker = object_tracker

        self.frame_count = 0
        self.prev_frame_at = 0

        self.all_detections = []
        self.detections_to_display = []

        self.tracker_failures = 0

    def start(self):
        images: ImageList = []
        annotations: AnnotationsByImage = dict()
        if annotations_available(self.video, self.annotations_path):
            (images, annotations) = load_annotations(self.video, self.annotations_path)

        frames = get_frames(self.video, images)
        while True:
            frame = next(frames)
            if frame.id == -1:
                break

            if self.frame_count % self.detection_rate == 0:
                self.detections_to_display = []
                self.object_tracker.reset_objects()
                (det_type, detections) = detect_objects(frame)
                for detection in detections:
                    (x, y, w, h) = detection.bbox
                    category_id = to_category_id(detection.category)
                    det_view = DetectionView(frame.id, x, y, w, h, detection.score, category_id, det_type,
                                             tracked=False)
                    self.all_detections.append(det_view)
                    self.detections_to_display.append(det_view)

                    self.object_tracker.add_object(frame, det_view)
            else:
                tracking_result = self.object_tracker.track_objects(frame)
                if not tracking_result:
                    self.tracker_failures += 1

                self.all_detections.extend(tracking_result)
                self.detections_to_display.extend(tracking_result)

            frame_at = time.time()
            fps = int(1 / (frame_at - self.prev_frame_at))
            self.prev_frame_at = frame_at

            self.frame_count += 1

            for received_detection in self.detections_to_display:
                display_detection(frame.data, received_detection)

            if annotations_available(self.video, self.annotations_path):
                for annotation in annotations[frame.id]:
                    display_annotation(frame.data, annotation)

            display_fps(frame.data, fps)

            cv.imshow('frame', frame.data)
            if cv.waitKey(1) == ord('q'):
                break

        if annotations_available(self.video, self.annotations_path):
            evaluate_detections(self.all_detections, self.annotations_path)

        cv.destroyAllWindows()
        print(f"{self.tracker_failures} tracker failures")
