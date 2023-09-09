from multiprocessing import Queue

import cv2 as cv
from typing import List, Tuple

from model import TrackerRecord, Frame, Detection, DetectionType


class MultiObjectTracker:
    trackers: List[TrackerRecord]
    min_score: int

    def __init__(self, min_score: int):
        self.trackers = []
        self.min_score = min_score

    def reset_objects(self):
        self.trackers = []

    def add_object(self, frame: Frame, detection: Detection, det_type: DetectionType):
        if detection.score < self.min_score:
            self.trackers.append(
                TrackerRecord(None, detection.bbox, detection.score, detection.category, det_type)
            )
            return

        tracker = cv.legacy.TrackerKCF_create()
        try:
            tracker.init(frame.resized_data, detection.bbox)
            self.trackers.append(
                TrackerRecord(tracker, detection.bbox, detection.score, detection.category, det_type)
            )
        except cv.error as e:
            print(f"Failed to init a tracker: {e}")

    def track_objects_until_current(self, prev_frames: List[Frame],
                                    current_frame: Frame) -> Tuple[List[Tuple[Detection, DetectionType]], int]:
        tracking_result = []
        tracker_failures = 0

        frames = prev_frames.copy()
        frames.append(current_frame)
        for frame in frames:
            tracking_result = self.track_objects(frame)
            if not tracking_result:
                tracker_failures += 1

        return tracking_result, tracker_failures

    def track_objects(self, frame: Frame) -> List[Tuple[Detection, DetectionType]]:
        result = []
        for tracker in self.trackers:
            if tracker.raw_tracker is None:
                new_score = int(0.9 * tracker.det_score)
                tracker.det_score = new_score
                result.append(
                    (Detection(tracker.det_category, new_score, list(map(int, tracker.det_bbox))), tracker.det_type)
                )
            else:
                ok, bbox = tracker.raw_tracker.update(frame.resized_data)
                if ok:
                    result.append(
                        (Detection(tracker.det_category, tracker.det_score, list(map(int, bbox))), tracker.det_type)
                    )

        return result


def track_objects_until_current_worker(input_queue: Queue, output_queue: Queue):
    while True:
        (detections, prev_frames, current_frame, min_score) = input_queue.get()

        if not detections:
            output_queue.put([])

        object_tracker = MultiObjectTracker(min_score=min_score)
        for detection in detections:
            object_tracker.add_object(current_frame, detection, DetectionType.CLOUD)

        tracking_result, _ = object_tracker.track_objects_until_current(prev_frames, current_frame)
        output_queue.put([detection for detection, _ in tracking_result])
