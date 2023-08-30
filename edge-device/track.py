import cv2 as cv
from typing import List, Tuple

from model import TrackerRecord, Frame, Detection, DetectionType


class MultiObjectTracker:
    trackers: List[TrackerRecord]

    def __init__(self):
        self.trackers = []

    def reset_objects(self):
        self.trackers = []

    def add_object(self, frame: Frame, detection: Detection, det_type: DetectionType):
        tracker = cv.legacy.TrackerKCF_create()
        try:
            tracker.init(frame.resized_data, detection.bbox)
            self.trackers.append(
                TrackerRecord(tracker, detection.score, detection.category, det_type)
            )
        except cv.error as e:
            print(f"Failed to init a tracker: {e}")

    def track_objects(self, frame: Frame) -> List[Tuple[Detection, DetectionType]]:
        result = []
        for tracker in self.trackers:
            ok, bbox = tracker.raw_tracker.update(frame.resized_data)
            if ok:
                result.append(
                    (Detection(tracker.det_category, tracker.det_score, bbox), tracker.det_type)
                )

        return result
