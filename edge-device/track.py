import cv2 as cv
from typing import List

from model import TrackerRecord, Frame, Detection, DetectionType, DetectionsWithTypes


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

        tracker = cv.legacy.TrackerMOSSE_create()
        try:
            tracker.init(frame.edge_data, detection.bbox)
            self.trackers.append(
                TrackerRecord(tracker, detection.bbox, detection.score, detection.category, det_type)
            )
        except cv.error as e:
            print(f"Failed to init a tracker: {e}")

    def track_objects_until_current(self, frames_until_current: List[Frame],
                                    current_frame: Frame, decay: float = 0.9) -> DetectionsWithTypes:
        for frame in frames_until_current:
            self.track_objects(frame, decay)

        return self.track_objects(current_frame, decay)

    def track_objects(self, frame: Frame, decay: float = 0.8) -> DetectionsWithTypes:
        result = []
        for tracker in self.trackers:
            if tracker.raw_tracker is None:
                new_score = int(decay * tracker.det_score)
                tracker.det_score = new_score
                result.append(
                    (Detection(tracker.det_category, new_score, list(map(int, tracker.det_bbox))), tracker.det_type)
                )
            else:
                ok, bbox = tracker.raw_tracker.update(frame.edge_data)
                if ok:
                    result.append(
                        (Detection(tracker.det_category, tracker.det_score, list(map(int, bbox))), tracker.det_type)
                    )
                else:
                    new_score = int(decay * tracker.det_score)
                    tracker.det_score = new_score
                    result.append(
                        (Detection(tracker.det_category, new_score, list(map(int, tracker.det_bbox))), tracker.det_type)
                    )

        return [(detection, det_type) for detection, det_type in result if detection.score > 0]
