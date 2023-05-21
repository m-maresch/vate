import cv2 as cv

from model import DetectionView, TrackerRecord, Frame


class MultiObjectTracker:
    trackers: list[TrackerRecord]

    def __init__(self):
        self.trackers = []

    def reset_objects(self):
        self.trackers = []

    def add_object(self, frame: Frame, detection: DetectionView):
        tracker = cv.legacy.TrackerKCF_create()
        try:
            tracker.init(frame.data, [detection.x, detection.y, detection.w, detection.h])
            self.trackers.append(TrackerRecord(tracker, detection.score, detection.category, detection.type))
        except cv.error as e:
            print(f"Failed to init a tracker: {e}")

    def track_objects(self, frame: Frame) -> list[DetectionView]:
        result = []
        for tracker in self.trackers:
            ok, bbox = tracker.raw_tracker.update(frame.data)
            if ok:
                (x, y, w, h) = map(int, bbox)
                result.append(DetectionView(
                    frame.id, x, y, w, h, tracker.det_score, tracker.det_category, tracker.det_type, tracked=True
                ))

        return result
