import cv2 as cv
import dataclasses
import glob
import json
import numpy as np
import time
from typing import Union, List, Tuple

from annotation import annotations_available, load_annotations
from bbox import scale
from category import to_category_id
from detection import EdgeCloudObjectDetector
from display import display_detection, display_annotation, display_fps
from evaluation import evaluate_detections
from frame import get_frames
from model import DetectionView, AnnotationsByImage, Frame, Detection, DetectionType, DetectionsWithTypes, Dimensions, \
    Image
from track import MultiObjectTracker


class EdgeDevice:
    dimensions: Dimensions
    max_fps: int
    detection_rate: int
    object_tracker: MultiObjectTracker
    object_detector: EdgeCloudObjectDetector
    sync: bool

    all_detections: List[DetectionView]
    all_fps: List[float]

    def __init__(self, dimensions: Dimensions, max_fps: int, detection_rate: int, object_tracker: MultiObjectTracker,
                 object_detector: EdgeCloudObjectDetector, sync: bool):
        self.dimensions = dimensions
        self.max_fps = max_fps
        self.detection_rate = detection_rate
        self.object_tracker = object_tracker
        self.object_detector = object_detector
        self.sync = sync

        self.all_detections = []
        self.all_fps = []

    def process(self, videos: Union[str, None], annotations_path: Union[str, None]):
        items = []
        multiple_videos = False

        if videos is not None:
            items = glob.glob(videos)
        else:
            self._process_camera()

        if len(items) > 0:
            if items[0].endswith(".jpg"):
                video = videos
                self._process_video(video, annotations_path)
            else:
                multiple_videos = True

        if multiple_videos:
            self._process_multiple_videos(videos, annotations_path)

        if annotations_available(videos, annotations_path):
            evaluate_detections(self.all_detections, annotations_path)

            print("Creating all_detections.json")
            with open('all_detections.json', 'w') as f:
                json.dump([dataclasses.asdict(detection) for detection in self.all_detections], f, default=str)

        cv.destroyAllWindows()

        fps_avg = int(sum(self.all_fps) / len(self.all_fps))
        fps_bins = np.append(np.arange(self.max_fps + 1), np.inf)
        fps_histogram = np.histogram(self.all_fps, bins=fps_bins)

        print(f"Total FPS average: {fps_avg}")
        print(f"Total FPS histogram: {fps_histogram} (includes {sum(fps_histogram[0])}/{len(self.all_fps)})")

    def _process_camera(self):
        self._process_video(None, None)

    def _process_multiple_videos(self, videos: str, annotations_path: Union[str, None]):
        mAP_per_video = []
        mAP_50_per_video = []
        fps_per_video = []

        for video in glob.glob(videos):
            self.object_tracker.reset_objects()
            self.object_detector.reset()

            result, fps_records = self._process_video(f"{video}/*", annotations_path)

            if annotations_available(video, annotations_path):
                mAP, mAP_50 = evaluate_detections(result, annotations_path)
                mAP_per_video.append(mAP)
                mAP_50_per_video.append(mAP_50)

            fps = int(sum(fps_records) / len(fps_records))
            print(f"FPS average: {fps}")
            fps_per_video.append(fps)

        print()

        if mAP_per_video:
            print(f"All mAPs: {mAP_per_video}")
        if mAP_50_per_video:
            print(f"All mAP_50s: {mAP_50_per_video}")
        print(f"All FPS averages: {fps_per_video}")

        print()

    def _process_video(self, video: Union[str, None],
                       annotations_path: Union[str, None]) -> Tuple[List[DetectionView], List[float]]:
        frame_count = 0
        prev_frame_at = 0.0
        first_frame = True

        fps_records: List[float] = []

        edge_detection_start = 0.0
        edge_detection_times = []

        frames_until_current: List[Frame] = []
        reset_frames_until_current = False

        current_detections: DetectionsWithTypes = []
        result: List[DetectionView] = []

        images: List[Image] = []
        annotations: AnnotationsByImage = dict()
        if annotations_available(video, annotations_path):
            (images, annotations) = load_annotations(video, annotations_path)

        frames = get_frames(video, images, self.dimensions, self.max_fps)
        while True:
            start = time.time()

            frame = next(frames)
            if frame.id == -1:
                break

            tracked = False
            reset_tracker = False

            if self.sync:
                if frame_count % self.detection_rate == 0:
                    edge_detection_start = time.time()
                    detections = self.object_detector.request_edge_detections_sync(frame, current_detections)
                    edge_detection_times.append(time.time() - edge_detection_start)

                    current_detections = detections
                    reset_tracker = True
                else:
                    tracked = True
                    current_detections = self.object_tracker.track_objects(frame)
            else:
                if not first_frame:
                    ok = self.object_detector.request_edge_detections_async(frame)
                    if ok:
                        reset_frames_until_current = True

                        edge_detection_times.append(time.time() - edge_detection_start)
                        edge_detection_start = time.time()

                if reset_frames_until_current:
                    frames_until_current.clear()
                    reset_frames_until_current = False
                frames_until_current.append(frame)

                if first_frame:
                    detections = self.object_detector.request_edge_detections_sync(frame, current_detections)

                    current_detections = detections
                    reset_tracker = True
                else:
                    block = len(frames_until_current) >= self.detection_rate
                    detections = self.object_detector.get_edge_detections(current_detections, block)

                    if detections is None:
                        tracked = True
                        current_detections = self.object_tracker.track_objects(frame)
                    else:
                        self.object_tracker.reset_objects()
                        for (detection, det_type) in detections:
                            self.object_tracker.add_object(
                                frames_until_current[0],
                                detection,
                                det_type
                            )

                        current_detections = self.object_tracker.track_objects_until_current(
                            frames_until_current[1:-1],
                            frame,
                            decay=1.0
                        )

            self.object_detector.record(frame)

            if frame_count % 5 == 0:
                self.object_detector.process_cloud_detections(frame)

            detections = self.object_detector.get_cloud_detections(current_detections)
            if detections is not None:
                current_detections = detections
                reset_tracker = True

            if reset_tracker:
                self.object_tracker.reset_objects()
                for (detection, det_type) in current_detections:
                    self.object_tracker.add_object(frame, detection, det_type)

            current_det_views = self._convert_to_views(current_detections, frame, tracked=tracked)
            result.extend(current_det_views)

            frame_at = time.time()
            fps = 1 / (frame_at - prev_frame_at)
            fps_records.append(fps)
            prev_frame_at = frame_at

            frame_count += 1
            first_frame = False

            for current_detection in current_det_views:
                display_detection(frame.data, current_detection)

            if annotations_available(video, annotations_path):
                for annotation in annotations[frame.id]:
                    display_annotation(frame.data, annotation)

            display_fps(frame.data, int(fps))

            end = time.time()

            cv.imshow('frame', frame.data)
            if cv.waitKey(1) == ord('q'):
                break

            print(f"Frame took: {end - start}s, with wait: {time.time() - start}s")

        edge_detection_times = edge_detection_times if self.sync else edge_detection_times[1:]
        average_edge_detection_time = sum(edge_detection_times) / len(edge_detection_times)
        print(f"Edge detections took: {edge_detection_times} (average: {average_edge_detection_time}s)")

        self.all_detections.extend(result)
        self.all_fps.extend(fps_records)

        return result, fps_records

    def _convert_to_views(self, detections: DetectionsWithTypes, frame: Frame, tracked: bool) -> List[DetectionView]:
        frame_height, frame_width, _ = frame.data.shape
        frame_width_scale = frame_width / self.dimensions.edge_processing_width
        frame_height_scale = frame_height / self.dimensions.edge_processing_height

        return [
            _to_view(detection, det_type, frame.id, frame_width_scale, frame_height_scale, tracked=tracked)
            for (detection, det_type) in detections
        ]


def _to_view(detection: Detection, det_type: DetectionType, frame_id: int, scale_width: float, scale_height: float,
             tracked: bool) -> DetectionView:
    (x, y, w, h) = scale(detection.bbox, scale_width, scale_height)
    category_id = to_category_id(detection.category)
    return DetectionView(frame_id, x, y, w, h, detection.score, category_id, det_type, tracked)
