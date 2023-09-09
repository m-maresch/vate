import cv2 as cv
import glob
import time
from typing import Union, List, Tuple

from annotation import annotations_available, load_annotations
from bbox import scale
from category import to_category_id
from detection import EdgeCloudObjectDetector
from display import display_detection, display_annotation, display_fps
from evaluation import evaluate_detections
from frame import frame_change_detected, get_frames
from model import DetectionView, ImageList, AnnotationsByImage, Frame, Detection, DetectionType
from track import MultiObjectTracker


class EdgeDevice:
    frame_processing_width: int
    frame_processing_height: int
    max_fps: int
    detection_rate: int
    object_tracker: MultiObjectTracker
    object_detector: EdgeCloudObjectDetector
    synchronous: bool

    all_detections: List[DetectionView]
    all_fps: List[float]

    skipped_frames: int

    def __init__(self, frame_processing_width: int, frame_processing_height: int, max_fps: int,
                 detection_rate: int, object_tracker: MultiObjectTracker, object_detector: EdgeCloudObjectDetector,
                 synchronous: bool):
        self.frame_processing_width = frame_processing_width
        self.frame_processing_height = frame_processing_height
        self.max_fps = max_fps
        self.detection_rate = detection_rate
        self.object_tracker = object_tracker
        self.object_detector = object_detector
        self.synchronous = synchronous

        self.all_detections = []
        self.all_fps = []

        self.skipped_frames = 0

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

        cv.destroyAllWindows()

        print(f"Total FPS average: {int(sum(self.all_fps) / len(self.all_fps))}")
        print(f"{self.skipped_frames} frames skipped")

    def _process_camera(self):
        self._process_video(None, None)

    def _process_multiple_videos(self, videos: str, annotations_path: Union[str, None]):
        mAP_per_video = []
        mAP_50_per_video = []
        fps_per_video = []

        for video in glob.glob(videos):
            self.object_tracker.reset_objects()
            self.object_detector.reset()

            result, fps = self._process_video(f"{video}/*", annotations_path)

            if annotations_available(video, annotations_path):
                mAP, mAP_50 = evaluate_detections(result, annotations_path)
                mAP_per_video.append(mAP)
                mAP_50_per_video.append(mAP_50)

            print(f"FPS average: {fps}")
            fps_per_video.append(fps)

        print()

        if mAP_per_video:
            print(f"All mAPs: {mAP_per_video}")
            print(f"Average mAP: {sum(mAP_per_video) / len(mAP_per_video)}")
        if mAP_50_per_video:
            print(f"All mAP_50s: {mAP_50_per_video}")
            print(f"Average mAP_50: {sum(mAP_50_per_video) / len(mAP_50_per_video)}")
        print(f"All FPS averages: {fps_per_video}")

        print()

    def _process_video(self, video: Union[str, None],
                       annotations_path: Union[str, None]) -> Tuple[List[DetectionView], int]:
        images: ImageList = []
        annotations: AnnotationsByImage = dict()
        if annotations_available(video, annotations_path):
            (images, annotations) = load_annotations(video, annotations_path)

        frames = get_frames(video, images, self.frame_processing_width, self.frame_processing_height, self.max_fps)
        prev_frame: Union[Frame, None] = None
        first_frame = True

        frames_until_current: List[Frame] = []
        reset_frames_until_current = False

        detections_to_display: List[DetectionView] = []

        frame_count = 0
        prev_frame_at = 0.0
        fps_records: List[float] = []

        result: List[DetectionView] = []
        while True:
            start = time.time()

            frame = next(frames)
            if frame.id == -1:
                break

            frame_changed = True
            if prev_frame is not None:
                frame_changed = frame_change_detected(frame, prev_frame)

            if not frame_changed:
                print('Skipping frame due to no changes')
                self.skipped_frames += 1
            elif frame_count % self.detection_rate == 0:
                if reset_frames_until_current:
                    frames_until_current.clear()
                    reset_frames_until_current = False

                detected = True
                (det_type, detections) = self.object_detector.detect_objects(frame,
                                                                             wait=first_frame or self.synchronous)

                tracking_result = []
                if det_type is None:
                    detected = False
                    tracking_result = self.object_tracker.track_objects(frame)
                elif self.synchronous:
                    detections_to_display = []
                    self.object_tracker.reset_objects()
                    frames_until_current.clear()  # not used if synchronous, clear here to not fill up memory

                    for detection in detections:
                        self.object_tracker.add_object(frame, detection, det_type)
                        tracking_result.append((detection, det_type))
                else:
                    detections_to_display = []
                    self.object_tracker.reset_objects()

                    if frames_until_current:
                        for detection in detections:
                            self.object_tracker.add_object(
                                frames_until_current[0],
                                detection,
                                det_type
                            )

                        tracking_result = self.object_tracker.track_objects_until_current(frames_until_current[1:],
                                                                                          frame)
                        reset_frames_until_current = True
                    else:
                        tracking_result = [(detection, det_type) for detection in detections]

                det_views = self._convert_to_views(tracking_result, frame, tracked=not detected)
                result.extend(det_views)
                detections_to_display.extend(det_views)
            else:
                tracking_result = self.object_tracker.track_objects(frame)
                det_views = self._convert_to_views(tracking_result, frame, tracked=True)
                result.extend(det_views)
                detections_to_display.extend(det_views)

            frame_at = time.time()
            fps = 1 / (frame_at - prev_frame_at)
            fps_records.append(fps)
            prev_frame = frame
            prev_frame_at = frame_at

            frame_count += 1
            frames_until_current.append(frame)
            self.object_detector.record(frame)
            first_frame = False

            for received_detection in detections_to_display:
                display_detection(frame.data, received_detection)

            if annotations_available(video, annotations_path):
                for annotation in annotations[frame.id]:
                    display_annotation(frame.data, annotation)

            display_fps(frame.data, int(sum(fps_records) / len(fps_records)))

            end = time.time()

            cv.imshow('frame', frame.data)
            if cv.waitKey(1) == ord('q'):
                break

            print(f"Frame took: {end - start}s, with wait: {time.time() - start}s")

        self.all_detections.extend(result)
        self.all_fps.extend(fps_records)

        fps = int(sum(fps_records) / len(fps_records))
        return result, fps

    def _convert_to_views(self, detections: List[Tuple[Detection, DetectionType]], frame: Frame,
                          tracked: bool) -> List[DetectionView]:
        frame_height, frame_width, _ = frame.data.shape
        frame_width_scale = frame_width / self.frame_processing_width
        frame_height_scale = frame_height / self.frame_processing_height

        return [
            _to_view(detection, det_type, frame.id, frame_width_scale, frame_height_scale, tracked=tracked)
            for (detection, det_type) in detections
        ]


def _to_view(detection: Detection, det_type: DetectionType, frame_id: int, scale_width: float, scale_height: float,
             tracked: bool) -> DetectionView:
    (x, y, w, h) = scale(detection.bbox, scale_width, scale_height)
    category_id = to_category_id(detection.category)
    return DetectionView(frame_id, x, y, w, h, detection.score, category_id, det_type, tracked)
