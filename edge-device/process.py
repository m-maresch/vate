from multiprocessing import Event, Queue
from queue import Empty, Full

from model import DetectionType
from track import MultiObjectTracker


def track_objects_until_current_worker(input_queue: Queue, output_queue: Queue, stop_event: Event):
    while True:
        try:
            (detections, frames_until_current, current_frame, min_score) = input_queue.get(timeout=10)

            if not detections:
                output_queue.put([], timeout=2)

            object_tracker = MultiObjectTracker(min_score=min_score)
            for detection in detections:
                object_tracker.add_object(frames_until_current[0], detection, DetectionType.CLOUD)

            tracking_result = object_tracker.track_objects_until_current(frames_until_current[1:], current_frame)
            output_queue.put([detection for detection, _ in tracking_result], timeout=2)
        except (Empty, Full):
            if stop_event.is_set():
                break

    drain(input_queue)
    drain(output_queue)


def drain(queue: Queue):
    try:
        while not queue.empty():
            queue.get_nowait()
    except Empty:
        pass
