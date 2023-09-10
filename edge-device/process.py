import time
import zmq

from multiprocessing import Event

from model import DetectionType
from track import MultiObjectTracker


def track_objects_until_current_worker(stop_event: Event):
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind("ipc:///tmp/edge-device/0")

    while True:
        if socket.poll(10000, zmq.POLLIN):
            identity = socket.recv_string()

            obj = socket.recv_pyobj()
            detections = obj['detections']
            frames_until_current = obj['frames_until_current']
            current_frame = obj['current_frame']
            min_score = obj['min_score']

            if not detections:
                socket.send_string(identity, zmq.SNDMORE)
                socket.send_pyobj([], protocol=-1)
                continue

            print(f"Cloud tracking num frames: {len(frames_until_current) + 1}")
            start = time.time()

            object_tracker = MultiObjectTracker(min_score=min_score)
            for detection in detections:
                object_tracker.add_object(frames_until_current[0], detection, DetectionType.CLOUD)

            tracking_result = object_tracker.track_objects_until_current(frames_until_current[1:], current_frame)

            end = time.time()
            print(f"Cloud tracking took: {end - start}s")

            socket.send_string(identity, zmq.SNDMORE)
            socket.send_pyobj([detection for detection, _ in tracking_result], protocol=-1)
        else:
            if stop_event.is_set():
                break
