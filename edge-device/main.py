import argparse
import cv2 as cv
import requests
import uuid
import time

font = cv.FONT_HERSHEY_SIMPLEX


def main(path):
    capture = cv.VideoCapture(path)
    if not capture.isOpened():
        print("Cannot open capture")
        exit()

    frame_count = 0
    tracker_failures = 0
    detection_rate = 6

    detections_to_display = []
    multi_tracker = None

    prev_frame_at = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Cannot receive frame")
            break

        if frame_count % detection_rate == 0:
            detections_to_display = []
            multi_tracker = _create_multi_tracker()
            detections = _detect_objects(frame)
            for detection in detections:
                (bbox, score, category) = _to_bbox_score_category(detection)
                if _is_significant(score):
                    (x, y, w, h) = bbox
                    detections_to_display.append((x, y, w, h, score, category))

                    tracker = _create_single_tracker()
                    multi_tracker.add(tracker, frame, bbox)
        else:
            tracking_result = _track_objects(multi_tracker, frame)
            if not tracking_result:
                tracker_failures += 1

            detections_to_display.extend(tracking_result)

        frame_at = time.time()
        fps = int(1 / (frame_at - prev_frame_at))
        prev_frame_at = frame_at

        frame_count += 1
        for detection in detections_to_display:
            _display(frame, detection)

        cv.putText(frame, f"{fps} FPS", (50, 50), font, 0.8, (0, 255, 0), 2)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    capture.release()
    cv.destroyAllWindows()
    print(f"{tracker_failures} tracker failures")


def _detect_objects(frame):
    encoded = cv.imencode(".jpg", frame)[1]
    frame_id = str(uuid.uuid4())
    file = {'file': (f'{frame_id}.jpg', encoded.tobytes(), 'image/jpeg')}
    data = {"id": f"{frame_id}"}
    response = requests.post("http://127.0.0.1:8000/detection", files=file, data=data, timeout=5)
    body = response.json()
    print(f"Got: {body}")
    return body['detections']


def _to_bbox_score_category(detection):
    bbox = detection['bbox']
    score = detection['score']
    category = detection['category']
    return bbox, score, category


def _is_significant(score):
    return score >= 30


def _track_objects(multi_tracker, frame):
    result = []
    ok, bboxes = multi_tracker.update(frame)
    if ok:
        for bbox in bboxes:
            (x, y, w, h) = map(int, bbox)
            result.append((x, y, w, h, None, None))
    return result


def _display(frame, detection):
    (x, y, w, h, score, category) = detection
    if score is not None and category is not None:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv.putText(frame, f"{category} {score}%", (x, y - 10), font, 0.5, (255, 0, 0), 2)
    else:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 180, 0), 2)


def _create_multi_tracker():
    return cv.legacy.MultiTracker_create()


def _create_single_tracker():
    return cv.legacy.TrackerKCF_create()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=0, help="path to video, defaults to using the camera")
    args = parser.parse_args()

    main(args.path)
