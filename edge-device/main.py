import argparse
import cv2 as cv
import json
import requests
import uuid
import time
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

font = cv.FONT_HERSHEY_SIMPLEX

categories = [
    {"id": 0, "name": "pedestrian"},
    {"id": 1, "name": "people"},
    {"id": 2, "name": "bicycle"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "van"},
    {"id": 5, "name": "truck"},
    {"id": 6, "name": "tricycle"},
    {"id": 7, "name": "awning-tricycle"},
    {"id": 8, "name": "bus"},
    {"id": 9, "name": "motor"}
]


def main(video, annotations_path):
    capture = cv.VideoCapture(video)
    if not capture.isOpened():
        print("Cannot open capture")
        exit()

    annotations = []
    if _annotations_available(video, annotations_path):
        annotations = _load_annotations(video, annotations_path)

    frame_count = 0
    tracker_failures = 0
    detection_rate = 6

    all_detections = []
    all_annotations = []
    detections_to_display = []
    trackers = []

    prev_frame_at = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Cannot receive frame")
            break

        if frame_count % detection_rate == 0:
            detections_to_display = []
            trackers = []
            detections = _detect_objects(frame)
            for detection in detections:
                (bbox, score, category) = _to_bbox_score_category(detection)
                (x, y, w, h) = bbox
                all_detections.append((x, y, w, h, score, category, False))
                if _is_significant(score):
                    detections_to_display.append((x, y, w, h, score, category, False))

                    tracker = _create_single_tracker()
                    tracker.init(frame, bbox)
                    trackers.append((tracker, score, category))
        else:
            tracking_result = _track_objects(trackers, frame)
            if not tracking_result:
                tracker_failures += 1

            all_detections.extend(tracking_result)
            detections_to_display.extend(tracking_result)

        frame_at = time.time()
        fps = int(1 / (frame_at - prev_frame_at))
        prev_frame_at = frame_at

        frame_count += 1
        for detection in detections_to_display:
            _display(frame, detection)

        if _annotations_available(video, annotations_path) and frame_count < len(annotations):
            for annotation in annotations[frame_count]:
                (x, y, w, h) = annotation['bbox']
                all_annotations.append((x, y, w, h, annotation['category_id']))
                _display_annotation(frame, (x, y, w, h, annotation['category_id']))

        cv.putText(frame, f"{fps} FPS", (50, 50), font, 0.8, (0, 255, 0), 2)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    if _annotations_available(video, annotations_path):
        _compute_map(all_detections, all_annotations)

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
    return score >= 0


def _annotations_available(video, annotations_path):
    return video != 0 and annotations_path != 0


def _load_annotations(video, annotations_path):
    with open(annotations_path) as annotations_file:
        annotations_json = json.load(annotations_file)
        video_name = video.split("/")[-2]
        images = [image for image in annotations_json['images'] if video_name in image['file_name']]
        images_sorted = sorted(images, key=lambda i: i['file_name'])
        image_ids = [image['id'] for image in images_sorted]

        annotations = []
        for image_id in image_ids:
            image_annotations = [annotation for annotation in annotations_json['annotations']
                                 if annotation['image_id'] == image_id]
            annotations.append(image_annotations)
        return annotations


def _track_objects(trackers, frame):
    result = []
    for (tracker, score, category) in trackers:
        ok, bbox = tracker.update(frame)
        if ok:
            (x, y, w, h) = map(int, bbox)
            result.append((x, y, w, h, score, category, True))

    return result


def _display(frame, detection):
    (x, y, w, h, score, category_id, tracked) = detection
    if not tracked:
        category_name = _category_name(category_id)
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv.putText(frame, f"{category_name} {score}%", (x, y - 10), font, 0.5, (255, 0, 0), 2)
    else:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 180, 0), 2)


def _display_annotation(frame, annotation):
    (x, y, w, h, category_id) = annotation
    category_name = _category_name(category_id)
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv.putText(frame, f"{category_name}", (x, y - 10), font, 0.5, (0, 255, 0), 1)


def _category_name(category_id):
    category = next(filter(lambda cat: cat["id"] == category_id, categories))
    return category["name"]


def _create_single_tracker():
    return cv.legacy.TrackerKCF_create()


def _compute_map(detections, annotations):
    preds = [
        dict(
            index=ind,
            boxes=[[x, y, w, h]],
            scores=[score],
            labels=[category_id]
        ) for ind, (x, y, w, h, score, category_id, _) in enumerate(detections)
    ]

    target = [
        dict(
            boxes=torch.tensor([[x, y, w, h]]),
            labels=torch.tensor([category_id]),
        ) for (x, y, w, h, category_id) in annotations
    ]

    diff = len(preds) - len(target)
    if diff > 0:
        sorted_preds = sorted(preds, key=lambda pred: pred['scores'])
        worst_preds = sorted_preds[:diff]
        worst_pred_ids = [worst_pred['index'] for worst_pred in worst_preds]
        preds = [
            dict(
                boxes=torch.tensor(pred['boxes']),
                scores=torch.tensor(pred['scores']),
                labels=torch.tensor(pred['labels'])
            ) for pred in preds if pred['index'] not in worst_pred_ids
        ]
    if diff < 0:
        for _ in range(abs(diff)):
            preds.append(
                dict(
                    boxes=torch.tensor([[0, 0, 0, 0]]),
                    scores=torch.tensor([0]),
                    labels=torch.tensor([10])
                )
            )

    metric = MeanAveragePrecision(box_format='xywh')
    metric.update(preds, target)
    computed = metric.compute()
    print(computed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default=0, help="path to video, defaults to using the camera")
    parser.add_argument("annotations", nargs="?", default=0, help="path to annotations, which are displayed if present")
    args = parser.parse_args()

    main(args.video, args.annotations)
