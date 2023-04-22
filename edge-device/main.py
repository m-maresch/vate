import argparse
import cv2 as cv
import json
import requests
import uuid
import time
import glob

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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

metrics = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
metric_to_coco_id = {
    'mAP': 0,
    'mAP_50': 1,
    'mAP_75': 2,
    'mAP_s': 3,
    'mAP_m': 4,
    'mAP_l': 5,
}


def main(video, annotations_path):
    images = []
    annotations = dict()
    if _annotations_available(video, annotations_path):
        (images, annotations) = _load_annotations(video, annotations_path)

    frame_count = 0
    tracker_failures = 0
    detection_rate = 6

    all_detections = []
    all_annotations = []
    detections_to_display = []
    trackers = []

    prev_frame_at = 0

    frames = _frames(video, images)
    while True:
        next_frame = next(frames)
        frame_id = next_frame['id']
        if frame_id == -1:
            break
        frame = next_frame['data']

        if frame_count % detection_rate == 0:
            detections_to_display = []
            trackers = []
            detections = _detect_objects(frame_id, frame)
            for detection in detections:
                (bbox, score, category) = _to_bbox_score_category(detection)
                (x, y, w, h) = bbox
                all_detections.append((frame_id, x, y, w, h, score, category, False))
                if _is_significant(score):
                    detections_to_display.append((frame_id, x, y, w, h, score, category, False))

                    tracker = _create_single_tracker()
                    tracker.init(frame, bbox)
                    trackers.append((tracker, score, category))
        else:
            tracking_result = _track_objects(trackers, frame_id, frame)
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

        if _annotations_available(video, annotations_path):
            for annotation in annotations[frame_id]:
                (x, y, w, h) = annotation['bbox']
                all_annotations.append((x, y, w, h, annotation['category_id']))
                _display_annotation(frame, (x, y, w, h, annotation['category_id']))

        cv.putText(frame, f"{fps} FPS", (50, 50), font, 0.8, (0, 255, 0), 2)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    if _annotations_available(video, annotations_path):
        _evaluate_detections(all_detections, annotations_path)

    cv.destroyAllWindows()
    print(f"{tracker_failures} tracker failures")


def _frames(video, images):
    if video == 0:
        capture = cv.VideoCapture(video)
        if not capture.isOpened():
            print("Cannot open capture")
            exit()

        while True:
            frame_id = str(uuid.uuid4())
            ret, data = capture.read()
            if not ret:
                print("Cannot receive frame")
                capture.release()
                yield {'id': -1, 'data': -1}
            yield {'id': frame_id, 'data': data}
    else:
        frame_id = 0
        for file in sorted(glob.glob(video)):
            if len(images) > 0:
                frame_id = next(filter(lambda img: file.endswith(img['file_name']), images))['id']
            else:
                frame_id = frame_id + 1
            data = cv.imread(file)
            yield {'id': frame_id, 'data': data}
        yield {'id': -1, 'data': -1}


def _detect_objects(frame_id, frame):
    encoded = cv.imencode(".jpg", frame)[1]
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

        annotations = dict()
        for image_id in image_ids:
            image_annotations = [annotation for annotation in annotations_json['annotations']
                                 if annotation['image_id'] == image_id]
            annotations[image_id] = image_annotations
        return images_sorted, annotations


def _track_objects(trackers, frame_id, frame):
    result = []
    for (tracker, score, category) in trackers:
        ok, bbox = tracker.update(frame)
        if ok:
            (x, y, w, h) = map(int, bbox)
            result.append((frame_id, x, y, w, h, score, category, True))

    return result


def _display(frame, detection):
    (_, x, y, w, h, score, category_id, tracked) = detection
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


def _evaluate_detections(detections, annotations_path):
    preds = [
        dict(
            image_id=frame_id,
            bbox=[x, y, w, h],
            score=score,
            category_id=category_id
        ) for (frame_id, x, y, w, h, score, category_id, _) in detections
    ]

    img_ids_start = preds[0]['image_id']
    img_ids_end = preds[-1]['image_id']

    coco_gt = COCO(annotations_path)
    coco_dt = coco_gt.loadRes(preds)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.catIds = list(range(0, 10))
    coco_eval.params.imgIds = list(range(img_ids_start, img_ids_end + 1))
    coco_eval.params.maxDets = [100, 500, 1000]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    eval_results = dict()
    for metric in metrics:
        coco_id = metric_to_coco_id[metric]
        stat = coco_eval.stats[coco_id]
        eval_results[metric] = float(f'{stat:.4f}')
    print(f'Bounding box evaluation results: {eval_results}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default=0, help="path to video, defaults to using the camera")
    parser.add_argument("annotations", nargs="?", default=0, help="path to annotations, which are displayed if present")
    args = parser.parse_args()

    main(args.video, args.annotations)
