import argparse
import json
import sys

sys.path.append('..')

from evaluation import evaluate_detections
from model import DetectionView, DetectionType


def main(annotations_path: str):
    with open('all_detections.json') as f:
        loaded_detections = json.load(f)
        detections = [
            DetectionView(detection['frame_id'], detection['x'], detection['y'], detection['w'], detection['h'],
                          detection['score'], detection['category'], DetectionType[detection['type'].split('.')[1]],
                          detection['tracked'])
            for detection in loaded_detections
        ]

        evaluate_detections(detections, annotations_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations")
    args = parser.parse_args()

    main(args.annotations)
