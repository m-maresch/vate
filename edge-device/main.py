import argparse
from typing import Union

from edge_device import EdgeDevice
from track import MultiObjectTracker


def main(videos: Union[str, None], annotations_path: Union[str, None]):
    detection_rate = 6
    object_tracker = MultiObjectTracker()

    edge_device = EdgeDevice(detection_rate, object_tracker)
    edge_device.process(videos, annotations_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="?", help="path to videos, defaults to using the camera")
    parser.add_argument("annotations", nargs="?", help="path to annotations, which are displayed if present")
    args = parser.parse_args()

    main(args.videos, args.annotations)
