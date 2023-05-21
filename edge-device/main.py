import argparse
from typing import Union

from edge_device import EdgeDevice
from track import MultiObjectTracker


def main(video: Union[str, None], annotations_path: Union[str, None]):
    detection_rate = 6
    object_tracker = MultiObjectTracker()

    edge_device = EdgeDevice(video, annotations_path, detection_rate, object_tracker)
    edge_device.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", help="path to video, defaults to using the camera")
    parser.add_argument("annotations", nargs="?", help="path to annotations, which are displayed if present")
    args = parser.parse_args()

    main(args.video, args.annotations)
