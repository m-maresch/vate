import argparse
from typing import Union

from edge_device import EdgeDevice
from edge_server import EdgeServer
from track import MultiObjectTracker


def main(videos: Union[str, None], annotations_path: Union[str, None], ipc: bool, sync: bool):
    print(f"Got: ipc={ipc}, sync={sync}")

    frame_processing_width = 1333
    frame_processing_height = 800
    max_fps = 30

    detection_rate = 3
    object_tracker = MultiObjectTracker()
    edge_server = EdgeServer(ipc)

    edge_device = EdgeDevice(frame_processing_width, frame_processing_height, max_fps, detection_rate,
                             object_tracker, edge_server, sync)
    edge_device.process(videos, annotations_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="?", help="path to videos, defaults to using the camera")
    parser.add_argument("annotations", nargs="?", help="path to annotations, which are displayed if present")
    parser.add_argument('--ipc', action='store_true', help="use ipc for communication with edge-server")
    parser.add_argument('--sync', action='store_true', help="wait for edge-server responses")
    args = parser.parse_args()

    main(args.videos, args.annotations, args.ipc, args.sync)
