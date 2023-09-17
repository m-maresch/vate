import argparse
from typing import Union

from cloud_server import CloudServer
from detection import EdgeCloudObjectDetector
from edge_device import EdgeDevice
from edge_server import EdgeServer
from model import Dimensions
from track import MultiObjectTracker


def main(videos: Union[str, None], annotations_path: Union[str, None], detection_rate: int, ipc: bool, sync: bool):
    print(f"Got: detection-rate={detection_rate}, ipc={ipc}, sync={sync}")

    dimensions = Dimensions(
        edge_processing_width=512,
        edge_processing_height=512,
        cloud_processing_width=1333,
        cloud_processing_height=800
    )
    max_fps = 300

    object_tracker = MultiObjectTracker(min_score=70)

    edge_server = EdgeServer(ipc)
    edge_server.connect()

    cloud_server = CloudServer(
        detection_model_url="http://127.0.0.1:9093/predictions/faster_rcnn_visdrone",
        detection_timeout=30
    )

    object_detector = EdgeCloudObjectDetector(edge_server, cloud_server, dimensions, cloud_tracking_min_score=70,
                                              cloud_tracking_stride=3, max_fps=max_fps)
    object_detector.start_cloud_tracking()

    edge_device = EdgeDevice(dimensions, max_fps, detection_rate, object_tracker, object_detector, sync)
    edge_device.process(videos, annotations_path)

    object_detector.stop_cloud_tracking()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="?", help="path to videos, defaults to using the camera")
    parser.add_argument("annotations", nargs="?", help="path to annotations, which are displayed if present")
    parser.add_argument('--detection-rate', nargs="?", type=int, default=5,
                        help="rate for edge-server detection requests")
    parser.add_argument('--ipc', action='store_true', help="use ipc for communication with edge-server")
    parser.add_argument('--sync', action='store_true', help="wait for edge-server detection responses")
    args = parser.parse_args()

    main(args.videos, args.annotations, args.detection_rate, args.ipc, args.sync)
