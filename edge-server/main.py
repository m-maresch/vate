import asyncio
import json
import time
import zmq.asyncio
from typing import List

from bbox import xyxy2xywh
from model import Detection, DetectionResponse, RawDetection
from services import get_object_detector

context = zmq.asyncio.Context()
socket = context.socket(zmq.ROUTER)
socket.bind("tcp://*:8000")

poll = zmq.asyncio.Poller()
poll.register(socket, zmq.POLLIN)

object_detector = get_object_detector()


async def detect(frame_id: int, video: str, frame) -> DetectionResponse:
    (det_type, detections) = await object_detector.detect_objects(video, frame)
    return DetectionResponse(
        frame_id=frame_id,
        type=det_type.name,
        detections=_convert_detections(detections)
    )


def _convert_detections(detections: List[RawDetection]) -> List[Detection]:
    return [Detection(
        bbox=xyxy2xywh(detection.bbox),
        score=int(detection.score * 100),
        category=detection.class_name
    ) for detection in detections]


async def main():
    while True:
        sockets = dict(await poll.poll(1000))
        if sockets:
            identity = await socket.recv_string()

            metadata = await socket.recv_string()
            metadata_json = json.loads(metadata)
            frame_id = metadata_json['frame_id']
            video = metadata_json['video']

            print(f"Identity: {identity}, frame id: {frame_id}, video: {video}")

            frame = await socket.recv(flags=0, copy=False, track=False)

            start_detect = time.time()
            detections = await detect(frame_id, video, frame)
            print(f"Detect took: {time.time() - start_detect}")

            await socket.send_string(identity, zmq.SNDMORE)
            await socket.send_string(detections.json())


if __name__ == '__main__':
    event_loop = asyncio.get_event_loop()
    event_loop.run_until_complete(main())

    pending = asyncio.Task.all_tasks()
    event_loop.run_until_complete(asyncio.gather(*pending))
