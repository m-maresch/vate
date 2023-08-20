from typing import Union, Iterator

import cv2 as cv
import uuid
import glob
import time

from model import Frame, ImageList


def get_frames(video: Union[str, None], images: ImageList, frame_processing_width: int,
               frame_processing_height: int, max_fps: int) -> Iterator[Frame]:
    if video is None:
        video = 'camera'
        capture = cv.VideoCapture(0)
        if not capture.isOpened():
            print("Cannot open capture")
            exit()

        while True:
            frame_id = uuid.uuid4().int
            ret, data = capture.read()
            if not ret:
                print("Cannot receive frame")
                capture.release()
                yield Frame(id=-1, video=video, data=-1, resized_data=-1)
            resized_data = cv.resize(data, (frame_processing_width, frame_processing_height),
                                     interpolation=cv.INTER_LINEAR)
            yield Frame(id=frame_id, video=video, data=data, resized_data=resized_data)
    else:
        frame_id = 0
        for file in sorted(glob.glob(video)):
            if len(images) > 0:
                image = next(filter(lambda img: file.endswith(img.file_name), images))
                frame_id = image.id
            else:
                frame_id = frame_id + 1
            data = cv.imread(file)
            resized_data = cv.resize(data, (frame_processing_width, frame_processing_height),
                                     interpolation=cv.INTER_LINEAR)

            before = time.time()
            yield Frame(id=frame_id, video=video, data=data, resized_data=resized_data)
            after = time.time()

            _wait_for_next_frame(processing_time=after - before, max_fps=max_fps)

        yield Frame(id=-1, video=video, data=-1, resized_data=-1)


def _wait_for_next_frame(processing_time: float, max_fps: int):
    remaining_time = (1 / max_fps) - processing_time
    if remaining_time > 0.0:
        time.sleep(remaining_time)
