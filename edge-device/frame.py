from typing import Union, Iterator

import cv2 as cv
import uuid
import glob

from model import Frame, ImageList

DONE_FRAME = Frame(id=-1, data=-1)


def get_frames(video: Union[str, None], images: ImageList) -> Iterator[Frame]:
    if video is None:
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
                yield DONE_FRAME
            yield Frame(id=frame_id, data=data)
    else:
        frame_id = 0
        for file in sorted(glob.glob(video)):
            if len(images) > 0:
                image = next(filter(lambda img: file.endswith(img.file_name), images))
                frame_id = image.id
            else:
                frame_id = frame_id + 1
            data = cv.imread(file)
            yield Frame(id=frame_id, data=data)
        yield DONE_FRAME
